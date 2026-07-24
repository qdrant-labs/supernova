# Multivector (ColBERT / late-interaction MaxSim) ground truth in nova-bf

nova-bf now computes exact top-K ground truth for **multivector** searches —
ColBERT-style late interaction, scored with **MaxSim**:

```
score(query q, doc d) = Σ over q's tokens ( max over d's tokens ( q_tok · d_tok ) )
```

This matches Qdrant's `MultiVectorComparator::MaxSim` (dot by default; cosine
normalizes every token first). It sits alongside the existing `dense` and
`sparse` vector types and reuses the same read-once / share-across-searches
engine — see the module docstring in `compute.py`.

## Config

```yaml
corpus:
  path: s3://…/pubmed/bge-m3
  multivector_column: multivector_embedding   # list<list<float32>>: outer=doc, inner=token
queries:
  path: …/queries
  multivector_column: multivector_embedding
params:
  # MaxSim has TWO memory axes (the per-slice score matrix is
  # block_query_tokens × slice_doc_tokens), so BOTH are tiled:
  multivector_batch_size: 2000     # docs per corpus-row slice
  multivector_query_block: 16      # queries per query-axis tile (whole queries only)
  # …or let a target element-count for that matrix derive both:
  # multivector_token_budget: 50_000_000
searches:
  - name: mv
    vector_type: multivector
    metric: dot            # dot (Qdrant default) | cosine ; euclidean rejected
    k: 1000
```

On-disk format is exactly what `nova-embed` writes
(`MULTIVECTOR_EMBEDDING_TYPE = list<list<float32>>`). A null or zero-token doc
decodes to a zero-width span and becomes a **non-candidate** (`-inf`), the same
way a zero-overlap doc is a non-candidate in the sparse path — Qdrant's inverted
MaxSim never surfaces a doc with no vectors either.

## How it stays inside the existing engine

The whole design principle is that `MultiVectorBatchSlice.score()` returns the
**same `(n_q × n_rows)` score matrix** every other slice type returns, so
`_merge_topk`, the shared per-file loop, filters (uniform + per-query), coalescing,
and `merge` all work unchanged. All the ragged/tiling machinery is hidden inside
`.score()`:

1. `P = Q_block_tokens @ C_slice_tokens.T` — the token×token dot products.
2. **segment-max** over each doc's tokens (`scatter_reduce_(amax)`, columns
   grouped by doc) → `(block_tokens × n_rows)`.
3. **segment-sum** over each query's tokens (`index_add_` straight into the
   output view) → `(block_queries × n_rows)`.

Unlike dense/sparse, the query axis **must** be tiled too: `P` is token×token, so
the naïve all-queries × whole-file product blows up far faster than a pooled
matmul. `multivector_batch_size` bounds the doc-token axis, `multivector_query_block`
the query-token axis. `cosine` L2-normalizes every token before the matmul — a
genuinely separate matmul, not a scalar rescale of the dot result (normalizing
changes each token's per-token argmax), so it's opt-in cost; a `dot`-only run
never pays it.

## Validation

- **Unit tests** (`tests/test_compute_multivector.py`): a hand-computed MaxSim
  pinned exactly; tiling-invariance across `multivector_batch_size ×
  multivector_query_block` (including tile=1 and tile>data) for dot and cosine;
  null/empty docs as non-candidates; a uniform filter; the `make_point_id` path;
  the token-budget auto-derivation; plus decoder tests (null/empty/sliced arrays).
- **Live-Qdrant parity** (`tests/test_qdrant_multivector_parity.py`): a random
  multivector dataset upserted into a Qdrant `MAX_SIM` collection, searched with
  `exact=True`, asserted to match nova-bf's top-K ids **and** scores (boundary-aware
  tolerance) for both dot and cosine. Skips cleanly without docker/`qdrant-client`.

## Profiling & performance

CPU-only box (no GPU here; `torch.cuda.is_available()` is False). Representative
synthetic corpus, `OMP_NUM_THREADS=4`. Times are per-op, isolated.

### Where the time goes (one scored slice: ~1.5k query tokens × ~158k doc tokens → 4000 docs)

| component | time | share |
|---|--:|--:|
| `matmul` `Q @ C.T` | 348 ms | **75%** |
| segment-max `scatter_reduce_(amax)` | 118 ms | 25% |
| segment-sum | 0.3 ms | ~0% |
| index building (`repeat_interleave`) | 0.1 ms | ~0% |

`score()` end-to-end is 467 ms — i.e. **essentially all of it is the matmul plus
the segment-max**, with no measurable overhead on top. End-to-end `run_compute`
(24k docs, 8 files, k=100) is 3.9 s, of which `score()` is 2.9 s; decode
(~65 ms/file) and IO overlap on the reader threads and do not bottleneck.

### The matmul is irreducible, and grows with dim

Exact MaxSim requires every (query-token, doc-token) dot product — there is no
correctness-preserving way to skip any. And its share **grows** at production
embedding width:

| dim D | matmul | segment-max |
|--:|--:|--:|
| 128 | 71% | 29% |
| 384 | 84% | 16% |
| **1024** (BGE-M3 colbert) | **94%** | 6% |

At the real embedding dim the score path is **94% matmul**. BLAS already handles
the transposed-view RHS optimally: `Q @ C.T` (view) == `Q @ C.T.contiguous()`
within noise, and the contiguous form additionally pays a 25 ms transpose — so
there is nothing to win there.

### What was tried for the 6–25% segment-max, and rejected

| alternative | result |
|---|---|
| `torch.segment_reduce(max)` (contiguous segments) | **3× slower** on CPU (350 vs 116 ms) |
| pad-to-max-tokens + `.amax(dim=2)` | **3× slower** on CPU (365 vs 118 ms) **and** ~1.2 GB scratch |

`scatter_reduce_(amax)` is the fastest CPU primitive for this ragged reduction,
so it stays.

### The change that was landed

**Segment-sum via `index_add_` written straight into the output view**, instead
of allocating a separate `(block_queries × n_rows)` buffer and copying it in.

- Verified bit-equivalent to the previous `scatter_reduce_(sum)` path.
- CPU time: **neutral** (segment-sum is <1% of the total; the component itself is
  ~7% faster, 0.246 → 0.229 ms — noise end-to-end).
- **The reason it's worth it is memory, for the GPU target**: at scale
  (`n_q`=100k, a large corpus batch) that eliminated temporary equals the
  output-slice size — a multi-GB transient on the GPU. Removing it gives the
  tiling knobs more headroom.

### Honest bottom line

On this CPU box there is **no meaningful speedup available**: the path is
matmul-bound (irreducible exact-MaxSim FLOPs, optimally laid out) and the one
non-trivial auxiliary reduction already uses the fastest available primitive.
The only landed change is a memory reduction that is time-neutral here.

### Likely GPU-specific findings (NOT validated on this box)

- The matmul will collapse to a small fraction on a GPU (cuBLAS / tensor-core
  friendly), which will make the **segment-max the dominant cost on GPU**.
- `scatter_reduce_(amax)` on GPU uses atomics with per-doc contention
  (~tokens/doc-way). The **pad-to-max + dense `.amax(dim)`** variant — which
  loses badly on CPU — has *no atomic contention* and may well **win on GPU**.
  It is a good experiment to run on real hardware; it was deliberately **not**
  landed here because it regresses 3× on CPU (the path tests and small runs use)
  and cannot be validated without a GPU.
- Prefer the **largest** `multivector_query_block` / `multivector_batch_size`
  that fits VRAM: smaller query blocks add ~16% overhead (measured at block=4)
  from extra Python-loop iterations and kernel launches, with no memory benefit
  beyond the bound they enforce.
