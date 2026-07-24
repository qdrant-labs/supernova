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

## Known, benign divergence from Qdrant (cosine, sub-normal magnitudes)

Under `cosine`, nova-bf L2-normalizes every token down to a `1e-12` floor —
the mathematically exact, scale-invariant cosine. Qdrant applies a **low-norm
guard**: a token vector whose norm is above ~`1e-3` is normalized as expected,
but below that Qdrant leaves it *unnormalized* and scores the raw dot instead
(verified by a magnitude sweep on a live `MAX_SIM` cosine collection: a doc
equal to the query direction scores `≈1.0` down to magnitude `1e-3`, then
`≈magnitude` from `1e-4` on down). So for any token with norm in roughly
`(0, 1e-3)` the two engines diverge — nova gives the true cosine, Qdrant gives
~0.

This never manifests on real data: dense/ColBERT token vectors have O(1)
magnitudes, far above the guard. An *exactly*-zero token agrees (both → 0).
`dot` is unaffected (no normalization). nova deliberately does **not** replicate
Qdrant's threshold — it's undocumented, version-specific, and matching it would
make nova less correct for the realistic case. The unit test
`test_cosine_is_scale_invariant` pins nova's chosen semantics.

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

### GPU validation — live A10G run (measured, not hypothesized)

Launched a single **g5.xlarge (NVIDIA A10G, 24 GB)** via SkyPilot, CUDA 13.0 /
`torch 2.13.0+cu130`, ran the BRANCH (synced workdir, `PYTHONPATH=src` — never
master), then tore the instance down.

**Correctness on GPU — all green:** the 23 multivector unit tests (whose
`run_compute` executes on `cuda`), the **2 live-Qdrant `MAX_SIM` parity tests**
(dot + cosine, exact search), and the profiler's own dot+cosine × tilings
spot-check (including a zero-token doc) all pass on the real GPU + CUDA path.

**Component split at D=1024 — the CPU/GPU flip the hypotheses predicted:**

| component | CPU (D=1024) | GPU A10G (D=1024) |
|---|--:|--:|
| matmul `Q @ C.T` | 94% | **65.7%** (13.5 ms) |
| segment-max `scatter_reduce(amax)` | 6% | **34.1%** (7.0 ms) |
| segment-sum `index_add_` | ~0% | 0.2% (0.04 ms) |

cuBLAS collapses the matmul's share from 94% → 66%, so — as predicted —
**segment-max becomes the prominent secondary cost on GPU** (34%).

**The `pad-to-max + dense .amax` hypothesis: CONFIRMED on GPU.** It ran **6.03 ms
vs scatter_reduce's 7.02 ms — ~14% faster** on the segment-max (bit-equivalent),
the exact *opposite* of CPU (where it lost 3×). No atomic contention on GPU is
the reason. **Still not landed as the default**, though: the win is ~14% of a
34% component (~5% end-to-end), and the padded buffer is `block_q_tokens ×
n_rows × max_tokens_per_doc` — fine here (`max_tok`=49, mean 40) but a real
corpus with a long doc-length tail (mean 80, max 512) blows that buffer up ~6×
and wastes the amax on mostly-`-inf` padding. `scatter_reduce` stays the robust
default (no skew blowup, works on CPU + GPU); pad+amax is a good **opt-in / adaptive**
follow-up for GPU runs on tight token-length distributions.

**The removed host-sync guard — validated:** the `bool((counts==0).any())` guard
(dropped in Round 3) cost **0.037 ms/call vs 0.026 ms/call** unconditional
(~30% on that op) on GPU, confirming it was a real per-query-block device→host
sync. Neutral on CPU, small but real win on GPU — the reason it was worth removing.

**Tiling:** `query_block=None` (45 ms) is fastest; smaller blocks are marginally
slower (48 ms at block=8) — same as CPU. The knobs are for memory-bounding only;
prefer the largest that fits VRAM.

### GPU optimization sweep — realistic scale (A10G), what to land

A second A10G run benchmarked the two candidate optimizations at realistic
ColBERT scale (D=1024, ~356k doc-tokens/slice), measuring speed, memory, and —
critically for a GT tool — whether they preserve the ranking / Qdrant parity.

**(a) Matmul precision — the 66% lever. TF32 landed as opt-in; bf16 rejected.**

| matmul | time | speedup | median rel err | top-100 vs f32 | Qdrant parity |
|---|--:|--:|--:|--:|--:|
| f32 (default) | 41.9 ms | 1.0× | — | — | ✅ |
| **TF32** | 24.0 ms | **1.75×** | 2.9e-4 | **100/100** | ✅ **still passes** |
| bf16 | 15.9 ms | 2.63× | 2.9e-3 | 100/100 | (not run) |

TF32 is 1.75× on the matmul (~1.4× on the score path) and — decisively — the
**live-Qdrant MaxSim parity test still passes with TF32 enabled**, with zero
top-100 ranking change. It is now exposed as **`params.allow_tf32`** (default
**off** — GT stays bit-exact f32; on = the measured speedup at ~3e-4 relative
error). bf16 is faster still but its `max|Δ|` was 0.75 (8-bit mantissa) — too
coarse to trust for exact GT, so it is not exposed.

**(b) Segment-max `pad+amax` — rejected.** The earlier "14% faster on GPU"
held only at an artificially small `max_tok` (49). At realistic doc-length
distributions it does not survive:

| distribution | max_tok | scatter_reduce | pad+amax |
|---|--:|--:|--:|
| tight (mean 120) | 168 | 26.8 ms / 8.1 GB | 25.7 ms / 10.2 GB (1.04×, 1.3× mem) |
| **skewed (mean 80, 5% tail→512)** | 512 | 20.7 ms / 7.6 GB | **37.9 ms / 13.9 GB (0.55×, 1.8× mem)** |

Real corpora have a doc-length tail, so `pad+amax` is a wash at best and **1.8×
slower + 1.8× more memory** under skew (it pads every doc to the longest and
`amax`es over mostly-`-inf`). `scatter_reduce` stays the default; `pad+amax` is
**not** landed.

Net: the one landed GPU optimization is the opt-in `allow_tf32` knob (~1.4× on
the score path when enabled), plus the already-landed host-sync removal.
