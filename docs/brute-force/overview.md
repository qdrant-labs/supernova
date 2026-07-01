# Brute-Force Search (Ground Truth)

`nova bf` computes **exact** k-nearest-neighbours for a set of queries against an embedded corpus — the ground truth you measure an approximate vector store against. Load the same corpus into Qdrant, run the same queries through it, and compare the returned ids to the brute-force `hit_ids` to get recall@k.

It's GPU-accelerated (torch + CUDA) and shards map-reduce style: each worker scores its slice of the corpus and writes a per-query top-K, then a `merge` step folds the partials into one global top-K per query. Because it's exact, it scales with `queries × corpus` — that's the point, and why it's built to fan out across a fleet.

```
                          ┌─▶ nova load ──▶ Qdrant ──▶ (your ANN search)
parquet (S3/HF/local) ────┤                               │  compare ids
                          └─▶ nova bf ────▶ ground-truth top-K  ◀── recall@k
```

## Configuration

Brute-force configs live in `configs/brute_force/`. The same file drives a single-GPU run and a distributed fleet (see [Running](#running)).

```yaml
corpus:
  # The embedded parquets to search over — the same files the loader ingests.
  path: s3://my-bucket/dataset/model
  dense_column: dense_embedding
  # id_column: id            # optional — see "Hit IDs" below

queries:
  # Query embeddings: a single parquet file or a directory of them.
  path: s3://my-bucket/dataset/model/queries.parquet
  dense_column: dense_embedding
  # id_column: query_id      # optional — an existing column to use as the query id
  payload_fields:            # columns carried from the queries file into each output row
    - text

output:
  path: s3://my-bucket/dataset/model/eval

params:
  k: 1000
  metric: cosine             # cosine | dot | euclidean
  io_workers: 16             # concurrent corpus-file reader threads
  io_thread_count: 0         # pyarrow IO-pool size (0 = pyarrow's default ~8)
  # corpus_batch_size: 4096  # bound GPU memory on huge files; omit = whole file at once
```

`${VAR}` / `${VAR:-default}` references are expanded from the environment, same as every other tool.

### Filtering the corpus

To evaluate recall for a *filtered* search, restrict which corpus rows are eligible neighbors with a top-level `filter`, shaped like a Qdrant filter:

```yaml
filter:
  must:
    - field: language
      match: eng          # scalar → equality; a list matches any of them (MatchAny)
    - field: cost
      range: {lt: 10}     # gt / gte / lt / lte — combinable in one condition
  should: []               # OR-at-least-one
  must_not: []              # AND-NOT
```

A condition's `field` is the only place you name a corpus column — there's no separate list to keep in sync, so `compute` reads exactly (and only) the columns the filter references. The filter applies uniformly to every query in the run: it restricts which corpus points are searchable, the same way a Qdrant search filter does — it never touches the queries themselves.

## Running

```bash
# single GPU — scan the whole corpus, write the final result
nova bf compute configs/brute_force/my_eval.yaml

# fleet — each rank scans a stride slice of the corpus files…
nova bf compute configs/brute_force/my_eval.yaml --num-jobs 8 --job-rank $RANK
# …then combine the per-rank partials into the final top-K (run once)
nova bf merge   configs/brute_force/my_eval.yaml
```

Single-GPU `compute` (no `--num-jobs`) writes the final result directly; no `merge` needed. For a fleet, see [`nova dist bf`](../distributed.md#bf), which provisions the GPU pool and runs the ranked jobs for you. Per-flag detail is in the [CLI reference](../reference/cli.md#nova-bf).

## How it works

It's a two-phase intra-then-inter-worker map-reduce:

1. **`compute` (map)** — each worker loads the query embeddings onto the GPU, takes a deterministic stride slice of the corpus files (`file_index % num_jobs == job_rank`), and for each file scores `queries × corpus_rows`, folding the file's top-K into a running per-query top-K held on the GPU. It writes one partial parquet.
   - The running top-K stores `(score, encoded_int)` where `encoded = global_file_index × MAX_ROWS_PER_FILE + row` — keeping an integer on the GPU (not id strings) makes the per-file merge a cheap `torch.topk`. Hit ids are materialised only for the final K per query.
2. **`merge` (reduce)** — slices are disjoint (stride partition → no overlapping hits), so merging is just: concatenate each query's candidates across partials and keep the global top-K. Runs on the controller.

## Output

`compute` (fleet) writes per-rank partials under:

```
{output.path}/_bf_partial_<queries-stem>_k<K>/rank<NNN>.parquet
```

and `merge` (or single-GPU `compute`) writes the final result:

```
{output.path}/bf_<queries-stem>_k<K>.parquet
```

| Column | Type | |
|--------|------|--|
| `query_id` | `str` | from `queries.id_column`, else `make_point_id(query_file, row)` |
| *(payload)* | — | each column listed in `queries.payload_fields` |
| `hit_ids` | `list[str]` | the K nearest corpus ids, best first |
| `hit_scores` | `list[float]` | their similarity scores, descending |

**Sanity check:** if a query also appears in the corpus, its top hit should be itself with score ≈ 1.0 (cosine).

## Hit IDs & recall evaluation

`hit_ids` are how you join ground truth back to a loaded collection, so they must match the point ids the store holds. Two modes:

- **Default — `make_point_id(corpus_file_key, row)`.** A deterministic UUID over `(parquet path, physical row)`, byte-identical to the loader's `vf_point_id` macro. So the brute-force hit ids equal the Qdrant point ids the loader produced, and recall is a straight id-set intersection — no extra columns needed. This is the right choice when the corpus has no natural id.
- **`corpus.id_column`.** Use an already-unique column verbatim (e.g. fineweb's `id` = `<urn:uuid:...>`). Transparent for public datasets and resolvable without reconstructing the loader's hashing. Such an id isn't recomputable from `(file, row)`, so it's read alongside the dense column and **kept in RAM per file** for the worker's slice — budget roughly `slice_rows × id_size` of host memory.

Whichever you pick, the corpus loaded into the vector store must use the **same** id scheme, or the id sets won't line up and recall reads ~0.

## Performance & tuning

The work splits into three layers: **reading** corpus parquet from S3, **decoding** it (parquet → Arrow → numpy, on CPU), and **scoring** on the GPU. For typical query counts the GPU is light; the read + decode path dominates, so tune those first.

| Knob | Default | Guidance |
|------|---------|----------|
| `params.io_thread_count` | `0` (≈8) | **The real S3 fetch concurrency.** pyarrow funnels every read through one global IO pool, so this — not `io_workers` — is what raises throughput once decode keeps up. Try `64`–`128` on a fat NIC. |
| `params.io_workers` | `16` | Concurrent corpus-file reader threads (each holds ~one file in RAM, so `io_workers × file_size` must fit host memory). Useful, but caps at `io_thread_count` — raising it alone won't lift throughput. |
| instance vCPUs | — | Parquet decode is CPU-bound and scales ~linearly with cores. The brute-force matmul is light, so **pick the instance for vCPUs, not the GPU** (e.g. a single-GPU, high-core `g5.16xlarge`). |
| `params.corpus_batch_size` | `None` | The per-file score matrix is `queries × rows`. Big files (or very large query sets) can OOM the GPU; set this to score in row-batches. Values below `k` are raised to `k`. Omit for the whole-file (fastest) path. |
| region | — | `nova bf` is S3-read-heavy — run workers in the **same region** as the corpus bucket to avoid the cross-region bandwidth cap and egress. |

A good starting point for a large corpus on AWS: a high-vCPU single-GPU instance, `io_thread_count: 128`, `io_workers: 32–64`. Raising query count shifts the balance toward the GPU — at that point batch the matmul (`corpus_batch_size`) and add GPUs/workers.

> **Reading fewer bytes** helps every layer: `compute` already projects only the dense column (plus `id_column`/`filter` fields when configured), so the heavy work is unavoidable corpus data. Storing the dense column as fp16 (half the bytes to transfer *and* decode) is the next lever if the read path is still the bottleneck.
