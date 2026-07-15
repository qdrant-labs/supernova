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

Every config lists one or more **searches** — there's no implicit default search, so a config always says explicitly what it's computing:

```yaml
corpus:
  # The embedded parquets to search over — the same files the loader ingests.
  path: s3://my-bucket/dataset/model
  dense_column: dense_embedding
  # sparse_column: sparse_embedding  # only read by a search with vector_type: sparse
  # id_column: id            # optional — see "Hit IDs" below

queries:
  # Query embeddings: a single parquet file or a directory of them.
  path: s3://my-bucket/dataset/model/queries.parquet
  dense_column: dense_embedding
  # sparse_column: sparse_embedding
  # id_column: query_id      # optional — an existing column to use as the query id
  payload_fields:            # columns carried from the queries file into each output row
    - text

output:
  path: s3://my-bucket/dataset/model/eval

params:
  io_workers: 16             # concurrent corpus-file reader threads
  io_thread_count: 0         # pyarrow IO-pool size (0 = pyarrow's default ~8)
  # dense_batch_size: 4096   # bound GPU memory on huge files; omit = whole file at once
  # sparse_batch_size: 4096  # same, for vector_type: sparse searches
  # merge_batch_size: null   # merge tuning, see Performance & tuning
  # merge_prefetch: false

searches:
  - name: dense_all          # optional, unique if set, [A-Za-z0-9_-]+ — goes into the output filename
    vector_type: dense       # dense | sparse
    metric: cosine           # cosine | dot | euclidean (euclidean unsupported with vector_type: sparse)
    k: 1000
    # filter: {...}            # see "Filtering the corpus" below
```

`${VAR}` / `${VAR:-default}` references are expanded from the environment, same as every other tool. `params` holds run-level IO/merge/GPU-batching tuning; every search-specific setting (`k`, `metric`, `vector_type`, `filter`) lives on its `searches[]` entry. GPU batch size (`dense_batch_size`/`sparse_batch_size`) lives on `params`, not per search — see [One search, or several in one pass](#one-search-or-several-in-one-pass) for why.

### One search, or several in one pass

A single `compute` run computes every entry in `searches:` — one is the common case, but you can list several **independent** top-K results (e.g. dense-unfiltered, sparse-unfiltered, and a filtered variant of either) and they'll share corpus file IO/decode: each corpus file is read and decoded only **once per vector_type any search needs**, not once per search. This is not a fused hybrid score — each search gets its own ranked list, own `k`/`metric`/`filter`, and own output file; they just cost roughly the price of one corpus scan instead of one scan per search (the read+decode path is what dominates, see [Performance & tuning](#performance--tuning)).

Per `vector_type`, searches share GPU work one of two ways:

- **If any search of that vector_type is unfiltered**, every search of that vector_type — filtered or not — shares one full-file GPU pass: the transfer/CSR build and each distinct `metric`'s score matrix are computed once per batch, and every search (including filtered ones) reads its top-K straight from those shared columns, masking down to its own filter's surviving rows first if it has one. This is exact, not an approximation — masking a raw score matrix commutes with computing it — so a filtered search's `metric` never needs to match anything else in the run; scoring one more metric on a batch that's already on the GPU is cheap, unlike a second transfer. In the example above, `dense_eng` and `sparse_eng` both ride `dense_all`/`sparse_all`'s pass this way.
- **Otherwise** (no search of that vector_type is unfiltered), there's no full-file pass to ride, so the shared grid instead compacts to the UNION of every distinct active filter's surviving rows, transferred/scored once, with each search masking down further to its own filter's subset of that union — never more row-scoring than treating each filter independently would, and less whenever two filters' surviving rows overlap. A per-query filter (see [Per-query filters](#per-query-filters)) contributes a cheap, safe over-approximation to this same union rather than an exact row-subset, so it benefits from (and doesn't block) compaction too.

Either way, GPU batch size for a vector_type is one run-level setting (`params.dense_batch_size`/`sparse_batch_size`) — it's not something you tune per search, since every search of a vector_type ends up sharing one GPU pass over the corpus regardless of grouping. What happens when it's set below some search's `k` differs by which of the two cases above applies: in the **shared-pass** case, your configured value is kept as-is (an under-filled batch just costs the larger-`k` search a few extra merge rounds — never a wrong answer, and never a memory bound you didn't ask for, since one unrelated search's large `k` would otherwise silently widen every OTHER search's GPU footprint too); in the **grouped-by-filter** case, each group's own batch size floors at the largest `k` among that GROUP's own (related, identically-filtered) members, same as before — there's no unrelated search sharing the grid to protect against there.

```yaml
searches:
  - name: dense_all
    vector_type: dense
    metric: cosine
    k: 1000
  - name: dense_eng
    vector_type: dense
    metric: cosine
    k: 1000
    filter:
      must:
        - field: language
          match: eng
  - name: sparse_all
    vector_type: sparse
    metric: dot
    k: 1000
  - name: sparse_eng
    vector_type: sparse
    metric: dot
    k: 1000
    filter:
      must:
        - field: language
          match: eng
```

`name` is optional — it's spliced into the output filename (`bf_<queries-stem>_<name>_k<K>.parquet`), so if you set it, it must be unique. Omit it and one is derived from `vector_type`/`metric` (e.g. `dense_cosine`), with `_filtered` appended when `filter` is set and any collision (with another default, or with an explicit name elsewhere in `searches`) disambiguated by an incrementing suffix (`_2`, `_3`, …) — so the single-search case above needs no `name:` line at all.

**Mixing `vector_type: dense` and `vector_type: sparse` in one run doubles the per-file host-RAM budget**: each in-flight file's reader decodes both columns at once, so `io_workers × file_size` (see [Performance & tuning](#performance--tuning)) becomes `io_workers × (dense_bytes + sparse_bytes)`. Lower `io_workers` accordingly on memory-constrained boxes when mixing vector_types.

### Sparse vectors

Set a search's `vector_type: sparse` to score a `struct<indices: list<uint32>, values: list<float32>>` column instead of the dense one — the same schema `nova embed`'s sparse embedders write and `nova load` reads (default column name `sparse_embedding`, override via `corpus.sparse_column` / `queries.sparse_column`). Only `metric: dot` and `metric: cosine` are supported (`euclidean` has no real use case for sparse retrieval and is rejected at config load).

Scoring densifies the query set once over its own token vocabulary (a corpus-only token id can never match any query, so dropping it is exact, not approximate) and keeps each corpus batch genuinely sparse (`torch.sparse_csr_tensor`) on the GPU, scored via `sparse @ dense` matmul — `params.sparse_batch_size` bounds GPU residency the same way `dense_batch_size` does for dense.

### Filtering the corpus

To evaluate recall for a *filtered* search, restrict which corpus rows are eligible neighbors with that search's `filter`, shaped like a Qdrant filter:

```yaml
searches:
  - name: dense_eng
    vector_type: dense
    filter:
      must:
        - field: language
          match: eng          # scalar → equality; a list matches any of them (MatchAny)
        - field: cost
          range: {lt: 10}     # gt / gte / lt / lte — combinable in one condition
        - field: text
          match_text: chronic fatigue syndrome  # every word must appear (Qdrant's MatchText)
      should: []               # OR-at-least-one
      must_not: []              # AND-NOT
```

A condition's `field` is the only place you name a corpus column — there's no separate list to keep in sync, so `compute` reads exactly (and only) the columns the filter references. By default a filter applies uniformly to every query in that search: it restricts which corpus points are searchable, the same way a Qdrant search filter does — it never touches the queries themselves. Each search has its own independent `filter` (or none). For a filter that varies PER QUERY instead, see [Per-query filters](#per-query-filters) below.

`match_text` requires every whitespace-separated word in the string to appear somewhere in the field, case-insensitively and in any order (an AND of words, not a phrase match) — the same semantics as Qdrant's own full-text-index `MatchText` condition, so ground truth built with it is directly comparable to a real Qdrant filtered search. It's a word-boundary-regex approximation of Qdrant's real tokenizer, not a byte-for-byte replica: a hyphenated query word like `high-fat` is matched as one literal token rather than split into `high`/`fat`, and a word ending directly in trailing punctuation (`C++`) can fail to get a boundary on that side. Good enough for keyword-style corpus filtering.

### Per-query filters

Every condition kind above has a per-query variant — `match_from_query`, `range_from_query`, `match_text_from_query` — that pulls its comparison value(s) from a column in the **queries** file instead of a literal in this config, so two different queries in the same search can each be restricted to a different corpus subset (tenant/user scoping, a per-query budget, a per-query search phrase):

```yaml
searches:
  - name: per_tenant_search
    vector_type: dense
    filter:
      must:
        - field: tenant_id             # corpus column
          match_from_query: tenant_id  # queries column — each query's own value
        - field: cost
          range_from_query: {lt: max_budget}   # queries column per bound
        - field: title
          match_text_from_query: search_phrase # queries column — each query's own phrase
```

A per-query and a static condition can appear together, in any of `must`/`should`/`must_not` — `should: [{field: is_public, match: true}, {field: tenant_id, match_from_query: tenant_id}]` means "public OR this query's own tenant." `match_from_query`'s queries column can hold either a scalar (equality) or a list per row (per-query MatchAny, matching Qdrant's `match` list semantics). `range_from_query` doesn't mix a literal bound with a per-query one in the same condition — express "cost > 0 for everyone AND cost < my own budget" as two separate conditions instead (a static `range: {gt: 0}` plus a `range_from_query: {lt: max_budget}`), which combine the same way any two `must` conditions do. A null/missing value on either side (corpus or queries column) never matches, same convention as a static filter's null handling.

**Cost**: `match_from_query` and `range_from_query` evaluate GPU-natively — a small per-query vocabulary/gather or a direct broadcast comparison, built once per file from small corpus/query-side arrays and evaluated entirely on the GPU — the FLOP cost is no more than an unfiltered search sharing the same batch, and no `(n_queries, rows)` mask is ever materialized on the CPU or shipped over PCIe for either. `match_text_from_query` is different: it dedupes by *distinct word* (not just distinct phrase) and evaluates each distinct word's regex once, but for genuinely per-query search text — the realistic case for real query logs, as opposed to a benchmark with heavily overlapping vocabulary — cost still approaches one regex pass per distinct word, additional CPU-side work that (unlike IO) doesn't overlap with GPU scoring, and stays CPU-only (torch has no string tensor type). Reach for `match_text_from_query` deliberately, not as a default.

**Memory**: unlike an unfiltered search (which never materializes any mask at all — rows are just used as-is), a filter with a `match_text`/`match_text_from_query` leaf anywhere still materializes a full `(n_queries, corpus_rows_in_this_file)` boolean array on the CPU per file (`match_from_query`/`range_from_query` never do, per the GPU-native path above). For a modest query set this is negligible; for a query-log-scale run (hundreds of thousands of queries) against a large corpus file with `match_text_from_query`, this is worth sizing.

A per-query filter can share a compacted batch grid with other filters of the same vector_type, not just ride the whole file: it contributes a cheap, safe over-approximation ("does at least one query's own value admit this row", built only from its `must`-group leaves) to the same row-union compaction every other filter of that vector_type benefits from (see [One search, or several in one pass](#one-search-or-several-in-one-pass)) — its own fine, per-query masking still applies afterward regardless of whether the shared grid ended up compacted or not.

## Running

```bash
# single GPU — scan the whole corpus, write the final result(s)
nova bf compute configs/brute_force/my_eval.yaml

# fleet — each rank scans a stride slice of the corpus files…
nova bf compute configs/brute_force/my_eval.yaml --num-jobs 8 --job-rank $RANK
# …then combine the per-rank partials into the final top-K (run once)
nova bf merge   configs/brute_force/my_eval.yaml
```

Single-GPU `compute` (no `--num-jobs`) writes the final result(s) directly; no `merge` needed. For a fleet, see [`nova dist bf`](../distributed.md#bf), which provisions the GPU pool and runs the ranked jobs for you. Per-flag detail is in the [CLI reference](../reference/cli.md#nova-bf).

## How it works

It's a two-phase intra-then-inter-worker map-reduce, run independently per search:

1. **`compute` (map)** — each worker loads the query embeddings onto the GPU, takes a deterministic stride slice of the corpus files (`file_index % num_jobs == job_rank`), and for each file scores `queries × corpus_rows`, folding the file's top-K into a running per-query top-K held on the GPU. It writes one partial parquet per search.
   - The running top-K stores `(score, encoded_int)` where `encoded = global_file_index × MAX_ROWS_PER_FILE + row` — keeping an integer on the GPU (not id strings) makes the per-file merge a cheap `torch.topk`. Hit ids are materialised only for the final K per query.
2. **`merge` (reduce)** — slices are disjoint (stride partition → no overlapping hits), so merging is just: concatenate each query's candidates across partials and keep the global top-K, per search. Runs on the controller.

## Output

`compute` (fleet) writes per-rank partials under, one directory per search:

```
{output.path}/_bf_partial_<queries-stem>_<name>_k<K>/rank<NNN>.parquet
```

and `merge` (or single-GPU `compute`) writes each search's final result:

```
{output.path}/bf_<queries-stem>_<name>_k<K>.parquet
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
- **`corpus.id_column`.** Use an already-unique column verbatim (e.g. fineweb's `id` = `<urn:uuid:...>`). Transparent for public datasets and resolvable without reconstructing the loader's hashing. Such an id isn't recomputable from `(file, row)`, so it's read alongside the vector column(s) and **kept in RAM per file** for the worker's slice — budget roughly `slice_rows × id_size` of host memory.

Whichever you pick, the corpus loaded into the vector store must use the **same** id scheme, or the id sets won't line up and recall reads ~0.

## Performance & tuning

The work splits into three layers: **reading** corpus parquet from S3, **decoding** it (parquet → Arrow → numpy, on CPU), and **scoring** on the GPU. For typical query counts the GPU is light; the read + decode path dominates, so tune those first.

| Knob | Default | Guidance |
|------|---------|----------|
| `params.io_thread_count` | `0` (≈8) | **The real S3 fetch concurrency.** pyarrow funnels every read through one global IO pool, so this — not `io_workers` — is what raises throughput once decode keeps up. Try `64`–`128` on a fat NIC. |
| `params.io_workers` | `16` | Concurrent corpus-file reader threads (each holds ~one file in RAM, so `io_workers × file_size` must fit host memory — **double that if any run mixes `vector_type: dense` and `vector_type: sparse` searches**, since each in-flight file then decodes both columns at once). Useful, but caps at `io_thread_count` — raising it alone won't lift throughput. |
| instance vCPUs | — | Parquet decode is CPU-bound and scales ~linearly with cores. The brute-force matmul is light, so **pick the instance for vCPUs, not the GPU** (e.g. a single-GPU, high-core `g5.16xlarge`). |
| `params.dense_batch_size` / `sparse_batch_size` | `None` | The per-file score matrix is `queries × rows`. Big files (or very large query sets) can OOM the GPU; set this to score in row-batches. Omit for the whole-file (fastest) path. One value per vector_type, run-wide — every search of a vector_type ends up sharing one GPU pass over the corpus (see [One search, or several in one pass](#one-search-or-several-in-one-pass)). When some search of that vector_type is unfiltered (the shared-pass case), your configured value is always kept as-is — it's a memory bound, so it's never silently raised, even if some search's `k` exceeds it (that search just takes a few extra merge rounds). Otherwise, each filter group's own batch size floors at the largest `k` among that group's own members. |
| region | — | `nova bf` is S3-read-heavy — run workers in the **same region** as the corpus bucket to avoid the cross-region bandwidth cap and egress. |

A good starting point for a large corpus on AWS: a high-vCPU single-GPU instance, `io_thread_count: 128`, `io_workers: 32–64`. Raising query count shifts the balance toward the GPU — at that point batch the matmul (`params.dense_batch_size`/`sparse_batch_size`) and add GPUs/workers.

> **Reading fewer bytes** helps every layer: `compute` projects only the vector column(s) any search needs (plus `id_column`/`filter` fields when configured), so the heavy work is unavoidable corpus data. Storing the dense column as fp16 (half the bytes to transfer *and* decode) is the next lever if the read path is still the bottleneck.

The `timing`/`bf-bench` log lines' `rows`/`gb` are the corpus's raw pre-filter row/byte count for each vector_type actually read, summed across every search sharing the run — not "rows that survived a filter," and not broken out per search, since several searches with different filters no longer have one single count to report.
