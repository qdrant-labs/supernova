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

### Giving each search its own query rows

By default every search scores every row of the queries file. 
However, `rows:` lets a search declare which rows are its own instead. The generic format is as follows: `rows: {column: <parquet_column_with_query_set_values>, isin: [<specific_query_set>]}`. For example, if you have a parquet with a `query_set` column, where the `query_set` values could either be `filtered_text` or `structured`, you would use the following yaml format:  

```yaml
queries:
  path: s3://…/ms_marco_10000_combined.parquet

searches:
  - name: filtered_text
    rows: {column: query_set, isin: [filtered_text]}
    filter:
      must:
        - field: text
          match_text_from_query: keyword_phrase

  - name: structured
    rows: {column: query_set, isin: [structured]}
    filter:
      must:
        - field: language_score
          range_from_query: {gte: ls_gte}
```

Omit `rows` and the search covers every query, as before. The selector column is read from the queries file directly, so it does not need to appear in `queries.payload_fields` (list it there anyway if you want it carried into the output). Values compare as strings. A selector that matches no row is an error, not an empty result.

- **Not supported with `vector_type: multivector`** — that path carries ragged
  per-query token offsets that a row subset would have to rebuild. Configuring
  both is rejected at config load rather than silently ignored.
- Order the unified query file rows based on the query_set value (i.e., keep the same values contiguous) to improve computational efficiency with row selection.
#### What `rows` saves

Specs sharing a `vector_type` still build queries and scores over the **union** of their `rows`, so those allocations only shrink if the union itself is smaller. When the subsets together cover the full query file, the main savings are per-spec top-K state and merge work.

For example, with `k=1000` and 10,000 queries, splitting queries between two specs cuts top-K state from 240 MB to 120 MB and roughly halves merge work.

`rows` also simplifies per-query filters and output: each spec only needs filter values and results for its own queries.

Per-query **filter masks** are the exception, and they shrink by a different
rule — see below.

#### Per-query filter masks shrink with the FILTER's rows, not the vector_type's

A per-query filter whose leaves are all `match_from_query` / `range_from_query`
/ static is evaluated GPU-natively, per corpus batch, and never materializes a
CPU-side mask at all. One with a `match_text` / `match_text_from_query` leaf
cannot be (torch has no string tensor type), so it falls back to
`filters.evaluate`'s numpy path and materializes a real
`(n_queries, file_rows)` boolean mask — bit-packed 8 queries/byte, but built
once per corpus file and held for that file's whole batch loop, once per
in-flight reader. It is the only allocation in a run that scales as
`n_queries × file_rows`, which makes its query axis worth being precise about.

That axis is the **union of the `rows` of the specs that use that filter** —
not the queries file's height, and not the vector_type's row union either.
Three consequences:

- Unioning an unrelated query set into the file does not enlarge it. A
  5,000-query text search costs the same whether it sits in its own 10,000-row
  queries file or in a 110,000-row union file. (Measured: 65.6 MiB → 3.0 MiB of
  packed mask for 5,000 owned queries out of 110,000 over a 5,000-row corpus
  file — a 22× drop, exactly the row ratio.)
- Two specs **sharing one filter** pool their rows, since they share one
  `keeps` entry. Give them distinct filter values and each narrows on its own.
- A spec with no `rows` pins that filter back to full file height, because it
  really does look at every query.

The one thing this does not narrow is a GPU-eligible filter, deliberately: its
per-query state is shared across filters by `FilterCondition`, and its mask is
per-batch (`n_queries × dense_batch_size`) rather than per-file, so it is
roughly three orders of magnitude smaller to begin with.

#### Sentinels are optional, but a GPU-eligible filter still wants one

`rows` means a foreign row's per-query filter value is never read for that
search, so the match-nothing sentinels the two-halves pattern needed
(`zzznomatchzzz`, a token-less phrase) are not required for correctness.

They still buy something for a **GPU-eligible** per-query filter. Such a filter
contributes a row-union over-approximation to the shared corpus batch grid
("does at least one query's own leaf admit this row" —
`_row_union_from_gpu_leaves`), and that reduction runs over the full query
axis. For `match_from_query` / `range_from_query` a **null** value reads as *no
restriction*, so leaving foreign rows null makes the union admit every corpus
row and quietly costs that search its corpus pruning — invisible in a run that
also contains an unfiltered search (which scans everything anyway), then
surprising when the filtered search is rerun alone. An explicit match-nothing
value (`dump_set: [zzznomatchzzz]`) keeps the union tight and costs nothing.

A **text** filter never had this problem, before or after the narrowing above,
and it is worth being precise about why rather than crediting `rows` for it: a
null or empty phrase is *token-less*, and a token-less phrase in a `must`
matches nothing while a null slot in a `should` contributes nothing. Either
way such a query's mask row is all-`False` and adds nothing to the OR. So for
a text filter whose foreign rows carry sentinels or nulls — i.e. every
configuration in this repo — narrowing the mask changes the corpus-row union
by exactly zero. Its payoff is the allocation, not the union. Foreign rows
holding *real* values in the filter's own columns are the only case where the
union genuinely tightens.

#### `rows` is not bit-exact against a full-file run

When a subset *does* shorten the query matrix — i.e. the union of the run's
`rows` is a strict subset of the file, so some rows no search owns — scores can
differ from the same search run over the whole file by ~1 float32 ULP
(observed ~5e-7 relative). Nothing is wrong: the matmul's query dimension
changed, so BLAS picks a different kernel and accumulates in a different order.
Hit **ids** are unaffected except where two documents' scores sit within that
margin, in which case they can swap.

Practically: two searches whose subsets cover every row (the layout above) stay
bit-exact, but rerunning just *one* of them from the same queries file will not
reproduce the combined run's scores to the last bit. Treat scores from a
narrowed run as ~1e-6-comparable, not identical — the same caveat
`params.allow_tf32` carries, at a much smaller magnitude.

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
          match_text: chronic fatigue syndrome  # every token must appear (Qdrant's MatchText, word tokenizer)
      should: []               # OR-at-least-one
      must_not: []              # AND-NOT
```

A condition's `field` is the only place you name a corpus column — there's no separate list to keep in sync, so `compute` reads exactly (and only) the columns the filter references. By default a filter applies uniformly to every query in that search: it restricts which corpus points are searchable, the same way a Qdrant search filter does — it never touches the queries themselves. Each search has its own independent `filter` (or none). For a filter that varies PER QUERY instead, see [Per-query filters](#per-query-filters) below.

`match_text` is **tokenized matching**, the same semantics as Qdrant's full-text-index `MatchText` condition against a `word`-tokenizer index with `lowercase: true`: both the query string and the field are lowercased and split into maximal alphanumeric runs (hyphens, underscores, punctuation, and symbols all separate tokens — Unicode letters/digits stay inside them), and a row matches iff **every** query token is one of the row's tokens (an AND of tokens, in any order — not a phrase match). So `high-fat` matches "a high fat diet", `C++` tokenizes to `c`, and `com_content` contains the token `content` — ground truth built with it is directly comparable to a real Qdrant filtered search over a `word`/`lowercase` text index. (Qdrant's other tokenizers — `whitespace`, `prefix`, `multilingual` — and options like stemming/stopwords are not replicated.)

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

**Cost**: `match_from_query` and `range_from_query` evaluate GPU-natively — a small per-query vocabulary/gather or a direct broadcast comparison, built once per file from small corpus/query-side arrays and evaluated entirely on the GPU — the FLOP cost is no more than an unfiltered search sharing the same batch, and no `(n_queries, rows)` mask is ever materialized on the CPU or shipped over PCIe for either. `match_text_from_query` is different: it stays CPU-only (torch has no string tensor type), but its cost is one **single tokenization pass** over the text column per file — lowercase + split each row once, then answer every distinct query token by a vectorized membership scatter — so it's essentially independent of how many distinct words the query set uses (it used to pay a full column scan per distinct word). The pass is row-batched (bounded transient memory on multi-GB text columns) and fans batches out across a thread pool, so it scales with the same vCPUs the decode path already wants. It's still extra CPU-side work that doesn't overlap GPU scoring — reach for `match_text_from_query` deliberately, not as a default.

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
| `hit_scores` | `list[float]` | the matching scores, descending — always larger-is-nearer (see below) |

**Score sign.** `hit_scores` is always ordered best-first, larger-is-nearer, so one ranking convention covers every metric. For `cosine` and `dot` that is the similarity itself. For **`euclidean` it is the NEGATED distance**, so every value is `≤ 0` and the nearest hit is the least negative:

```
euclidean:  scores=[-0.0007, -1.2913, -2.3253]   # distances 0.0007, 1.2913, 2.3253
cosine:     scores=[ 1.0,     0.8110,  0.5593]
```

Recover a distance with `distance = -score`. This matters when comparing against a vector store directly: Qdrant reports Euclid as a *positive* distance sorted *ascending*, so line the two up by negating one side. It does not affect recall, which is an id-set intersection over `hit_ids` and never reads scores.

Do not confuse this with a negative `dot` or `cosine` score, which is **not** a sign convention — it is the real similarity of two vectors pointing in opposing directions, written through unmodified (`cosine` spans `[-1, 1]`; `dot` is unbounded). The tell is the proportion: `euclidean` is negative for *every* hit by construction, while `dot`/`cosine` are negative only where the data says so. A positive value in a `euclidean` column means something is wrong.

**Sanity check:** if a query also appears in the corpus, its top hit should be itself — with score ≈ 1.0 for cosine, or ≈ 0.0 for euclidean. Expect the euclidean self-hit to be a small negative number rather than exactly `-0.0`: it is computed as `‖q‖² + ‖c‖² − 2q·c`, whose cancellation resolves a true zero to roughly `sqrt(float32 eps) · ‖q‖` (~1e-2 at `‖q‖≈35`). That is inherent to the expansion — `torch.cdist` uses the same one at any real corpus size — and it bounds how finely euclidean *distances* can be trusted between near-duplicates, though not the ranking of ordinary neighbours.

### One run's partials, and all of them

A search's partial directory is addressed by `(queries stem, search name, k)` alone, so **any two runs agreeing on those three write into the same directory**, and rank files overwrite only the ranks the newer run has. A 32-way run landing on a 64-rank run's leftovers leaves 32 fresh partials beside 32 stale ones. Nothing about the rows says so — same schema, same query rows, same hit-id shape — and the merge is clean: the corpus gets double-counted where the two strides overlap and missed where neither covered, producing a top-K that looks entirely normal and is wrong.

Every partial therefore carries a **run fingerprint** in its parquet metadata, and `merge` refuses to reduce a directory whose partials disagree on it. The fingerprint is content-derived rather than a per-invocation id: every rank computes the same value from the same inputs without coordinating, and re-running one failed rank later reproduces it exactly, so the documented recovery path still works. It covers the config (`nova_bf.config_fingerprint` — spec, metric, `k`, filter, `rows`, input paths and columns, `allow_tf32`), the corpus file list *in index order*, `num_jobs`, the tie-break rule, and whether `--max-files` truncated the slice — so a benchmarking partial can never merge with a real one.

Sharded partials also stamp `nova_bf.num_jobs` and `nova_bf.job_rank`, and merge checks the ranks present are exactly `0..num_jobs-1`. This is what catches a **missing** rank: the older "every search has the same partial count" check cannot, because a rank that dies before writing anything leaves every search short by exactly one. A missing rank's slice is simply absent from the merged top-K, silently lowering every recall number computed against it.

Merge also recomputes the config fingerprint from the config *it* was handed and compares — the generalization of the existing `tiebreak` guard to every other field that changes results. Only result-affecting fields are in it: `io_workers`, batch sizes and `output.path` are not, so two runs differing only in those still merge.

Partials written before fingerprinting log a loud warning rather than failing — refusing them would strand hours of legitimate GPU work over a check that would have passed. A directory where only *some* partials carry the stamp is an error, since that mixture is itself the failure.

### Run manifest

Each phase also writes one JSON manifest next to its outputs. The parquet's schema metadata (`nova_bf.*` keys: corpus/queries path, metric, `k`, `allow_tf32`, the stored vector dtypes, the tie-break rule) says what the ground truth **is**; the manifest says what the **run** was — how many ranks, which files this worker took, how long each phase took, what hardware it ran on, and the full filter each search used. It is the same shape `nova embed` writes (`source` / `destination` / `created_at` / `compute` / settings / counts / timing / `output_files`), so both toolsets' artifacts read the same way.

```
{output.path}/_bf_manifest_<queries-stem>_compute.json            # single-GPU compute
{output.path}/_bf_manifest_<queries-stem>_compute/rank<NNN>.json  # one per rank when sharded
{output.path}/_bf_manifest_<queries-stem>_merge.json              # merge
```

| Block | |
|-------|--|
| `source` / `destination` | corpus + queries paths, `include`/`exclude`, and the id/vector/payload columns actually read |
| `source.corpus.fingerprint` (compute) | file count and a sha256 over the corpus file list **in the order the run indexed it** — with no `corpus.id_column` the hit ids are `make_point_id(file_key, row)`, so that order is part of the id scheme: add or rename one file and every later file's ids shift. Comparing this hash proves two runs saw the same corpus, and catches `include`/`exclude` drift between ranks. Merge has no corpus listing of its own, so it carries no fingerprint |
| `compute` | instance type, region, AZ, GPU + count and total memory, torch/CUDA version, the device scoring really ran on, and this run's peak GPU allocated/reserved bytes |
| `code` | supernova workspace version, `git describe` / commit / branch / dirty flag, and the python, numpy and pyarrow versions — the scoring and tie-break kernels *are* the ground truth, so the revision that produced it is part of the record |
| `job` | SkyPilot task/cluster/job ids when present, plus hostname and pid — what ties a suspicious rank's manifest back to its log |
| `params` | every run-level knob **as resolved** — CLI overrides like `--io-workers`, the per-vector-type batch sizes in `batch_size_by_vector_type`, and the multivector tile sizes derived from `multivector_token_budget` all replace the configured (often `null`) values, so it records what ran, not what the YAML said |
| `searches` | per search: `vector_type`, `metric`, `k`, the **full filter** and `rows` selector, query count, output file, and the corpus/queries storage dtypes |
| `sharding` (compute) | `num_jobs`, `job_rank`, corpus files total vs. this worker's, and `partial_slice: true` when `--max-files` made the output invalid as ground truth |
| `counts` / `timing` | `queries_in_file` and `queries_searched` (they differ when a search uses `rows`), corpus rows scanned, bytes decoded; elapsed vs. scan seconds and the `io_wait` / `gpu` / `read` / `filter` split behind the `bf-bench` log line |
| `output_files` | the files this phase wrote — the only record of which are a given rank's once names collide across ranks |

The filter is dumped in full because for a filtered GT the filter *is* the search: recall numbers from two runs are comparable only if the predicates were identical, and a YAML edit between runs is otherwise invisible in the artifacts. A merge manifest also records `partials` per search — how many ranks' candidates were actually folded in, which the final parquet cannot tell you and which a dead rank silently changes.

Writing a manifest is best-effort: it is a record *of* outputs that already landed, so a failure to write one logs a warning and never fails the run.

## Hit IDs & recall evaluation

`hit_ids` are how you join ground truth back to a loaded collection, so they must match the point ids the store holds. Two modes:

- **Default — `make_point_id(corpus_file_key, row)`.** A deterministic UUID over `(parquet path, physical row)`, byte-identical to the loader's `vf_point_id` macro. So the brute-force hit ids equal the Qdrant point ids the loader produced, and recall is a straight id-set intersection — no extra columns needed. This is the right choice when the corpus has no natural id.
- **`corpus.id_column`.** Use an already-unique column verbatim (e.g. fineweb's `id` = `<urn:uuid:...>`). Transparent for public datasets and resolvable without reconstructing the loader's hashing. Such an id isn't recomputable from `(file, row)`, so it's read alongside the vector column(s) and **kept in RAM per file** for the worker's slice — budget roughly `slice_rows × id_size` of host memory.

Whichever you pick, the corpus loaded into the vector store must use the **same** id scheme, or the id sets won't line up and recall reads ~0.

## Ties

Two documents can score **exactly** equal, and more often than you'd guess: duplicate documents embed to bit-identical vectors, sparse and filtered searches collide constantly, and at production `k` over thousands of candidates float32 stops separating scores deep in the tail. Something has to decide which one makes the cut, and left alone `torch.topk` (and numpy's `argpartition`/`argsort`, all unstable) decided by whatever the selector happened to visit first — so the answer moved with `dense_batch_size` and `--num-jobs`.

`params.tiebreak` fixes the rule:

| Value | Rule | Cost |
|---|---|---|
| `ordinal` *(default)* | Earlier in the corpus wins. | Free. No id column needed. |
| `id` | Lower `corpus.id_column` wins; among equal ids, earlier in the corpus. | One sort of this worker's id column at startup. Requires `corpus.id_column`. |

`id` is closer to a property of the data than of how it happens to be laid out on disk, so it survives re-sharding the corpus into different files; `ordinal` does not. Neither costs anything at scoring time.

Ordering follows the column's **type**: a numeric id column compares numerically (9 before 10), a string column bytewise. There is no length or entropy limit — what is keyed is the id's *position* in sorted order, not its bytes, so 128-bit UUIDs, long shared prefixes like `<urn:uuid:…>`, zero-padded ids and the full `uint64` range all separate exactly. A null id is rejected under `id` (it has no ordering position, and every null row would report the same `"None"` hit id).

### What this guarantees, and what it does not

Given two candidates whose scores are equal **bit for bit**, which one survives and in what order they appear is invariant to `--num-jobs`, every batch size, `io_workers`, thread counts, device, and file arrival order.

It does **not** make the scores themselves reproducible. A matmul's reduction order is part of its answer, so re-tiling can move a score by one ULP and change whether two documents tie *at all* — measured on a real corpus, `dense_batch_size: 64` and `512` disagreed in the last bit of a score, and `dot` is not immune. `--num-jobs` preserves batch shapes and so is safe; batch size is not. **Pin `dense_batch_size` (and leave `allow_tf32` off) if you need a bit-reproducible artifact across runs.**

Two further limits worth knowing:

- **4.29B rows per worker.** The rule rides in a 32-bit field, so one worker may hold at most that many rows — not a limit on the corpus, and not on how many distinct ids exist. Exceeding it is a hard error pointing at `--num-jobs`; at 10B rows over 64 workers there is 27× headroom.
- **This makes the artifact reproducible, not the engines identical.** Qdrant's `ScoredPoint` compares score only and has no tie-break of its own, so nova-bf and Qdrant may still keep different points at a tie boundary. That is what nova-storm's tie-tolerant recall absorbs.

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
