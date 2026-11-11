# Storm

`nova storm` sends sustained query load to a vector store and reports latency and throughput. With ground truth from `nova bf`, it also reports recall and rank agreement (RBO).

Qdrant is built in; Elasticsearch, OpenSearch, and Milvus are optional.

## Config

```yaml
target:
  type: qdrant
  url: http://localhost:6334
  collection_name: my-collection

query:
  top_k: 10
  source:
    uri: s3://my-bucket/eval/bf_queries_dense_all_k1000.parquet
    column: dense_embedding
    limit: 5000
    ground_truth_column: hit_ids
    ground_truth_score_column: hit_scores
  search_params:
    hnsw_ef: 128

load:
  concurrency: 32
  duration_s: 60
  # rps: 200
  batch_size: 1

# report:
#   format: csv
#   path: storm_timeseries.csv
```

Environment variables support `${VAR}` and `${VAR:-default}` expansion.

## Query

| Key | Default | Description |
|---|---:|---|
| `top_k` | `10` | Results per query |
| `vector_name` | none | Named vector to search |
| `vector_type` | `dense` | `dense`, `sparse` or `hybrid` (Qdrant) |
| `with_payload` | `false` | Payload fields to return |
| `search_params` | none | Backend-specific search settings |
| `filter` | none | Qdrant filter |
| `tie_epsilon` | datatype-dependent | Score tolerance for ties |
| `rbo_p` | derived from `top_k` | RBO persistence |

`query.source.uri` and `column` are required. `limit` controls the number of queries loaded. `ground_truth_column` enables recall; `ground_truth_score_column` enables tie-aware recall.

### Hybrid Queries

`vector_type: hybrid` sends one Qdrant query per row with two prefetches, dense and sparse, and fuses them. `source.column` is the dense half and `source.sparse_column` the sparse half, read from the same row. Each prefetch gets the run's filter; `search_params` apply to the dense prefetch only.

```yaml
query:
  vector_type: hybrid
  vector_name: dense              # the dense prefetch's vector
  top_k: 10
  hybrid:
    sparse_vector_name: sparse    # the sparse prefetch's vector
    prefetch_limit: 100           # candidates per prefetch; default top_k
    fusion: rrf                   # rrf (default) or dbsf
  source:
    uri: queries.parquet
    column: dense_embedding
    sparse_column: sparse_embedding
```

Recall compares the fused results with `ground_truth_column`, so that ground truth must itself be hybrid. `ground_truth_score_column` is rejected: fused scores are not distances.

### Search Params

| Target | Settings |
|---|---|
| `qdrant` | `hnsw_ef`, `exact`, `quantization` |
| `opensearch` | `ef_search`, `nprobes`, `rescore` |
| `elastic` | `num_candidates` |
| `milvus` | `ef` or `nprobe` |

## Filters

Qdrant supports the same `must`, `should`, and `must_not` filters as `nova bf`, including static and per-query `match`, `range`, and `match_text` conditions.

```yaml
filter:
  must:
    - field: tenant_id
      match_from_query: tenant_id
    - field: cost
      range: {lt: 10}
```

## Targets

Supported targets are `qdrant`, `opensearch`, `elastic`, and `milvus`.

Build optional backends with:

```bash
make storm STORM_FEATURES=elastic,opensearch,milvus
```

## Running

```bash
nova storm my_storm.yaml
nova storm my_storm.yaml --json
```

`--json` emits machine-readable output for `nova sweep`.

In distributed runs, each worker executes the same query workload.

## Load Model

By default, Storm runs closed-loop: each worker sends another request as soon as the previous one completes, measuring maximum throughput.

Set `rps` for open-loop execution at a fixed request rate:

```yaml
load:
  concurrency: 32
  duration_s: 60
  rps: 200
```

| Setting | Description |
|---|---|
| `concurrency` | Maximum requests in flight |
| `duration_s` | Run duration |
| `rps` | Requests/s per worker; `0` means closed-loop |
| `passes` | Run every query a fixed number of times |
| `batch_size` | Queries per request |
| `defer_scoring` | Compute recall after the load window |

There is no warm-up phase.

## Output

Storm reports request count and errors, requests/s, queries/s, p50/p95/p99/max latency, and accuracy metrics when ground truth is available.

Optional `report` output writes per-request results to CSV or JSONL.

## Accuracy

Recall is scored once per query.

- `recall@k`: recall for queries with at least `top_k` ground-truth results.
- `recall@k_short`: recall for queries with fewer than `top_k` valid results.
- `rbo@k`: agreement with the ground-truth ranking.

A non-zero `missing_from_gt` usually indicates stale ground truth or a mismatched collection.

### Ties

Documents tied at the ground-truth cutoff may be equally valid even if different IDs are returned. When `ground_truth_score_column` is provided, Storm reports a tie-aware recall range:

```text
recall@10: 0.8630 – 1.0000
```

The lower bound counts exact ID matches; the upper bound also accepts returned results whose scores tie the ground-truth cutoff.

`tie_epsilon` controls how close scores must be to count as tied. By default it is `5e-6` for float32 and `2e-3` otherwise. If `nova bf` used `allow_tf32: true`, use `1e-3` or larger.

Tie-aware reporting is unavailable when comparable scores are not returned, including Milvus, Elasticsearch, sparse search, and quantized Qdrant searches without rescoring.

### Rank Agreement (RBO)

Recall measures which results were returned but ignores their order. `rbo@k` measures ranking agreement, weighting differences near the top more heavily.

Storm also reports normalized RBO, where `1.0` represents a perfect ranking.

`rbo_p` controls how strongly top ranks are weighted. Set it explicitly when comparing runs with different `top_k`.

Report RBO alongside recall: recall measures result overlap, while RBO measures ordering.