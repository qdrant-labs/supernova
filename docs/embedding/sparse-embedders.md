# Sparse Embedders

Sparse embedders produce sparse vectors where most values are zero. They complement dense embedders for hybrid retrieval — dense captures semantic meaning, sparse captures exact-term and keyword signals.

## Sentence Transformers (SparseEncoder)

The `sentence_transformer` type uses sentence-transformers' `SparseEncoder`. It supports SPLADE-family models and BM42.

```yaml
sparse_embedder:
  type: sentence_transformer
  model: naver/splade-cocondenser-ensembledistil
  batch_size: 64
  dtype: float32
```

### Supported parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `Alibaba-NLP/gte-multilingual-base` | HF model ID |
| `batch_size` | `32` | Texts per forward pass |
| `device` | auto | `cuda`, `mps`, or `cpu` |
| `dtype` | `float32` | `float32`, `float16`, `bfloat16` |
| `trust_remote_code` | `false` | Required for some models |

### Recommended models

| Model | Notes |
|-------|-------|
| `naver/splade-cocondenser-ensembledistil` | Strong SPLADE model, good general-purpose sparse retrieval |
| `naver/splade-v3` | Newer SPLADE, slightly higher quality |
| `Qdrant/bm42-all-minilm-l6-v2-attentions` | BM42 — neural BM25 approximation, no corpus fitting needed |
| `Alibaba-NLP/gte-multilingual-base` | Multilingual, works alongside the dense variant for hybrid |

## FastEmbed (BM25)

The `fastembed` type uses Qdrant's [fastembed](https://github.com/qdrant/fastembed) library. It's the recommended way to produce true BM25 sparse vectors — no GPU required, no corpus fitting needed.

```yaml
sparse_embedder:
  type: fastembed
  model: Qdrant/bm25
  batch_size: 256
```

### Supported parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `Qdrant/bm25` | fastembed model ID |
| `batch_size` | `256` | Texts per batch (CPU-only, can be large) |
| `cache_dir` | system default | Override model download location |

### Recommended models

| Model | Notes |
|-------|-------|
| `Qdrant/bm25` | Pure BM25 — fast, lexical, no GPU needed |
| `Qdrant/bm42-all-minilm-l6-v2-attentions` | BM42 — neural BM25 approximation, slightly higher quality |

BM25 has no transformer token limit — text length is governed by the pipeline's `max_text_length` setting instead.

## Hybrid mode

When `dense_embedder` and `sparse_embedder` both use the same `sentence_transformer` model, supernova runs a single forward pass and produces both vectors at once:

```yaml
dense_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16

sparse_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  batch_size: 64
  dtype: bfloat16
```

The optimization is detected automatically — no extra config needed. The output parquet will have both `dense_embedding` and `sparse_embedding` columns.

## Output format

Sparse embeddings are stored as a struct with two parallel arrays:

```
sparse_embedding: {
  indices: [42, 1337, 8192, ...],   -- token IDs with non-zero weight
  values:  [0.82, 0.31, 0.14, ...]  -- corresponding weights
}
```

Query with DuckDB:

```sql
SELECT row_id, text[:80] AS preview,
       len(sparse_embedding.indices) AS nonzero_dims
FROM 's3://bucket/prefix/**/*.parquet'
LIMIT 10;
```