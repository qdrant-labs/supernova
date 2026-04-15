# Sparse Embedders

## Sentence Transformers (SparseEncoder)

Uses sentence-transformers' `SparseEncoder` for models like SPLADE and gte-multilingual-base:

```yaml
sparse_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  batch_size: 64
  dtype: bfloat16
```

Sparse embeddings are stored as a struct with two parallel arrays (`indices` and `values`) in the output parquet files.

## Hybrid mode

When both `dense_embedder` and `sparse_embedder` point to the same model (same type and model name), vectorforge automatically uses a hybrid encoder that produces both in fewer forward passes:

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

No special config needed -- the optimization is detected and applied automatically. Both `dense_embedding` and `sparse_embedding` columns appear in the output parquet.
