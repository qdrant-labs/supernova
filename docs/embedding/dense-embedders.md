# Dense Embedders

## OpenAI

Uses the OpenAI API. Best for smaller datasets or when you don't have GPUs.

```yaml
dense_embedder:
  type: openai
  model: text-embedding-3-small    # or text-embedding-3-large
  dimensions: 1536                 # optional dimension selection
  batch_size: 128
  max_concurrent: 8                # parallel API calls
```

- Rate limiting with exponential backoff
- Text splitting via tiktoken
- Max 8192 tokens per text

### OpenAI-compatible APIs

The OpenAI embedder supports any OpenAI-compatible API via `base_url` -- llama.cpp, vLLM, Ollama, etc.

```yaml
dense_embedder:
  type: openai
  model: llama-3-8b
  base_url: http://localhost:8080/v1
  api_key: none                    # skip OPENAI_API_KEY check for local servers
  batch_size: 32
```

Set `api_key: none` for local servers that don't require auth. If omitted, the client reads `OPENAI_API_KEY` from the environment.

## Sentence Transformers

Runs models locally. Best for large datasets with GPU access.

```yaml
dense_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16                  # float32, float16, bfloat16
```

- Auto-detects CUDA, MPS (Apple Silicon), or CPU
- Supports bfloat16/float16 for faster inference
- Text splitting via the model's tokenizer
