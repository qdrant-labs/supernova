# Dense Embedders

Dense backends are selected per entry in the `embedders:` list with `kind: dense` plus a `type`. Every entry also declares its `input_column` and `modality` — see the [overview](overview.md#configuration).

## OpenAI

Uses the OpenAI API. Best for smaller datasets or when you don't have GPUs.

```yaml
embedders:
  - name: openai_small
    kind: dense
    type: openai
    model: text-embedding-3-small  # or text-embedding-3-large
    input_column: text
    modality: text
    dimensions: 1536               # optional dimension selection
    batch_size: 128
    max_concurrent: 8              # parallel API calls
```

- Rate limiting with exponential backoff
- Text splitting via tiktoken
- Max 8192 tokens per text

### OpenAI-compatible APIs

The OpenAI embedder supports any OpenAI-compatible API via `base_url` -- llama.cpp, vLLM, Ollama, etc.

```yaml
embedders:
  - name: llama
    kind: dense
    type: openai
    model: llama-3-8b
    input_column: text
    modality: text
    base_url: http://localhost:8080/v1
    api_key: none                  # skip OPENAI_API_KEY check for local servers
    batch_size: 32
```

Set `api_key: none` for local servers that don't require auth. If omitted, the client reads `OPENAI_API_KEY` from the environment.

## Sentence Transformers

Runs models locally. Best for large datasets with GPU access.

```yaml
embedders:
  - name: gte
    kind: dense
    type: sentence_transformer
    model: Alibaba-NLP/gte-multilingual-base
    input_column: text
    modality: text
    trust_remote_code: true
    batch_size: 64
    dtype: bfloat16                # float32, float16, bfloat16
```

- Auto-detects CUDA, MPS (Apple Silicon), or CPU
- Supports bfloat16/float16 for faster inference
- Text splitting via the model's tokenizer

### Images (CLIP-family models)

The same backend embeds images when the model supports it (`clip-ViT-*`, jina-clip, …) — declare `modality: image` and point `input_column` at a column holding file paths, raw bytes, or HuggingFace `{bytes, path}` image structs:

```yaml
embedders:
  - name: clip
    kind: dense
    type: sentence_transformer
    model: clip-ViT-B-32
    input_column: image
    modality: image
```

Transport decoding (path vs bytes vs struct) is handled by the pipeline; the model always receives decoded images. A text-only model fed images fails loudly in the forward pass.
