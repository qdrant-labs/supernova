# Cost and Time Estimation

Before embedding a dataset, estimate how long it will take and what it will cost. The fastest path is `vf throughput-predict`; the back-of-envelope arithmetic below is useful when you want a rough sense without running the predictor.

## `vf throughput-predict`

The predictor reads dataset, model, text rendering, and per-text cutoff straight from your embedder YAML. It tokenizes a sample of the source to build an empirical length distribution, simulates batched padding, and prints throughput + wall-clock + cost estimates.

```bash
# Single-cutoff estimate
vf throughput-predict configs/embedder/finewiki_en_all_mini.yaml --gpu a10g

# Sweep across cutoffs to find the padding sweet spot
vf throughput-predict configs/embedder/finewiki_en_all_mini.yaml \
  --gpu a10g --cutoff 256 --cutoff 512 --cutoff 1024

# Distributed wall-clock estimate (10× A10G in parallel)
vf throughput-predict configs/embedder/finewiki_en_all_mini.yaml \
  --gpu a10g --num-gpus 10

# Save JSON + plot
vf throughput-predict configs/embedder/finewiki_en_all_mini.yaml \
  --gpu a10g --output predictions.json
```

`vf throughput-predict --help` lists every override (model, total rows, batch size, sample size, GPU rate, …). The GPU catalogue lives in `vectorforge/throughput.py:GPU_TABLE`.

## Manual back-of-envelope

```
total_tokens = rows * mean_tokens_per_row
gpu_hours    = total_tokens / throughput_tok_s / 3600
cost         = gpu_hours * price_per_gpu_hour
wall_time    = gpu_hours / num_gpus
```

For a confidence interval on `mean_tokens_per_row`, profile a sample of the source with your model's tokenizer using vectorforge's parquet-level source:

```python
from transformers import AutoTokenizer
from vectorforge.sources.huggingface import HuggingFaceSource

tok = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
src = HuggingFaceSource(
    dataset_name="HuggingFaceFW/finewiki",
    split="train",
    path_filter="en",
    text_field="text",
    limit=100_000,
)
lengths = [len(tok.encode(r["text"])) for r in src.stream()]
mean_tokens = sum(lengths) / len(lengths)
```

With a 100k sample the standard error is small enough to ignore for planning purposes.

## Reference values

| Parameter | Value | Notes |
|-----------|-------|-------|
| Throughput (gte-multilingual-base, A10G, bfloat16) | ~50,000 tok/s | Budget estimate; benchmarks show 45K-87K depending on text length |
| Throughput (snowflake-arctic-embed-l-v2.0, A10G, bfloat16) | ~35,000 tok/s | Single GPU |
| A10G price (AWS spot, g5.xlarge) | ~$0.38/hr | |
| T4 price (GCP spot) | ~$0.18/hr | |
| OpenAI text-embedding-3-small | $0.02 / 1M tokens | API pricing |

Throughput varies with text length: short texts (~20 tok avg) achieve ~58K tok/s due to padding waste, while medium-to-long texts (~700+ tok avg) hit ~80-87K tok/s. Always use `bfloat16` on modern GPUs — it's ~2× faster than `float32` with no quality loss.

## Example

finewiki (English): 1.82M rows, mean 676 tok/row:

```
total_tokens = 1,820,000 * 676 = 1.23B tokens
gpu_hours    = 1,230,000,000 / 50,000 / 3600 = 6.8 GPU-hours
cost (AWS)   = 6.8 * $0.38 = $2.60   (spot g5.xlarge)
wall_time    = 6.8 / 10 GPUs = 41 min
```

Compare with OpenAI API:

```
cost (OpenAI) = 1,230,000,000 / 1,000,000 * $0.02 = $24.60
```

Self-hosted GPU embedding is ~10× cheaper than OpenAI at this scale.

## Padding waste

Transformer models pad every batch to the length of the longest sequence in that batch. `vf throughput-predict --cutoff <N1> --cutoff <N2> --cutoff <N3>` sweeps cutoffs and reports padding efficiency for each.

Typical findings:

- **No truncation** (cutoff ≥ 8192): 15-25% efficiency
- **Cutoff = 1024**: ~70% efficiency with ~70% of content retained
- **Cutoff = 512**: ~87% efficiency with ~50% of content retained

Set `dense_embedder.max_tokens` (or `multivector_embedder.max_tokens`) in your config to apply a truncation cutoff. Combine with `truncate: true` to drop content past the cutoff entirely; leave `truncate: false` (default) to split long docs into multiple records and embed each piece separately.

## Caveats

- **Throughput varies by model and GPU.** `throughput_exp/throughput_bench.py` measures the actual sustained tok/s on your hardware.
- **Long texts create multiple records.** A 50K-token text with `max_tokens=8192` becomes ~6 records (one embedding per piece) when `truncate: false`.
- **GPU utilization matters.** Add ~20% buffer for data loading, uploads, and container startup.
- **Cost scales linearly, wall time doesn't.** More GPUs reduces wall time but total cost stays the same.
