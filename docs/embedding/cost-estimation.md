# Cost and Time Estimation

Before embedding a dataset, estimate how long it will take and what it will cost.

## Step 1: Profile the dataset

Use `token_stats.py` to sample token lengths:

```bash
python scripts/token_stats.py HuggingFaceFW/finewiki --config en --sample 100000
```

This gives you the **mean tokens per row** from a random sample.

## Step 2: Estimate total tokens

```
total_tokens = rows * mean_tokens_per_row
```

For a confidence interval:

```
standard_error = stdev / sqrt(sample_size)
total_tokens = rows * mean +/- rows * 1.96 * standard_error  (95% CI)
```

With a 100K sample, the standard error is typically small enough to ignore.

## Step 3: Estimate time and cost

```
gpu_hours = total_tokens / throughput_tok_s / 3600
cost      = gpu_hours * price_per_gpu_hour
wall_time = gpu_hours / num_gpus
```

## Reference values

| Parameter | Value | Notes |
|-----------|-------|-------|
| Throughput (gte-multilingual-base, A10G, bfloat16) | 50,000 tok/s | Budget estimate; benchmarks show 45K-87K depending on text length |
| Throughput (snowflake-arctic-embed-l-v2.0, A10G, bfloat16) | ~35,000 tok/s | Single GPU |
| A10G price (AWS) | ~$0.38/hr | Spot, g5.xlarge |
| T4 price (GCP) | ~$0.18/hr | Spot |
| OpenAI text-embedding-3-small | $0.02/1M tokens | API pricing |

Throughput varies with text length: short texts (~20 tok avg) achieve ~58K tok/s due to padding waste, while medium-to-long texts (~700+ tok avg) hit ~80-87K tok/s. Always use bfloat16 -- it's ~2x faster than float32 with no quality loss.

## Example

finewiki (English): 1.82M rows, mean 676 tok/row:

```
total_tokens = 1,820,000 * 676 = 1.23B tokens
gpu_hours    = 1,230,000,000 / 50,000 / 3600 = 6.8 GPU-hours
cost (AWS)   = 6.8 * $0.38 = $2.60  (spot g5.xlarge)
wall_time    = 6.8 / 10 GPUs = 41 min
```

Compare with OpenAI API:

```
cost (OpenAI) = 1,230,000,000 / 1,000,000 * $0.02 = $24.60
```

Self-hosted GPU embedding is ~10x cheaper than OpenAI at this scale.

## Padding waste

Transformer models pad every batch to the length of the longest sequence. Run the padding simulator to quantify waste:

```bash
python throughput_exp/padding_sim.py --dataset HuggingFaceFW/finewiki --hf-config en
```

Typical findings:

- **No truncation** (cutoff=8192): 15-25% efficiency
- **Cutoff=1024**: ~70% efficiency with ~70% of content retained
- **Cutoff=512**: ~87% efficiency with ~50% of content retained

Set `pipeline.max_text_length` in your config to apply a truncation cutoff.

## Caveats

- **Throughput varies by model and GPU.** Use `throughput_exp/bench.py` to measure yours.
- **Long texts create multiple chunks.** A 50K-token text becomes ~6 chunks at 8192 max_seq_length.
- **GPU utilization matters.** Add ~20% buffer for data loading, uploads, and container startup.
- **Cost scales linearly, wall time doesn't.** More GPUs reduces wall time but total cost stays the same.
