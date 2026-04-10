# Throughput Benchmark Results

Rough benchmarks to establish a baseline tok/s for cost estimation. Run on a single AWS g5.xlarge (NVIDIA A10G, 24GB VRAM).

## Setup

- **Instance**: g5.xlarge (A10G)
- **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Amazon Linux 2023)
- **Sample size**: 10,000 rows per dataset (random, streamed)
- **Metric**: Tokens/sec measured end-to-end via `SentenceTransformer.encode()`

## Results

### gte-multilingual-base (bfloat16)

| Dataset | Mean tok/row | Batch size | tok/s | texts/s | VRAM GB |
|---------|-------------|-----------|-------|---------|---------|
| laion/aesthetics_v2_4.75 | 19 | 32 | 47,975 | 2,541 | 1.1 |
| laion/aesthetics_v2_4.75 | 19 | 64 | 55,743 | 2,952 | 1.6 |
| laion/aesthetics_v2_4.75 | 19 | 128 | 57,892 | 3,066 | 2.6 |
| laion/aesthetics_v2_4.75 | 19 | 256 | 53,776 | 2,848 | 4.7 |
| laion/aesthetics_v2_4.75 | 19 | 512 | 45,836 | 2,428 | 8.7 |
| HuggingFaceFW/finewiki | 676 | 8 | 81,286 | 136 | 2.9 |
| HuggingFaceFW/finewiki | 676 | 16 | 81,231 | 136 | 5.1 |
| HuggingFaceFW/finewiki | 676 | 32 | 77,916 | 130 | 9.6 |
| HuggingFaceFW/finewiki | 676 | 64 | 74,741 | 125 | 18.5 |
| HuggingFaceFW/finephrase | 1,132 | 8 | 87,237 | 80 | 2.9 |
| HuggingFaceFW/finephrase | 1,132 | 16 | 87,297 | 80 | 5.1 |
| HuggingFaceFW/finephrase | 1,132 | 32 | 85,811 | 79 | 9.6 |
| HuggingFaceFW/finephrase | 1,132 | 64 | 83,181 | 76 | 18.5 |

### gte-multilingual-base (float32)

| Dataset | Mean tok/row | Batch size | tok/s | texts/s | VRAM GB |
|---------|-------------|-----------|-------|---------|---------|
| laion/aesthetics_v2_4.75 | 19 | 64 | 28,266 | 1,497 | 3.3 |
| laion/aesthetics_v2_4.75 | 19 | 128 | 25,266 | 1,338 | 5.3 |
| laion/aesthetics_v2_4.75 | 19 | 256 | 22,160 | 1,174 | 9.3 |
| laion/aesthetics_v2_4.75 | 19 | 512 | 17,463 | 925 | 17.3 |

### snowflake-arctic-embed-l-v2.0 (bfloat16)

| Dataset | Mean tok/row | Batch size | tok/s | texts/s | VRAM GB |
|---------|-------------|-----------|-------|---------|---------|
| HuggingFaceFW/finewiki | 676 | 4 | 35,269 | 59 | 3.2 |
| HuggingFaceFW/finewiki | 676 | 8 | 35,029 | 58 | 5.2 |
| HuggingFaceFW/finewiki | 676 | 16 | 33,830 | 56 | 9.2 |
| HuggingFaceFW/finewiki | 676 | 32 | 30,958 | 52 | 17.3 |

## Observations

- **bfloat16 is ~2x faster than float32** with no embedding quality loss.
- **Throughput scales with text length**: short texts (~20 tok) waste GPU cycles on padding (~58K tok/s peak), while longer texts (~700+ tok) saturate the GPU better (~80-87K tok/s).
- **Throughput drops at large batch sizes** due to VRAM pressure and diminishing returns on parallelism. Sweet spot varies by text length.
- **Snowflake arctic-embed-l is ~2.4x slower** than gte-multilingual-base (larger model).
- Results are broadly consistent with reported benchmarks of 100K-200K tok/s for optimized production setups on A10G/H200 hardware.

## Budget assumption

Given the range of 45K-87K tok/s depending on dataset and config, we use a conservative **50,000 tok/s per A10G** for cost estimation. This accounts for:

- Mixed text lengths across datasets
- Container startup and data loading overhead
- Non-optimal batch sizes in the production pipeline
- Network I/O for uploading results

Cost formula: `gpu_hours = total_tokens / 50,000 / 3600`