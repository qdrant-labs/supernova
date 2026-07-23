# Throughput Prediction

`nova embed predict` estimates what an embedding run will do **before you commit GPUs**: texts/second, GPU-hours, and dollars — computed entirely from the dataset's token distribution, the model's parameter count, and a GPU spec table. No GPU required; nothing is embedded.

```bash
nova embed predict configs/embedder/my_dataset.yaml --gpu h100 --num-gpus 8
```

It reads the same config as `nova embed run` — the dataset, models, input columns, batch sizes, and truncation all come from the YAML, so the prediction prices exactly the run you'd launch. Only GPU, cost, and simulation knobs are flags.

## Example

```console
$ nova embed predict configs/embedder/example_fused.yaml --gpu h100 --batch-size 128 --sample 2000
================================================================
  THROUGHPUT PREDICTION REPORT
================================================================
  Config:           configs/embedder/example_fused.yaml
  Dataset:          mteb/tweet_sentiment_extraction
  GPU:              NVIDIA H100  (395 effective TFLOPS bf16)
  Rate:             $3.9492/hr
  Forward passes:   1

--- Pass: bge_m3_dense + bge_m3_sparse + bge_m3_multivector  [fused] ---
  Model:            BAAI/bge-m3  (567,754,752 params, model.parameters())
  Input:            text  (2,000 sampled)
  Tokens:           mean=19  median=18  p95=37  p99=42  max=57
  Fit:              lognormal (mu=2.81, sigma=0.59, KS=0.0842)
  Padding (cutoff=8194, batch=128): eta=43.4%  waste=56.6%  truncated=0.0% of texts
  Throughput:       T_max=347,861 tok/s  useful=151,111 tok/s  ->  7,800 texts/s

--- Pipeline (1 sequential pass(es) per chunk) ---
  Combined:         7,800 texts/s
```

Two things this example shows well:

- **Fusion is priced as fusion.** The three bge-m3 entries [fuse into one forward pass](overview.md#configuration), so the report plans one pass, not three. Break the fusion (different models, different inputs) and you get one pass per entry, each priced separately.
- **Padding efficiency is where throughput goes to die.** Tweets average 19 tokens, but batches of 128 pad every text to the batch's longest — here that wastes 56.6% of the compute (`eta=43.4%`). The padding simulation makes this visible *before* the run.

## How it works

The prediction is four steps, each visible in the report:

**1. Sample the token distribution.** Stream the first `--sample` rows (default 100k) from the config's source, render any `render_columns`, and tokenize each entry's input column with **that model's own tokenizer**. Per-entry `max_length` character truncation is applied first, exactly as the pipeline would.

**2. Simulate padding (Monte Carlo).** Draw `--num-batches` random batches (default 10k) of the config's `batch_size` from the empirical length distribution, truncated at the cutoff. Each batch pads to its longest member, so its efficiency is `real tokens / (batch_max × batch_size)`; the mean over batches is **η (eta)**, the fraction of compute doing useful work.

**3. Price each forward pass.** The compute ceiling uses the standard 2-FLOPs-per-parameter-per-token estimate:

```
T_max tok/s   = GPU effective TFLOPS × 1e12 / (2 × params)
useful tok/s  = T_max × η
texts/s       = useful tok/s ÷ mean tokens per text (post-truncation)
```

Passes are planned with the **same fusion grouping the engine uses**, so the prediction always matches what `nova embed run` would actually execute. Parameter counts come from the Hub's safetensors metadata when available (no download), falling back to loading the model; `--params N` skips the lookup. The cutoff defaults to each model's positional limit; override with `--cutoff`.

**4. Combine and cost.** The engine runs its passes sequentially per chunk, so per-pass rates combine harmonically (`1 / Σ(1/rateᵢ)`) into a whole-pipeline texts/s. Cost is then `total_rows ÷ texts/s` → GPU-hours × the GPU's $/hr × `--overhead` (default 1.2×), with a wall-clock divide for `--num-gpus`.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--gpu` | `a10g` | GPU key: `b200`, `h200`, `h100`, `6000`, `l40s`, `a100`, `a10g`, `l4`, `t4` |
| `--gpu-scale` | `1.0` | Multiplier on effective TFLOPS (e.g. `0.85` for thermal headroom) |
| `--rate` | GPU table | $/hr override |
| `--num-gpus` | 1 | Parallel GPUs for the wall-clock estimate |
| `--overhead` | `1.2` | Cost overhead multiplier (startup, retries, stragglers) |
| `--cutoff` | model limit | Token cutoff for the padding simulation |
| `--batch-size` | config entries | Override the per-pass batch size |
| `--sample` | `100000` | Rows tokenized for the empirical distribution |
| `--num-batches` | `10000` | Synthetic batches drawn in the simulation |
| `--total-rows` | source metadata | Row count override (skips the footer sweep on huge datasets) |
| `--params` | auto | Model parameter count override |
| `--plot PATH` | — | Save token-distribution plot(s) (requires matplotlib) |
| `--output PATH` | — | Write the full results as JSON |

The GPU table's `effective_tflops_bf16` values are *achieved* encoder-workload throughput, not datasheet peaks — tune with `--gpu-scale` if your measurements disagree.

## What it can and can't price

- **Text entries only.** The FLOPs model is token-based; image and multimodal entries are listed as skipped in the report and excluded from the totals.
- **Compute-bound assumption.** The model prices the forward pass. Runs bottlenecked elsewhere — dataset download, tokenization on a weak CPU, parquet upload — will come in under the prediction. The 1.2× overhead default absorbs the usual amount of this.
- **It's an estimate.** Expect it to land within a few tens of percent, and treat it as a *comparator* first: cutoff A vs cutoff B, one GPU type vs another, fused vs unfused — those ratios are far more reliable than any absolute number.

## Sweeping and scripting

`--output results.json` writes everything the report prints (per-pass token stats, the lognormal fit, η, throughput, cost) as JSON, so sweeps are a shell loop:

```bash
for gpu in h100 l40s a10g; do
  nova embed predict cfg.yaml --gpu $gpu --output pred_$gpu.json
done
```
