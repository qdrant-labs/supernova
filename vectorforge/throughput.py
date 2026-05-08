#!/usr/bin/env python3
"""
Unified embedding throughput prediction pipeline.

Samples a dataset, profiles the token length distribution, runs Monte Carlo
padding simulations, then predicts tok/s, texts/s, and cost for a given
model + GPU + cutoff combination -- no GPU required.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from scipy import stats as sp_stats
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from vectorforge.sources.huggingface import HuggingFaceSource

log = logging.getLogger(__name__)

GPU_TABLE: dict[str, dict] = {
    "b200": {
        "name": "NVIDIA B200",
        "effective_tflops_bf16": 880,
        "rate_per_hr": 6.2496,
    },
    "h200": {
        "name": "NVIDIA H200",
        "effective_tflops_bf16": 600,
        "rate_per_hr": 4.5396,
    },
    "h100": {
        "name": "NVIDIA H100",
        "effective_tflops_bf16": 395,
        "rate_per_hr": 3.9492,
    },
    "6000": {
        "name": "NVIDIA RTX PRO 6000",
        "effective_tflops_bf16": 200,
        "rate_per_hr": 3.0312,
    },
    "l40s": {
        "name": "NVIDIA L40S",
        "effective_tflops_bf16": 145,
        "rate_per_hr": 1.9512,
    },
    "a100": {
        "name": "NVIDIA A100",
        "effective_tflops_bf16": 125,
        "rate_per_hr": 2.0988,
    },
    "a10g": {"name": "NVIDIA A10G", "effective_tflops_bf16": 50, "rate_per_hr": 1.1016},
    "l4": {"name": "NVIDIA L4", "effective_tflops_bf16": 48, "rate_per_hr": 0.7992},
    "t4": {"name": "NVIDIA T4", "effective_tflops_bf16": 26, "rate_per_hr": 0.5904},
}


def detect_total_rows(dataset: str, hf_config: str | None, split: str) -> int | None:
    # hf_config is now folded into a path_filter glob if the caller relied on
    # it (e.g. "20231101.en" for wikipedia configs). Most native-parquet repos
    # use one parquet tree per split and ignore config entirely.
    try:
        src = HuggingFaceSource(
            dataset_name=dataset,
            split=split,
            path_filter=hf_config,
            text_field="__never__",  # placeholder; we never call extract_text here
        )
        n = src.get_total_rows()
        return n if n and n > 0 else None
    except Exception:
        return None


def _render_text(row: dict, column: str | None, template: str | None) -> str | None:
    if template:
        try:
            return template.format(**row)
        except (KeyError, IndexError):
            return None
    return row.get(column) if column else None


def sample_token_lengths(
    dataset: str,
    hf_config: str | None,
    split: str,
    column: str | None,
    template: str | None,
    tokenizer_name: str,
    n: int,
) -> np.ndarray:
    src = HuggingFaceSource(
        dataset_name=dataset,
        split=split,
        path_filter=hf_config,
        text_field=column,
        text_template=template,
        limit=n,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    lengths: list[int] = []
    for row in tqdm(src.stream(), total=n, desc="Tokenizing"):
        text = _render_text(row, column, template)
        if text and text.strip():
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))

    return np.array(lengths, dtype=np.int64)


def compute_token_stats(lengths: np.ndarray) -> dict:
    return {
        "count": len(lengths),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "stdev": float(lengths.std()),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def fit_lognormal(lengths: np.ndarray) -> dict:
    shape, loc, scale = sp_stats.lognorm.fit(lengths, floc=0)
    ks = sp_stats.kstest(lengths, "lognorm", args=(shape, loc, scale))
    return {
        "s": float(shape),
        "loc": float(loc),
        "scale": float(scale),
        "underlying_mu": float(np.log(scale)),
        "underlying_sigma": float(shape),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def simulate_padding(
    lengths: np.ndarray, cutoff: int, batch_size: int = 64, num_batches: int = 10_000
) -> dict:
    truncated = np.minimum(lengths, cutoff)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(truncated), size=(num_batches, batch_size))
    batches = truncated[indices]

    batch_maxes = batches.max(axis=1)
    batch_sums = batches.sum(axis=1)
    efficiencies = batch_sums / (batch_maxes * batch_size)

    eta = float(efficiencies.mean())
    return {
        "cutoff": cutoff,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "eta": eta,
        "eta_median": float(np.median(efficiencies)),
        "eta_p5": float(np.percentile(efficiencies, 5)),
        "padding_waste_pct": float((1 - eta) * 100),
        "tokens_retained_pct": float(truncated.sum() / lengths.sum() * 100),
        "pct_texts_truncated": float((lengths > cutoff).mean() * 100),
        "mean_truncated_tokens": float(truncated.mean()),
    }


def count_model_params(model_name: str) -> tuple[int, str]:
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        for attr in ("num_parameters", "n_params"):
            if hasattr(config, attr):
                n = getattr(config, attr)
                if n and n > 0:
                    return n, f"config.{attr}"
    except Exception:
        pass

    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        n = sum(p.numel() for p in model.parameters())
        del model
        return n, "model.parameters()"
    except Exception as e:
        raise RuntimeError(
            f"Could not determine parameter count for {model_name}. Error: {e}"
        )


def predict_throughput(
    params: int,
    gpu_tflops: float,
    cutoff: int,
    gpu_scale: float = 1.0,
    eta: float = 1.0,
    mean_tokens_per_text: float | None = None,
) -> dict:
    effective_tflops = gpu_tflops * gpu_scale
    t_max = effective_tflops * 1e12 / (2 * params)
    useful_tok_s = t_max * eta
    # actual texts/s is bounded by useful tokens per second divided by the tokens
    # we actually do work on per text (after truncation), not the padding-cutoff
    denom = mean_tokens_per_text if mean_tokens_per_text else cutoff
    return {
        "t_max_tok_s": t_max,
        "useful_tok_s": useful_tok_s,
        "texts_per_s": useful_tok_s / denom,
        "effective_tflops": effective_tflops,
    }


def estimate_cost(
    total_rows: int, texts_per_s: float, rate_per_hr: float, overhead: float = 1.2
) -> dict:
    gpu_seconds = total_rows / texts_per_s
    gpu_hours = gpu_seconds / 3600
    raw_cost = gpu_hours * rate_per_hr
    return {
        "gpu_hours": gpu_hours,
        "raw_cost": raw_cost,
        "total_cost": raw_cost * overhead,
        "overhead_factor": overhead,
        "wall_clock_hours": gpu_hours,
    }


def plot_distribution(
    lengths: np.ndarray, fit: dict, cutoff: int, output_path: str = None
) -> None:
    """
    Generates and saves a token distribution plot.
    """
    import matplotlib.pyplot as plt

    log.info("\nGenerating distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 5))

    # empirical histogram
    bins = np.logspace(np.log10(max(1, lengths.min())), np.log10(lengths.max()), 100)
    ax.hist(
        lengths,
        bins=bins,
        density=True,
        alpha=0.5,
        color="steelblue",
        label="Empirical",
    )

    # fitted curve
    x = np.logspace(np.log10(max(1, lengths.min())), np.log10(lengths.max()), 500)
    pdf = sp_stats.lognorm.pdf(x, fit["s"], fit["loc"], fit["scale"])
    ax.plot(x, pdf, "r-", lw=2, label=f"lognormal fit (KS={fit['ks_stat']:.3f})")

    # percentile markers
    for p, color in [(50, "green"), (95, "orange"), (99, "red")]:
        val = np.percentile(lengths, p)
        ax.axvline(val, color=color, ls="--", alpha=0.7, label=f"p{p}: {val:,.0f}")

    # cutoff marker
    ax.axvline(cutoff, color="black", ls="-", lw=2, label=f"Cutoff: {cutoff}")

    ax.set_xscale("log")
    ax.set_xlabel("Token count per text")
    ax.set_ylabel("Density")
    ax.set_title("Token Length Distribution & Truncation Cutoff")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        log.info(f"  Saved plot to {output_path}")
    else:
        plt.show()


def print_report(
    token_stats: dict,
    fit: dict,
    padding: dict,
    throughput: dict,
    cost: dict | None,
    rate_per_hr: float,
    gpu: dict,
    gpu_scale: float,
    model_name: str,
    params: int,
    params_method: str,
    dataset: str,
    config: str | None,
    column: str | None,
    template: str | None,
    cutoff: int,
    total_rows: int | None,
    total_rows_source: str | None,
    num_gpus: int | None,
) -> None:
    W = 64
    print(f"\n{'=' * W}")
    print("  THROUGHPUT PREDICTION REPORT")
    print(f"{'=' * W}")

    print("\n--- Dataset ---")
    print(f"  Dataset:          {dataset} (config={config})")
    if template:
        print(f"  Template:         {template!r}")
    else:
        print(f"  Column:           {column}")
    print(f"  Sampled:          {token_stats['count']:,} texts")

    print("\n--- Token Distribution ---")
    print(f"  Mean:             {token_stats['mean']:,.0f} tokens")
    print(f"  Median:           {token_stats['median']:,.0f} tokens")
    print(f"  Stdev:            {token_stats['stdev']:,.0f} tokens")
    print(f"  p95:              {token_stats['p95']:,.0f} tokens")
    print(f"  p99:              {token_stats['p99']:,.0f} tokens")
    print(f"  Max:              {token_stats['max']:,} tokens")
    print(
        f"  Fit:              lognormal (mu={fit['underlying_mu']:.2f}, sigma={fit['underlying_sigma']:.2f}, KS={fit['ks_stat']:.4f})"
    )

    print("\n--- Model ---")
    print(f"  Model:            {model_name}")
    print(f"  Parameters:       {params:,} ({params_method})")
    print(f"  FLOPs/token:      ~{2 * params:,}")

    print("\n--- GPU ---")
    print(f"  GPU:              {gpu['name']}")
    print(f"  Effective TFLOPS: {throughput['effective_tflops']:.0f} (bf16)")
    if gpu_scale != 1.0:
        print(f"  Scale factor:     {gpu_scale}")
    print(f"  Rate:             ${rate_per_hr:.4f}/hr")

    print(f"\n--- Padding Simulation (cutoff={cutoff}) ---")
    print(f"  Efficiency (eta): {padding['eta']:.1%}")
    print(f"  Padding waste:    {padding['padding_waste_pct']:.1f}%")
    print(
        f"  Tokens retained:  {padding['tokens_retained_pct']:.1f}%  (semantic coverage)"
    )
    print(f"  Texts truncated:  {padding['pct_texts_truncated']:.1f}%")

    print("\n--- Predicted Throughput ---")
    print(f"  T_max (tok/s):    {throughput['t_max_tok_s']:,.0f}")
    print(
        f"  Useful tok/s:     {throughput['t_max_tok_s'] * padding['eta']:,.0f}  (T_max * eta)"
    )
    print(f"  texts/s:          {throughput['texts_per_s']:,.0f}")
    print(f"  Cutoff:           {cutoff} tokens")

    if cost and total_rows:
        print("\n--- Cost Estimate ---")
        rows_label = f"{total_rows:,}" + (
            f"  ({total_rows_source})" if total_rows_source else ""
        )
        print(f"  Total rows:       {rows_label}")
        print(f"  GPU hours:        {cost['gpu_hours']:,.1f}")
        print(f"  Raw cost:         ${cost['raw_cost']:,.2f}")
        print(f"  With {cost['overhead_factor']}x overhead: ${cost['total_cost']:,.2f}")
        print(
            f"  Wall clock:       {cost['wall_clock_hours']:,.1f} hrs ({cost['wall_clock_hours'] / 24:.1f} days) [single GPU]"
        )
        if num_gpus and num_gpus > 1:
            parallel_hrs = cost["wall_clock_hours"] / num_gpus
            print(
                f"  With {num_gpus} GPUs:     {parallel_hrs:,.1f} hrs ({parallel_hrs / 24:.1f} days)"
            )
    print(f"\n{'=' * W}")


def normalize_gpu_key(gpu: str) -> str:
    """Normalize a user-supplied GPU label to a key in GPU_TABLE."""
    return gpu.lower().replace(" ", "").replace("-", "")


def run_prediction(
    *,
    dataset: str,
    model: str,
    cutoffs: list[int],
    hf_config: str | None = None,
    split: str = "train",
    column: str | None = None,
    template: str | None = None,
    sample: int = 100_000,
    gpu: str = "a10g",
    gpu_scale: float = 1.0,
    rate: float | None = None,
    batch_size: int = 64,
    num_batches: int = 10_000,
    total_rows: int | None = None,
    total_rows_source: str | None = None,
    num_gpus: int | None = None,
    overhead: float = 1.2,
    params: int | None = None,
    output: str | None = None,
) -> list[dict]:
    """
    End-to-end prediction pipeline. Returns one result dict per cutoff.

    Required: ``dataset``, ``model``, ``cutoffs``. Provide exactly one of
    ``column`` or ``template`` to render rows into text. ``hf_config`` is the
    HuggingFace dataset config (e.g. ``"20231101.en"`` for wikimedia/wikipedia),
    not a file path.
    """
    if bool(column) == bool(template):
        raise ValueError("Provide exactly one of `column` or `template`")
    if not cutoffs:
        raise ValueError("`cutoffs` must contain at least one value")

    gpu_key = normalize_gpu_key(gpu)
    if gpu_key not in GPU_TABLE:
        raise ValueError(
            f"Unknown GPU {gpu!r}. Choose from: {', '.join(GPU_TABLE.keys())}"
        )
    gpu_spec = GPU_TABLE[gpu_key]
    rate_per_hr = rate if rate is not None else gpu_spec["rate_per_hr"]

    cutoffs = sorted(cutoffs)
    t_start = time.perf_counter()

    log.info("[1/4] Sampling %s rows...", f"{sample:,}")
    lengths = sample_token_lengths(
        dataset, hf_config, split, column, template, model, sample
    )
    token_stats = compute_token_stats(lengths)

    log.info("\n[2/4] Fitting lognormal distribution...")
    fit = fit_lognormal(lengths)

    if params is not None:
        params_method = "user-provided"
    else:
        log.info("\n[3/4] Counting parameters...")
        params, params_method = count_model_params(model)

    if total_rows is not None and total_rows_source is None:
        total_rows_source = "user-provided"
    if total_rows is None:
        total_rows = detect_total_rows(dataset, hf_config, split)
        total_rows_source = "dataset metadata" if total_rows else None

    log.info(
        "\n[4/4] Simulating & predicting for %d cutoff(s): %s", len(cutoffs), cutoffs
    )
    results = []
    for cutoff in cutoffs:
        padding = simulate_padding(lengths, cutoff, batch_size, num_batches)
        throughput = predict_throughput(
            params,
            gpu_spec["effective_tflops_bf16"],
            cutoff,
            gpu_scale,
            eta=padding["eta"],
            mean_tokens_per_text=padding["mean_truncated_tokens"],
        )
        cost = (
            estimate_cost(total_rows, throughput["texts_per_s"], rate_per_hr, overhead)
            if total_rows
            else None
        )

        plot_distribution(lengths, fit, cutoff, output)
        print_report(
            token_stats,
            fit,
            padding,
            throughput,
            cost,
            rate_per_hr,
            gpu_spec,
            gpu_scale,
            model,
            params,
            params_method,
            dataset,
            hf_config,
            column,
            template,
            cutoff,
            total_rows,
            total_rows_source,
            num_gpus,
        )
        results.append(
            {
                "cutoff": cutoff,
                "prediction": throughput,
                "cost": cost,
                "padding": padding,
            }
        )

    log.info(f"\nTotal prediction time: {time.perf_counter() - t_start:.1f} seconds")

    if output:
        with open(output, "w") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "hf_config": hf_config,
                    "model": model,
                    "gpu": gpu_spec["name"],
                    "results": results,
                },
                f,
                indent=2,
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict embedding throughput and cost."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--column", default=None, help="Field name in the dataset row to tokenize"
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Python str.format() template rendered against each row, e.g. "
        "'Title: {title}\\n\\nReview: {description}'. Use instead of --column.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--params", type=int, default=None)
    parser.add_argument("--gpu", default="a10g")
    parser.add_argument("--gpu-scale", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--cutoff", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=10_000)
    parser.add_argument("--total-rows", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--overhead", type=float, default=1.2)
    parser.add_argument("--output", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    hf_config = None if args.config in (None, "None") else args.config

    try:
        run_prediction(
            dataset=args.dataset,
            model=args.model,
            cutoffs=args.cutoff,
            hf_config=hf_config,
            split=args.split,
            column=args.column,
            template=args.template,
            sample=args.sample,
            gpu=args.gpu,
            gpu_scale=args.gpu_scale,
            rate=args.rate,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            total_rows=args.total_rows,
            num_gpus=args.num_gpus,
            overhead=args.overhead,
            params=args.params,
            output=args.output,
        )
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
