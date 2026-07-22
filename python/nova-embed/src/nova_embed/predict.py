"""Embedding throughput & cost prediction from an embed config — no GPU needed.

Resurrected from the pre-restructure ``supernova.throughput`` module and
adapted to the modern config schema. The formula is unchanged:

    T_max tok/s  = GPU effective TFLOPS x 1e12 / (2 x params)
    useful tok/s = T_max x eta   (eta = Monte Carlo padding efficiency over
                                  the dataset's empirical token distribution)
    texts/s      = useful tok/s / mean tokens per text (post-truncation)
    cost         = rows / texts_per_s -> GPU-hours x $/hr x overhead

New versus the original: prediction is per FORWARD PASS, planned with the
same fusion grouping the engine uses (``_fusion_groups``) — a fused bge-m3
dense+sparse+multivector config predicts ONE pass, not three. Per-pass rates
combine harmonically into a whole-pipeline texts/s (the engine runs its units
sequentially per chunk).

Only text-modality entries participate: the FLOPs model is token-based.
Image/multimodal entries are reported as skipped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np

from nova_embed.config import EmbedConfig
from nova_embed.embedders.engine import _fusion_groups
from nova_embed.media import Modality
from nova_embed.registry import SOURCES

log = logging.getLogger(__name__)

GPU_TABLE: dict[str, dict] = {
    "b200": {"name": "NVIDIA B200", "effective_tflops_bf16": 880, "rate_per_hr": 6.2496},
    "h200": {"name": "NVIDIA H200", "effective_tflops_bf16": 600, "rate_per_hr": 4.5396},
    "h100": {"name": "NVIDIA H100", "effective_tflops_bf16": 395, "rate_per_hr": 3.9492},
    "6000": {"name": "NVIDIA RTX PRO 6000", "effective_tflops_bf16": 200, "rate_per_hr": 3.0312},
    "l40s": {"name": "NVIDIA L40S", "effective_tflops_bf16": 145, "rate_per_hr": 1.9512},
    "a100": {"name": "NVIDIA A100", "effective_tflops_bf16": 125, "rate_per_hr": 2.0988},
    "a10g": {"name": "NVIDIA A10G", "effective_tflops_bf16": 50, "rate_per_hr": 1.1016},
    "l4": {"name": "NVIDIA L4", "effective_tflops_bf16": 48, "rate_per_hr": 0.7992},
    "t4": {"name": "NVIDIA T4", "effective_tflops_bf16": 26, "rate_per_hr": 0.5904},
}

DEFAULT_BATCH_SIZE = 64


def normalize_gpu_key(gpu: str) -> str:
    """
    Normalize a user-supplied GPU label to a key in GPU_TABLE.
    """
    return gpu.lower().replace(" ", "").replace("-", "")


@dataclass
class PassPlan:
    """
    One forward pass per chunk: the unit the throughput model prices.
    """

    entries: list[str]  # entry names served by this pass
    model: str | None
    input_column: str
    max_length: int | None  # per-entry char truncation, applied before tokenizing
    batch_size: int
    fused: bool

    @property
    def label(self) -> str:
        joined = " + ".join(self.entries)
        return f"{joined}  [fused]" if self.fused else joined


def plan_passes(cfg: EmbedConfig) -> tuple[list[PassPlan], list[str]]:
    """
    (passes, skipped-entry notes) — mirrors build_engine's unit assembly,
    without instantiating anything.
    """
    passes: list[PassPlan] = []
    skipped: list[str] = []

    groups = _fusion_groups(cfg.embedders)
    fused_names = {e.name for _, group in groups for e in group}

    def batch_of(entry) -> int | None:
        return entry.backend_kwargs().get("batch_size")

    for _, group in groups:
        e0 = group[0]
        sizes = [b for e in group if (b := batch_of(e)) is not None]
        passes.append(
            PassPlan(
                entries=[e.name for e in group],
                model=e0.model,
                input_column=e0.input_column,
                max_length=e0.max_length,
                batch_size=min(sizes) if sizes else DEFAULT_BATCH_SIZE,
                fused=True,
            )
        )

    for e in cfg.embedders:
        if e.name in fused_names:
            continue
        if e.modality != Modality.TEXT:
            skipped.append(
                f"{e.name} (modality={e.modality.value}: the FLOPs model is token-based)"
            )
            continue
        if e.model is None:
            skipped.append(f"{e.name} (no model field to price)")
            continue
        passes.append(
            PassPlan(
                entries=[e.name],
                model=e.model,
                input_column=e.input_column,
                max_length=e.max_length,
                batch_size=batch_of(e) or DEFAULT_BATCH_SIZE,
                fused=False,
            )
        )
    return passes, skipped

def sample_texts(cfg: EmbedConfig, columns: set[str], n: int) -> dict[str, list[str]]:
    """
    Stream the first `n` rows of the config's source; collect each needed
    input column's non-empty texts (after render_columns, via format_record).
    """
    from tqdm import tqdm

    source_dict = cfg.source.build_dict()
    source_dict["limit"] = n
    source_dict.setdefault("required_columns", sorted(columns))
    source = SOURCES.build(source_dict)

    texts: dict[str, list[str]] = {c: [] for c in columns}
    for row in tqdm(source.stream(), total=n, desc="Sampling rows"):
        rendered = source.format_record(row).row
        for col in columns:
            v = rendered.get(col)
            if isinstance(v, str) and v.strip():
                texts[col].append(v)
    return texts


def token_lengths(
    tokenizer_name: str, texts: list[str], max_length: int | None
) -> np.ndarray:
    from tqdm import tqdm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    lengths = [
        len(tokenizer.encode(t[:max_length] if max_length else t, add_special_tokens=False))
        for t in tqdm(texts, desc=f"Tokenizing ({tokenizer_name})")
    ]
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


def fit_lognormal(lengths: np.ndarray) -> dict | None:
    """Lognormal fit + KS stat, for the report. Returns None without scipy —
    the fit is descriptive only; nothing downstream depends on it."""
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return None
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
    """
    Monte Carlo padding efficiency: draw random batches from the empirical
    (truncated) length distribution; eta = real tokens / padded tokens.
    """
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
    """
    Parameter count, cheapest source first: hub safetensors metadata (no
    download), then a full AutoModel load as last resort.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_name)
        st = getattr(info, "safetensors", None)
        total = getattr(st, "total", None) if st else None
        if total:
            return int(total), "hub safetensors metadata"
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
            f"Could not determine parameter count for {model_name} "
            f"(pass --params to skip the lookup): {e}"
        )


def model_max_tokens(model_name: str) -> int | None:
    """
    The model's positional limit, from config alone (no weights).
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        v = getattr(config, attr, None)
        if v and v > 0:
            return int(v)
    return None


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
    # texts/s is bounded by useful tokens/s over the tokens we actually do work
    # on per text (post-truncation mean), not the padding cutoff
    denom = mean_tokens_per_text if mean_tokens_per_text else cutoff
    return {
        "t_max_tok_s": t_max,
        "useful_tok_s": useful_tok_s,
        "texts_per_s": useful_tok_s / denom,
        "effective_tflops": effective_tflops,
    }


def combine_texts_per_s(rates: list[float]) -> float:
    """
    Passes run sequentially per chunk, so pipeline rate combines harmonically.
    """
    return 1.0 / sum(1.0 / r for r in rates)


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

@dataclass
class PassResult:
    plan: PassPlan
    params: int
    params_method: str
    cutoff: int
    token_stats: dict
    fit: dict | None
    padding: dict
    throughput: dict
    lengths: np.ndarray = field(repr=False, default=None)


def _print_pass(r: PassResult) -> None:
    ts, pad, thr = r.token_stats, r.padding, r.throughput
    print(f"\n--- Pass: {r.plan.label} ---")
    print(f"  Model:            {r.plan.model}  ({r.params:,} params, {r.params_method})")
    print(f"  Input:            {r.plan.input_column}  ({ts['count']:,} sampled)")
    print(
        f"  Tokens:           mean={ts['mean']:,.0f}  median={ts['median']:,.0f}  "
        f"p95={ts['p95']:,.0f}  p99={ts['p99']:,.0f}  max={ts['max']:,}"
    )
    if r.fit:
        print(
            f"  Fit:              lognormal (mu={r.fit['underlying_mu']:.2f}, "
            f"sigma={r.fit['underlying_sigma']:.2f}, KS={r.fit['ks_stat']:.4f})"
        )
    print(
        f"  Padding (cutoff={r.cutoff}, batch={pad['batch_size']}): "
        f"eta={pad['eta']:.1%}  waste={pad['padding_waste_pct']:.1f}%  "
        f"truncated={pad['pct_texts_truncated']:.1f}% of texts"
    )
    print(
        f"  Throughput:       T_max={thr['t_max_tok_s']:,.0f} tok/s  "
        f"useful={thr['useful_tok_s']:,.0f} tok/s  ->  {thr['texts_per_s']:,.0f} texts/s"
    )


def print_report(
    *,
    config_label: str,
    dataset: str,
    gpu: dict,
    gpu_scale: float,
    rate_per_hr: float,
    results: list[PassResult],
    skipped: list[str],
    combined_texts_per_s: float,
    cost: dict | None,
    total_rows: int | None,
    total_rows_source: str | None,
    num_gpus: int | None,
) -> None:
    W = 64
    print(f"\n{'=' * W}")
    print("  THROUGHPUT PREDICTION REPORT")
    print(f"{'=' * W}")
    print(f"  Config:           {config_label}")
    print(f"  Dataset:          {dataset}")
    print(f"  GPU:              {gpu['name']}  "
          f"({gpu['effective_tflops_bf16'] * gpu_scale:.0f} effective TFLOPS bf16"
          f"{f', scale={gpu_scale}' if gpu_scale != 1.0 else ''})")
    print(f"  Rate:             ${rate_per_hr:.4f}/hr")
    print(f"  Forward passes:   {len(results)}")

    for r in results:
        _print_pass(r)

    for note in skipped:
        print(f"\n  (skipped: {note})")

    print(f"\n--- Pipeline ({len(results)} sequential pass(es) per chunk) ---")
    print(f"  Combined:         {combined_texts_per_s:,.0f} texts/s")

    if cost and total_rows:
        rows_label = f"{total_rows:,}" + (
            f"  ({total_rows_source})" if total_rows_source else ""
        )
        print("\n--- Cost Estimate ---")
        print(f"  Total rows:       {rows_label}")
        print(f"  GPU hours:        {cost['gpu_hours']:,.1f}")
        print(f"  Raw cost:         ${cost['raw_cost']:,.2f}")
        print(f"  With {cost['overhead_factor']}x overhead: ${cost['total_cost']:,.2f}")
        print(
            f"  Wall clock:       {cost['wall_clock_hours']:,.1f} hrs "
            f"({cost['wall_clock_hours'] / 24:.1f} days) [single GPU]"
        )
        if num_gpus and num_gpus > 1:
            parallel = cost["wall_clock_hours"] / num_gpus
            print(f"  With {num_gpus} GPUs:     {parallel:,.1f} hrs ({parallel / 24:.1f} days)")
    print(f"\n{'=' * W}")


def plot_distribution(
    lengths: np.ndarray, fit: dict | None, cutoff: int, output_path: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(np.log10(max(1, lengths.min())), np.log10(lengths.max()), 100)
    ax.hist(lengths, bins=bins, density=True, alpha=0.5, color="steelblue", label="Empirical")
    if fit:
        try:
            from scipy import stats as sp_stats

            x = np.logspace(np.log10(max(1, lengths.min())), np.log10(lengths.max()), 500)
            pdf = sp_stats.lognorm.pdf(x, fit["s"], fit["loc"], fit["scale"])
            ax.plot(x, pdf, "r-", lw=2, label=f"lognormal fit (KS={fit['ks_stat']:.3f})")
        except ImportError:
            pass
    for p, color in [(50, "green"), (95, "orange"), (99, "red")]:
        val = np.percentile(lengths, p)
        ax.axvline(val, color=color, ls="--", alpha=0.7, label=f"p{p}: {val:,.0f}")
    ax.axvline(cutoff, color="black", ls="-", lw=2, label=f"Cutoff: {cutoff}")
    ax.set_xscale("log")
    ax.set_xlabel("Token count per text")
    ax.set_ylabel("Density")
    ax.set_title("Token Length Distribution & Truncation Cutoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to %s", output_path)

def run_prediction(
    cfg: EmbedConfig,
    *,
    config_label: str,
    gpu: str = "a10g",
    gpu_scale: float = 1.0,
    rate: float | None = None,
    num_gpus: int | None = None,
    overhead: float = 1.2,
    cutoff: int | None = None,
    sample: int = 100_000,
    batch_size: int | None = None,
    num_batches: int = 10_000,
    total_rows: int | None = None,
    params: int | None = None,
    plot: str | None = None,
    output: str | None = None,
) -> dict:
    """
    Predict throughput + cost for every forward pass an embed config implies.

    `cutoff` / `batch_size` / `params` override the per-pass derived values
    for ALL passes (params only makes sense with a single distinct model).
    Returns the full result dict (also written to `output` as JSON if given).
    """
    gpu_key = normalize_gpu_key(gpu)
    if gpu_key not in GPU_TABLE:
        raise ValueError(f"Unknown GPU {gpu!r}. Choose from: {', '.join(GPU_TABLE)}")
    gpu_info = GPU_TABLE[gpu_key]
    rate_per_hr = rate if rate is not None else gpu_info["rate_per_hr"]

    passes, skipped = plan_passes(cfg)
    if not passes:
        raise ValueError(
            "No predictable forward passes in this config "
            f"(skipped: {skipped or 'nothing — no entries?'})"
        )
    models = {p.model for p in passes}
    if params is not None and len(models) > 1:
        raise ValueError(
            f"--params is ambiguous with {len(models)} distinct models ({sorted(models)})"
        )

    # one sampling stream for all passes; one tokenization per distinct
    # (model, column, max_length)
    columns = {p.input_column for p in passes}
    texts = sample_texts(cfg, columns, sample)
    for col, t in texts.items():
        if not t:
            raise ValueError(f"Sampled 0 non-empty texts from column {col!r}")

    lengths_cache: dict[tuple, np.ndarray] = {}
    params_cache: dict[str, tuple[int, str]] = {}
    results: list[PassResult] = []
    for p in passes:
        lkey = (p.model, p.input_column, p.max_length)
        if lkey not in lengths_cache:
            lengths_cache[lkey] = token_lengths(p.model, texts[p.input_column], p.max_length)
        lengths = lengths_cache[lkey]

        if params is not None:
            n_params, method = params, "--params"
        else:
            if p.model not in params_cache:
                params_cache[p.model] = count_model_params(p.model)
            n_params, method = params_cache[p.model]

        pass_cutoff = cutoff or model_max_tokens(p.model)
        if not pass_cutoff:
            raise ValueError(
                f"No cutoff for {p.model!r}: its config exposes no positional "
                f"limit — pass --cutoff N"
            )

        padding = simulate_padding(
            lengths, pass_cutoff, batch_size or p.batch_size, num_batches
        )
        throughput = predict_throughput(
            n_params,
            gpu_info["effective_tflops_bf16"],
            pass_cutoff,
            gpu_scale=gpu_scale,
            eta=padding["eta"],
            mean_tokens_per_text=padding["mean_truncated_tokens"],
        )
        results.append(
            PassResult(
                plan=p,
                params=n_params,
                params_method=method,
                cutoff=pass_cutoff,
                token_stats=compute_token_stats(lengths),
                fit=fit_lognormal(lengths),
                padding=padding,
                throughput=throughput,
                lengths=lengths,
            )
        )

    combined = combine_texts_per_s([r.throughput["texts_per_s"] for r in results])

    total_rows_source = "--total-rows" if total_rows is not None else None
    if total_rows is None:
        source_dict = cfg.source.build_dict()
        if source_dict.get("total_rows_override"):
            total_rows = int(source_dict["total_rows_override"])
            total_rows_source = "config total_rows_override"
        else:
            try:
                total_rows = SOURCES.build(source_dict).get_total_rows()
                total_rows_source = "source metadata"
            except Exception as e:
                log.warning("Could not determine total rows (%s); skipping cost", e)

    cost = (
        estimate_cost(total_rows, combined, rate_per_hr, overhead)
        if total_rows
        else None
    )

    print_report(
        config_label=config_label,
        dataset=cfg.source.build_dict().get("dataset_name", cfg.source.type),
        gpu=gpu_info,
        gpu_scale=gpu_scale,
        rate_per_hr=rate_per_hr,
        results=results,
        skipped=skipped,
        combined_texts_per_s=combined,
        cost=cost,
        total_rows=total_rows,
        total_rows_source=total_rows_source,
        num_gpus=num_gpus,
    )

    if plot:
        from pathlib import Path

        base = Path(plot)
        for i, r in enumerate(results):
            path = base if len(results) == 1 else base.with_stem(f"{base.stem}_pass{i}")
            plot_distribution(r.lengths, r.fit, r.cutoff, str(path))

    payload = {
        "gpu": gpu_key,
        "gpu_scale": gpu_scale,
        "rate_per_hr": rate_per_hr,
        "combined_texts_per_s": combined,
        "total_rows": total_rows,
        "cost": cost,
        "passes": [
            {
                "entries": r.plan.entries,
                "fused": r.plan.fused,
                "model": r.plan.model,
                "input_column": r.plan.input_column,
                "batch_size": r.padding["batch_size"],
                "cutoff": r.cutoff,
                "params": r.params,
                "token_stats": r.token_stats,
                "lognormal_fit": r.fit,
                "padding": r.padding,
                "throughput": r.throughput,
            }
            for r in results
        ],
        "skipped_entries": skipped,
    }
    if output:
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Wrote results to %s", output)
    return payload
