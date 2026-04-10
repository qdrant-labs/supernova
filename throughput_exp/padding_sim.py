#!/usr/bin/env python3
# Monte Carlo padding simulation for embedding throughput estimation.
#
# Samples a HuggingFace dataset, fits the token length distribution, then
# simulates batch padding waste across truncation cutoffs and batch sizes.
#
# Usage:
#   python padding_sim.py --dataset HuggingFaceFW/finewiki --hf-config en
#   python padding_sim.py --dataset HuggingFaceTB/dclm-edu --sample 100000
#   python padding_sim.py --dataset laion/aesthetics_v2_4.75 --hf-config default --column TEXT
#
# Output:
#   - padding_sim_results.json    — raw simulation data
#   - padding_dist.png            — token length distribution + fitted curve
#   - padding_heatmap.png         — padding efficiency heatmap (cutoff x batch_size)
#   - padding_tradeoff.png        — efficiency vs cutoff per batch size + semantic coverage

import argparse
import json
import time

import numpy as np
from scipy import stats as sp_stats
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# -- sample & tokenize --

def sample_token_lengths(dataset, hf_config, split, column, tokenizer_name, n):
    print(f"Sampling {n:,} rows from {dataset} (config={hf_config})...")
    ds = load_dataset(dataset, hf_config, split=split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    lengths = []
    for row in tqdm(ds.shuffle(seed=42).take(n), total=n, desc="Tokenizing"):
        text = row.get(column)
        if text and text.strip():
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))

    arr = np.array(lengths, dtype=np.int64)
    print(f"  {len(arr):,} non-empty texts tokenized")
    return arr


# -- fit parametric distributions --

def fit_distributions(lengths):
    fits = {}

    # lognormal
    shape, loc, scale = sp_stats.lognorm.fit(lengths, floc=0)
    ks = sp_stats.kstest(lengths, "lognorm", args=(shape, loc, scale))
    fits["lognormal"] = {
        "params": {"s": shape, "loc": loc, "scale": scale},
        "ks_stat": ks.statistic,
        "ks_pvalue": ks.pvalue,
    }

    # gamma
    a, loc, scale = sp_stats.gamma.fit(lengths, floc=0)
    ks = sp_stats.kstest(lengths, "gamma", args=(a, loc, scale))
    fits["gamma"] = {
        "params": {"a": a, "loc": loc, "scale": scale},
        "ks_stat": ks.statistic,
        "ks_pvalue": ks.pvalue,
    }

    best = min(fits, key=lambda k: fits[k]["ks_stat"])
    fits["best"] = best
    print(f"  Best fit: {best} (KS stat={fits[best]['ks_stat']:.4f}, p={fits[best]['ks_pvalue']:.4g})")
    return fits


# -- monte carlo batch padding simulation --

def simulate_padding(lengths, cutoffs, batch_sizes, num_batches=10_000, sorted_batching=False):
    # padding_efficiency = mean(sum(batch) / (max(batch) * batch_size))
    # 1.0 = no padding, 0.0 = all padding
    results = []

    for cutoff in cutoffs:
        truncated = np.minimum(lengths, cutoff)
        tokens_retained = truncated.sum() / lengths.sum()

        for bs in batch_sizes:
            if sorted_batching:
                sorted_lens = np.sort(truncated)
                n_full_batches = len(sorted_lens) // bs
                if n_full_batches == 0:
                    continue
                batches = sorted_lens[: n_full_batches * bs].reshape(n_full_batches, bs)
            else:
                indices = np.random.default_rng(42).integers(0, len(truncated), size=(num_batches, bs))
                batches = truncated[indices]

            batch_maxes = batches.max(axis=1)
            batch_sums = batches.sum(axis=1)
            batch_padded = batch_maxes * bs

            efficiencies = batch_sums / batch_padded

            results.append({
                "cutoff": cutoff,
                "batch_size": bs,
                "sorted": sorted_batching,
                "padding_efficiency_mean": float(efficiencies.mean()),
                "padding_efficiency_median": float(np.median(efficiencies)),
                "padding_efficiency_p5": float(np.percentile(efficiencies, 5)),
                "padding_waste_mean": float(1 - efficiencies.mean()),
                "tokens_retained_pct": float(tokens_retained * 100),
                "pct_texts_truncated": float((lengths > cutoff).mean() * 100),
                "num_batches": len(batches),
            })

    return results


# -- plots --

def plot_distribution(lengths, fits, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    # empirical histogram
    bins = np.logspace(np.log10(max(1, lengths.min())), np.log10(lengths.max()), 100)
    ax.hist(lengths, bins=bins, density=True, alpha=0.5, color="steelblue", label="Empirical")

    # fitted curve
    x = np.logspace(np.log10(max(1, lengths.min())), np.log10(lengths.max()), 500)
    best = fits["best"]
    params = fits[best]["params"]
    if best == "lognormal":
        pdf = sp_stats.lognorm.pdf(x, params["s"], params["loc"], params["scale"])
    else:
        pdf = sp_stats.gamma.pdf(x, params["a"], params["loc"], params["scale"])
    ax.plot(x, pdf, "r-", lw=2, label=f"{best} fit (KS={fits[best]['ks_stat']:.3f})")

    # percentile markers
    for p, color in [(50, "green"), (95, "orange"), (99, "red")]:
        val = np.percentile(lengths, p)
        ax.axvline(val, color=color, ls="--", alpha=0.7, label=f"p{p}: {val:,.0f}")

    ax.set_xscale("log")
    ax.set_xlabel("Token count per text")
    ax.set_ylabel("Density")
    ax.set_title("Token Length Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved {output}")


def plot_heatmap(results, output):
    import matplotlib.pyplot as plt

    # random batching only
    rand_results = [r for r in results if not r["sorted"]]
    if not rand_results:
        return

    cutoffs = sorted(set(r["cutoff"] for r in rand_results))
    batch_sizes = sorted(set(r["batch_size"] for r in rand_results))

    grid = np.zeros((len(cutoffs), len(batch_sizes)))
    for r in rand_results:
        i = cutoffs.index(r["cutoff"])
        j = batch_sizes.index(r["batch_size"])
        grid[i, j] = r["padding_efficiency_mean"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")

    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(cutoffs)))
    ax.set_yticklabels([f"{c:,}" for c in cutoffs])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Truncation cutoff (tokens)")
    ax.set_title("Padding Efficiency (higher = less waste)")

    for i in range(len(cutoffs)):
        for j in range(len(batch_sizes)):
            val = grid[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, label="Efficiency (real tokens / padded tokens)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved {output}")


def plot_tradeoff(results, output):
    import matplotlib.pyplot as plt

    rand_results = [r for r in results if not r["sorted"]]
    sorted_results = [r for r in results if r["sorted"]]

    batch_sizes = sorted(set(r["batch_size"] for r in rand_results))
    cutoffs = sorted(set(r["cutoff"] for r in rand_results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left: padding efficiency vs cutoff, one line per batch size
    for bs in batch_sizes:
        subset = sorted([r for r in rand_results if r["batch_size"] == bs], key=lambda r: r["cutoff"])
        ax1.plot(
            [r["cutoff"] for r in subset],
            [r["padding_efficiency_mean"] for r in subset],
            "o-", label=f"batch={bs}", markersize=4,
        )

    # sorted batching comparison at median batch size
    if sorted_results:
        mid_bs = batch_sizes[len(batch_sizes) // 2]
        sorted_sub = sorted([r for r in sorted_results if r["batch_size"] == mid_bs], key=lambda r: r["cutoff"])
        if sorted_sub:
            ax1.plot(
                [r["cutoff"] for r in sorted_sub],
                [r["padding_efficiency_mean"] for r in sorted_sub],
                "s--", color="black", label=f"sorted batch={mid_bs}", markersize=5,
            )

    ax1.set_xlabel("Truncation cutoff (tokens)")
    ax1.set_ylabel("Padding efficiency")
    ax1.set_title("Padding Efficiency vs Cutoff")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    # right: semantic coverage vs cutoff
    seen_cutoffs = set()
    cov_cutoffs, cov_values, cov_pct_trunc = [], [], []
    for r in rand_results:
        if r["cutoff"] not in seen_cutoffs:
            seen_cutoffs.add(r["cutoff"])
            cov_cutoffs.append(r["cutoff"])
            cov_values.append(r["tokens_retained_pct"])
            cov_pct_trunc.append(r["pct_texts_truncated"])
    order = np.argsort(cov_cutoffs)
    cov_cutoffs = [cov_cutoffs[i] for i in order]
    cov_values = [cov_values[i] for i in order]
    cov_pct_trunc = [cov_pct_trunc[i] for i in order]

    ax2.plot(cov_cutoffs, cov_values, "o-", color="tab:blue", label="Tokens retained %")
    ax2.set_xlabel("Truncation cutoff (tokens)")
    ax2.set_ylabel("Tokens retained (%)", color="tab:blue")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(cov_cutoffs, cov_pct_trunc, "s--", color="tab:red", label="Texts truncated %")
    ax2b.set_ylabel("Texts truncated (%)", color="tab:red")
    ax2b.set_ylim(0, 105)

    ax2.set_title("Semantic Coverage vs Cutoff")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved {output}")


# -- main --

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo padding simulation")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--hf-config", default=None, help="Dataset config (use 'None' to skip)")
    parser.add_argument("--column", default="text")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--tokenizer", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--batch-sizes", default="32,64,128,256,512")
    parser.add_argument("--cutoffs", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--num-batches", type=int, default=10_000)
    parser.add_argument("--output-dir", default="padding_sim_results")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    hf_config = None if args.hf_config == "None" else args.hf_config
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    cutoffs = [int(x) for x in args.cutoffs.split(",")]

    # sample & tokenize
    lengths = sample_token_lengths(
        args.dataset, hf_config, args.split, args.column, args.tokenizer, args.sample,
    )

    print(f"\n  Mean:   {lengths.mean():,.0f} tokens")
    print(f"  Median: {np.median(lengths):,.0f} tokens")
    print(f"  Stdev:  {lengths.std():,.0f} tokens")
    print(f"  p95:    {np.percentile(lengths, 95):,.0f} tokens")
    print(f"  p99:    {np.percentile(lengths, 99):,.0f} tokens")
    print(f"  Max:    {lengths.max():,} tokens")

    # fit distributions
    print("\nFitting distributions...")
    fits = fit_distributions(lengths)

    # monte carlo
    print(f"\nRunning padding simulation ({len(cutoffs)} cutoffs x {len(batch_sizes)} batch sizes)...")
    t0 = time.perf_counter()
    results_random = simulate_padding(lengths, cutoffs, batch_sizes, args.num_batches, sorted_batching=False)
    results_sorted = simulate_padding(lengths, cutoffs, batch_sizes, args.num_batches, sorted_batching=True)
    all_results = results_random + results_sorted
    print(f"  Simulated {len(all_results)} configs in {time.perf_counter() - t0:.1f}s")

    # plots
    print("\nGenerating plots...")
    plot_distribution(lengths, fits, os.path.join(args.output_dir, "padding_dist.png"))
    plot_heatmap(all_results, os.path.join(args.output_dir, "padding_heatmap.png"))
    plot_tradeoff(all_results, os.path.join(args.output_dir, "padding_tradeoff.png"))

    # summary table
    print(f"\n{'='*90}")
    print(f"{'Cutoff':>8} {'Batch':>6} {'Efficiency':>11} {'Sorted Eff':>11} {'Waste':>8} {'Retained':>10} {'Truncated':>10}")
    print(f"{'='*90}")
    for r_rand in results_random:
        r_sort = next((r for r in results_sorted if r["cutoff"] == r_rand["cutoff"] and r["batch_size"] == r_rand["batch_size"]), None)
        sorted_eff = f"{r_sort['padding_efficiency_mean']:.1%}" if r_sort else "—"
        print(f"{r_rand['cutoff']:>8,} {r_rand['batch_size']:>6} "
              f"{r_rand['padding_efficiency_mean']:>10.1%} {sorted_eff:>11} "
              f"{r_rand['padding_waste_mean']:>7.1%} "
              f"{r_rand['tokens_retained_pct']:>9.1f}% "
              f"{r_rand['pct_texts_truncated']:>9.1f}%")

    # save json
    output = {
        "dataset": args.dataset,
        "hf_config": hf_config,
        "column": args.column,
        "tokenizer": args.tokenizer,
        "sample_size": len(lengths),
        "stats": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "stdev": float(lengths.std()),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        },
        "fits": {k: v for k, v in fits.items() if k != "best"},
        "best_fit": fits["best"],
        "simulations": all_results,
    }
    out_path = os.path.join(args.output_dir, "padding_sim_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()