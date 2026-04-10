#!/usr/bin/env python3
# plot padding efficiency (from sim) vs measured throughput (from bench)
#
# usage:
#   python plot_correlation.py \
#     --sim padding_sim_results/padding_sim_results.json \
#     --bench dclm_cutoff_results.json \
#     --batch-size 64

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", required=True, help="padding_sim_results.json from padding_sim.py")
    parser.add_argument("--bench", required=True, help="results.json from bench.py with --cutoffs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size to filter on")
    parser.add_argument("--output", default="correlation.png")
    args = parser.parse_args()

    with open(args.sim) as f:
        sim_data = json.load(f)
    with open(args.bench) as f:
        bench_data = json.load(f)

    # build lookup: cutoff -> padding efficiency (random batching, matching batch size)
    sim_by_cutoff = {}
    for r in sim_data["simulations"]:
        if not r["sorted"] and r["batch_size"] == args.batch_size:
            sim_by_cutoff[r["cutoff"]] = r["padding_efficiency_mean"]

    # match bench results
    points = []
    for r in bench_data:
        cutoff = r.get("cutoff")
        if cutoff and cutoff in sim_by_cutoff:
            points.append({
                "cutoff": cutoff,
                "efficiency": sim_by_cutoff[cutoff],
                "tok_s": r["tokens_per_s"],
            })

    if not points:
        print("No matching cutoffs found between sim and bench results.")
        return

    points.sort(key=lambda p: p["efficiency"])
    effs = np.array([p["efficiency"] for p in points])
    toks = np.array([p["tok_s"] for p in points])

    # linear fit
    slope, intercept = np.polyfit(effs, toks, 1)
    r_squared = 1 - np.sum((toks - (slope * effs + intercept))**2) / np.sum((toks - toks.mean())**2)
    fit_x = np.linspace(0, 1, 100)
    fit_y = slope * fit_x + intercept

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(effs, toks, s=80, color="#8b5cf6", zorder=5)
    ax.plot(fit_x, fit_y, "--", color="#a0a0a0", lw=1.5, label=f"Linear fit (R²={r_squared:.3f})")

    # label each point with its cutoff
    for p in points:
        ax.annotate(
            f"{p['cutoff']:,}",
            (p["efficiency"], p["tok_s"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, color="#555",
        )

    ax.set_xlabel("Padding Efficiency (simulated)", fontsize=12)
    ax.set_ylabel("Throughput (tok/s, measured)", fontsize=12)
    ax.set_title("Padding Efficiency vs GPU Throughput", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(toks) * 1.15)
    ax.grid(alpha=0.3)

    # add prediction formula
    ax.text(
        0.05, max(toks) * 1.05,
        f"predicted_tok_s = {slope:,.0f} × efficiency + {intercept:,.0f}",
        fontsize=9, color="#555", family="monospace",
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Saved {args.output}")
    print(f"\nLinear fit: tok/s = {slope:,.0f} * efficiency + {intercept:,.0f}")
    print(f"R² = {r_squared:.4f}")
    print(f"\nPrediction table:")
    print(f"  {'Cutoff':>8} {'Simulated Eff':>14} {'Measured tok/s':>15} {'Predicted tok/s':>16}")
    for p in points:
        predicted = slope * p["efficiency"] + intercept
        print(f"  {p['cutoff']:>8,} {p['efficiency']:>13.1%} {p['tok_s']:>15,.0f} {predicted:>16,.0f}")


if __name__ == "__main__":
    main()
