#!/usr/bin/env python3
"""Predicted-vs-actual throughput scatter for `nova embed predict`.

Regenerates the validation figure in docs/embedding/throughput-prediction.md
from the experiment data next to this script:

    uv run --with pandas,seaborn,matplotlib \
        scripts/plot_predicted_vs_actual.py

Writes vector SVG + PDF (300 dpi for any rasterized elements) into docs/fig/.
Styling matches the original figure from the throughput experiments
(pre-restructure throughput_exp/plot_predicted_vs_actual.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "throughput_experiments.csv"
DEFAULT_OUT_DIR = HERE.parent / "docs" / "fig"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--formats", default="svg,pdf", help="comma-separated: svg,pdf,png"
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pd
    import seaborn as sns

    # predicted/actual arrive as quoted thousands-separated strings ("1,279")
    df = pd.read_csv(args.csv, thousands=",")

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial"],
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "svg.fonttype": "none",  # keep text as text in the SVG, not paths
        }
    )

    # hue order = CSV appearance order, so colors stay stable across regenerations
    models = list(dict.fromkeys(df["model"]))
    palette = sns.color_palette("deep", n_colors=len(models))

    # slightly wider than tall (matches the original figure's aspect) so the
    # upper-left legend clears the gte/e5 points around y=1,200
    fig, ax = plt.subplots(figsize=(6.1, 5.5), dpi=300)

    sns.scatterplot(
        data=df,
        x="predicted",
        y="actual",
        hue="model",
        hue_order=models,
        palette=palette,
        s=50,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        ax=ax,
    )

    lo = min(df["predicted"].min(), df["actual"].min()) * 0.8
    hi = max(df["predicted"].max(), df["actual"].max()) * 1.2
    ax.plot(
        [lo, hi], [lo, hi], ls="--", lw=1.2, color="0.3", zorder=0,
        label="y = x (Perfect)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted throughput (texts/s)", fontweight="bold")
    ax.set_ylabel("Actual throughput (texts/s)", fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, fontsize=8, loc="upper left", title="Models")

    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        out = args.out_dir / f"predicted_vs_actual.{fmt.strip()}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
