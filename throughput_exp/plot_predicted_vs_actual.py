import sys

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "predicted_vs_actual.csv"

    df = (
        pl.read_csv(csv_path)
        .with_columns(
            pl.col("predicted").str.replace_all(",", "").cast(pl.Float64),
            pl.col("actual").str.replace_all(",", "").cast(pl.Float64),
        )
    )

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

    palette = sns.color_palette("deep", n_colors=df["model"].n_unique())

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)

    sns.scatterplot(
        data=df.to_pandas(),
        x="predicted",
        y="actual",
        hue="model",
        palette=palette,
        s=50,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        ax=ax,
    )

    # y = x reference line
    lo = min(df["predicted"].min(), df["actual"].min()) * 0.9
    hi = max(df["predicted"].max(), df["actual"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="0.4", zorder=0, label="y = x")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted throughput (text/s)")
    ax.set_ylabel("Actual throughput (text/s)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper left")

    sns.despine(trim=True)
    fig.tight_layout()

    out = csv_path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()