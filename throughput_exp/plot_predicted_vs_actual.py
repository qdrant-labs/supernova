import sys
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "predicted_vs_actual.csv"

    try:
        # Load and cleanly cast string numbers to floats
        df = pl.read_csv(csv_path).with_columns(
            pl.col("predicted").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64),
            pl.col("actual").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64),
        )
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    # Visual setup
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
        }
    )

    unique_models = df["model"].unique().to_list()
    palette = sns.color_palette("deep", n_colors=len(unique_models))

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)

    # Seaborn requires pandas, so we convert just for the plot layer
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
    lo = min(df["predicted"].min(), df["actual"].min()) * 0.8
    hi = max(df["predicted"].max(), df["actual"].max()) * 1.2
    ax.plot(
        [lo, hi],
        [lo, hi],
        ls="--",
        lw=1.2,
        color="0.3",
        zorder=0,
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

    # Add evaluation metrics as an anchored text box
    metrics_text = ""
    for model in unique_models:
        # Filter natively in Polars and convert to numpy for sklearn
        model_df = df.filter(pl.col("model") == model)
        act = model_df["actual"].to_numpy()
        pred = model_df["predicted"].to_numpy()

        # Calculate metrics
        mape = mean_absolute_percentage_error(act, pred) * 100
        rmse = np.sqrt(mean_squared_error(act, pred))

        # R2 can be unstable with very few points, handle gracefully
        try:
            r2 = r2_score(act, pred)
            r2_str = f"{r2:.2f}"
        except ValueError:
            r2_str = "N/A"

        short_name = model.split("/")[-1][:15]  # Truncate long model names
        metrics_text += f"{short_name}:\n  MAPE: {mape:.1f}%\n  RMSE: {rmse:.0f}\n  R²: {r2_str}\n\n"

    # Clean up and place the text box
    # metrics_text = metrics_text.strip()
    # props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    # ax.text(0.96, 0.04, metrics_text, transform=ax.transAxes, fontsize=8,
    #         verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, frameon=True, fontsize=8, loc="upper left", title="Models"
    )

    fig.tight_layout()

    out = csv_path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
