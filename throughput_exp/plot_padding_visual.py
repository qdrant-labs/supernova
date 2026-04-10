#!/usr/bin/env python3
# visualize padding waste as batch grids across cutoffs
#
# draws one panel per cutoff, each showing a batch grid:
#   each row = one sequence, each column = one token position
#   colored = real tokens, grey = padding
#
# usage:
#   python plot_padding_visual.py --from-json padding_sim_results/padding_sim_results.json
#   python plot_padding_visual.py --from-json padding_sim_results/padding_sim_results.json --cutoffs 256,512,1024,2048,8192
#   python plot_padding_visual.py --dataset HuggingFaceTB/dclm-edu --hf-config None --cutoffs 256,512,1024,4096

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba


def draw_batch_grid(ax, lengths, max_len, title, batch_size, display_cols=64):
    scale = max_len / display_cols if max_len > display_cols else 1

    purple = to_rgba("#8b5cf6")  # tailwind violet-500
    grey = to_rgba("#d0d0d0")

    for row in range(batch_size):
        real_cols = int(lengths[row] / scale) if scale > 0 else 0
        for col in range(display_cols):
            color = purple if col < real_cols else grey
            rect = mpatches.FancyBboxPatch(
                (col, batch_size - 1 - row), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=0.3,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, display_cols + 0.5)
    ax.set_ylim(-1.5, batch_size + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    # efficiency label
    total_real = sum(lengths)
    total_padded = max_len * batch_size
    eff = total_real / total_padded if total_padded > 0 else 0
    waste = 1 - eff
    ax.text(
        display_cols / 2, -1.2,
        f"{eff:.0%} efficient — {waste:.0%} wasted on padding",
        ha="center", fontsize=9, color="#555",
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize padding waste across cutoffs")
    parser.add_argument("--dataset", default="HuggingFaceTB/dclm-edu")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--column", default="text")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--cutoffs", default="256,512,1024,2048,4096,8192")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-json", default=None)
    parser.add_argument("--output", default="padding_visual.png")
    args = parser.parse_args()

    hf_config = None if args.hf_config == "None" else args.hf_config
    rng = np.random.default_rng(args.seed)
    bs = args.batch_size
    cutoffs = [int(x) for x in args.cutoffs.split(",")]

    # get token lengths
    if args.from_json:
        with open(args.from_json) as f:
            data = json.load(f)
        fit_name = data["best_fit"]
        params = data["fits"][fit_name]["params"]
        from scipy import stats as sp_stats
        if fit_name == "lognormal":
            all_lengths = sp_stats.lognorm.rvs(params["s"], params["loc"], params["scale"], size=10000, random_state=rng).astype(int)
        else:
            all_lengths = sp_stats.gamma.rvs(params["a"], params["loc"], params["scale"], size=10000, random_state=rng).astype(int)
        all_lengths = np.clip(all_lengths, 1, None)
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from tqdm import tqdm

        print(f"Sampling 1000 rows from {args.dataset}...")
        ds = load_dataset(args.dataset, hf_config, split=args.split, streaming=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        all_lengths = []
        for row in tqdm(ds.shuffle(seed=args.seed).take(1000), total=1000):
            text = row.get(args.column)
            if text and text.strip():
                all_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        all_lengths = np.array(all_lengths, dtype=np.int64)

    # pick one batch — same indices across all cutoffs for fair comparison
    batch_indices = rng.choice(len(all_lengths), size=bs, replace=False)
    raw_batch = np.minimum(all_lengths[batch_indices], args.max_seq_length)

    # shuffle for jagged visual
    rng.shuffle(raw_batch)

    # draw one panel per cutoff
    n_panels = len(cutoffs)
    panel_height = bs * 0.35
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, panel_height * n_panels + 2))
    if n_panels == 1:
        axes = [axes]

    for i, cutoff in enumerate(cutoffs):
        lengths = np.minimum(raw_batch, cutoff)
        pad_to = int(lengths.max())
        draw_batch_grid(axes[i], lengths, pad_to, f"Cutoff = {cutoff:,} tokens", bs)

    fig.suptitle("Padding Waste by Truncation Cutoff", fontsize=14, fontweight="bold")

    # legend at bottom
    real_patch = mpatches.Patch(facecolor="#8b5cf6", edgecolor="white", label="Real tokens")
    pad_patch = mpatches.Patch(facecolor="#d0d0d0", edgecolor="white", label="Padding (wasted compute)")
    fig.legend(handles=[real_patch, pad_patch], loc="lower center", ncol=2, fontsize=10, frameon=False)

    fig.subplots_adjust(hspace=0.15, top=0.95, bottom=0.03)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()