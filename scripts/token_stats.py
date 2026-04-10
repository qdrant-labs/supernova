"""
Sample a HuggingFace dataset and report token length statistics for a text column.

Usage:
  python scripts/token_stats.py HuggingFaceFW/finewiki --config en --column text
  python scripts/token_stats.py nick007x/arxiv-papers --column abstract
  python scripts/token_stats.py mteb/tweet_sentiment_extraction --column text --tokenizer cl100k_base --sample 5000
"""

import argparse
import statistics

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Token length stats for a HuggingFace dataset column")
    parser.add_argument("dataset", help="HuggingFace dataset name")
    parser.add_argument("--config", default=None, help="Dataset config (e.g. 'en', '20231101.en')")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--column", default="text", help="Text column to measure (default: text)")
    parser.add_argument("--tokenizer", default="Alibaba-NLP/gte-multilingual-base", help="HF tokenizer or tiktoken encoding name (default: gte-multilingual-base)")
    parser.add_argument("--sample", type=int, default=10_000, help="Number of rows to sample (default: 10000)")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.dataset} (config={args.config}, split={args.split}, streaming=True)...")
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    # Try HF tokenizer first, fall back to tiktoken
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        import tiktoken
        enc = tiktoken.get_encoding(args.tokenizer)
        encode = lambda text: enc.encode(text)

    print(f"Sampling {args.sample:,} random rows, column='{args.column}'...")

    lengths = []
    skipped = 0
    for row in tqdm(ds.shuffle(seed=42).take(args.sample), total=args.sample, desc="Processing rows"):
        text = row.get(args.column)
        if not text or not text.strip():
            skipped += 1
            continue
        lengths.append(len(encode(text)))

    if not lengths:
        print("No non-empty rows found.")
        return

    lengths_sorted = sorted(lengths)
    p95 = lengths_sorted[int(len(lengths) * 0.95)]
    p99 = lengths_sorted[int(len(lengths) * 0.99)]

    print()
    print(f"{'Dataset:':<12} {args.dataset}")
    print(f"{'Column:':<12} {args.column}")
    print(f"{'Tokenizer:':<12} {args.tokenizer}")
    print(f"{'Sampled:':<12} {len(lengths):,} rows ({skipped} empty skipped)")
    print(f"{'Mean:':<12} {statistics.mean(lengths):,.0f} tokens")
    print(f"{'Median:':<12} {statistics.median(lengths):,.0f} tokens")
    print(f"{'Stdev:':<12} {statistics.stdev(lengths):,.0f} tokens")
    print(f"{'p95:':<12} {p95:,.0f} tokens")
    print(f"{'p99:':<12} {p99:,.0f} tokens")
    print(f"{'Max:':<12} {max(lengths):,.0f} tokens")


if __name__ == "__main__":
    main()