#!/usr/bin/env python3
"""
Embedding throughput benchmark.

Samples rows from a HuggingFace dataset, then sweeps across batch sizes
and dtypes to measure tokens/sec on the current GPU.

Usage:
  # Default: gte-multilingual-base on finewiki english
  python3.11 throughput_bench.py

  # Custom dataset
  python3.11 throughput_bench.py --dataset nick007x/arxiv-papers --column abstract --hf-config None

  # Sweep specific batch sizes
  python3.11 throughput_bench.py --batch-sizes 64,128,256,512

  # Test flash attention
  python3.11 throughput_bench.py --flash-attn

  # Multiple dtypes
  python3.11 throughput_bench.py --dtypes float16,bfloat16

  # Cap text length before splitting
  python3.11 throughput_bench.py --max-text-length 50000

  python throughput_bench.py --dataset HuggingFaceTB/dclm-edu --cutoffs 256,512,1024,2048,4096,8192 --batch-sizes 64 --sample 10000 --output dclm_cutoff_results.json --hf-config None
  python throughput_bench.py --dataset CohereLabs/wikipedia-2023-11-embed-multilingual-v3 --cutoffs 64,128,256 --batch-sizes 64 --sample 10000 --output results.json --hf-config en

"""

import argparse
import json
import time

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm


def sample_texts(dataset: str, hf_config: str | None, split: str, column: str, n: int) -> list[str]:
    """
    Stream and randomly sample n texts from a HuggingFace dataset.
    """
    print(f"Sampling {n:,} rows from {dataset} (config={hf_config}, column={column})...")
    ds = load_dataset(dataset, hf_config, split=split, streaming=True)

    texts = []
    for row in tqdm(ds.shuffle(seed=42).take(n), total=n, desc="Sampling"):
        text = row.get(column)
        if text and text.strip():
            texts.append(text)
    print(f"  Got {len(texts):,} non-empty texts")
    return texts


def count_tokens(texts: list[str], tokenizer) -> list[int]:
    """
    Count tokens per text.
    """
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in tqdm(texts, desc="Tokenizing")]


def run_trial(
    model: SentenceTransformer,
    texts: list[str],
    token_counts: list[int],
    batch_size: int,
    warmup_batches: int = 3,
) -> dict:
    """
    Run a single embedding trial, return metrics.
    """
    total_tokens = sum(token_counts)
    total_texts = len(texts)

    # warmup
    warmup_texts = texts[: batch_size * warmup_batches]
    model.encode(warmup_texts, batch_size=batch_size, show_progress_bar=False)
    torch.cuda.synchronize()

    # timed run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "texts": total_texts,
        "tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "texts_per_s": round(total_texts / elapsed, 1),
        "tokens_per_s": round(total_tokens / elapsed, 1),
        "gpu_mem_peak_gb": round(gpu_mem_gb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Embedding throughput benchmark")
    parser.add_argument("--dataset", default="HuggingFaceFW/finewiki")
    parser.add_argument("--hf-config", default="en", help="HuggingFace dataset config (use 'None' to skip)")
    parser.add_argument("--column", default="text")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--model", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--batch-sizes", default="32,64,128,256,512", help="Comma-separated batch sizes to test")
    parser.add_argument("--dtypes", default="bfloat16", help="Comma-separated dtypes to test (float32,float16,bfloat16)")
    parser.add_argument("--flash-attn", action="store_true", help="Enable flash attention 2")
    parser.add_argument("--max-text-length", type=int, default=None, help="Truncate texts to this many chars before embedding")
    parser.add_argument("--cutoffs", default=None, help="Comma-separated token cutoffs to sweep (e.g. 256,512,1024,2048,4096)")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    hf_config = None if args.hf_config == "None" else args.hf_config
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    dtypes_to_test = args.dtypes.split(",")
    cutoffs_to_test = [int(x) for x in args.cutoffs.split(",")] if args.cutoffs else [None]
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    # sample dataset
    texts = sample_texts(args.dataset, hf_config, args.split, args.column, args.sample)

    if args.max_text_length:
        before = sum(len(t) for t in texts)
        texts = [t[: args.max_text_length] for t in texts]
        after = sum(len(t) for t in texts)
        print(f"Truncated texts: {before:,} -> {after:,} total chars ({100*after/before:.1f}% kept)")

    # run experiments
    all_results = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"\nGPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Flash attention: {args.flash_attn}\n")

    for dtype_name in dtypes_to_test:
        torch_dtype = dtype_map[dtype_name]

        print(f"\n{'='*60}")
        print(f"Loading model: {args.model} (dtype={dtype_name}, flash_attn={args.flash_attn})")
        print(f"{'='*60}")

        model_kwargs = {"dtype": torch_dtype}
        if args.flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = SentenceTransformer(
            args.model,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
            model_kwargs=model_kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        max_tokens = model.max_seq_length

        # tokenize all texts once upfront
        print("Tokenizing all texts...")
        t_tok = time.perf_counter()
        all_token_ids = [tokenizer.encode(t, add_special_tokens=False) for t in tqdm(texts, desc="Tokenizing")]
        tokenize_time = time.perf_counter() - t_tok

        for cutoff in cutoffs_to_test:
            # 1. Clamp the requested cutoff to the model's physical limit
            actual_cutoff = min(cutoff, max_tokens) if cutoff else max_tokens
            
            # 2. Label it clearly so you know when a clamp occurred
            if cutoff and cutoff > max_tokens:
                cutoff_label = f"cutoff={cutoff} (clamped to {max_tokens})"
            else:
                cutoff_label = f"cutoff={cutoff}" if cutoff else "no cutoff"

            # 3. Use actual_cutoff for text generation AND token counting
            if cutoff:
                trial_texts = [
                    tokenizer.decode(toks[:actual_cutoff], skip_special_tokens=True) if len(toks) > actual_cutoff else texts[i]
                    for i, toks in enumerate(all_token_ids)
                ]
                token_counts = [min(len(toks), actual_cutoff) for toks in all_token_ids]
            else:
                trial_texts = texts
                token_counts = [min(len(toks), max_tokens) for toks in all_token_ids]

            total_tokens = sum(token_counts)
            mean_tokens = total_tokens / len(token_counts)
            print(f"\n  [{cutoff_label}] {len(trial_texts):,} texts, {total_tokens:,} tokens (mean {mean_tokens:.0f})")

            for bs in batch_sizes:
                print(f"\n--- batch_size={bs}, dtype={dtype_name}, {cutoff_label} ---")
                result = run_trial(model, trial_texts, token_counts, batch_size=bs)
                result.update({
                    "model": args.model,
                    "dtype": dtype_name,
                    "batch_size": bs,
                    "cutoff": cutoff,
                    "flash_attn": args.flash_attn,
                    "gpu": gpu_name,
                    "dataset": args.dataset,
                    "source_texts": len(trial_texts),
                    "mean_tokens": round(mean_tokens, 1),
                    "max_text_length": args.max_text_length,
                    "tokenize_s": round(tokenize_time, 2),
                    "tokenize_tok_per_s": round(sum(len(t) for t in all_token_ids) / tokenize_time, 0),
                })
                all_results.append(result)

                print(f"  {result['tokens_per_s']:,.0f} tok/s | {result['texts_per_s']:,.0f} texts/s | "
                      f"{result['gpu_mem_peak_gb']:.1f} GB peak | {result['elapsed_s']:.1f}s")

        # free GPU memory before next dtype
        del model
        torch.cuda.empty_cache()

    # summary table
    has_cutoffs = any(r["cutoff"] is not None for r in all_results)
    if has_cutoffs:
        print(f"\n\n{'='*90}")
        print(f"{'Model':<40} {'dtype':<10} {'cutoff':>7} {'batch':>6} {'tok/s':>10} {'texts/s':>10} {'VRAM GB':>8}")
        print(f"{'='*90}")
        for r in all_results:
            co = f"{r['cutoff']:,}" if r['cutoff'] else "none"
            print(f"{r['model']:<40} {r['dtype']:<10} {co:>7} {r['batch_size']:>6} "
                  f"{r['tokens_per_s']:>10,.0f} {r['texts_per_s']:>10,.0f} {r['gpu_mem_peak_gb']:>8.1f}")
    else:
        print(f"\n\n{'='*80}")
        print(f"{'Model':<40} {'dtype':<10} {'batch':>6} {'tok/s':>10} {'texts/s':>10} {'VRAM GB':>8}")
        print(f"{'='*80}")
        for r in all_results:
            print(f"{r['model']:<40} {r['dtype']:<10} {r['batch_size']:>6} "
                  f"{r['tokens_per_s']:>10,.0f} {r['texts_per_s']:>10,.0f} {r['gpu_mem_peak_gb']:>8.1f}")

    # save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
