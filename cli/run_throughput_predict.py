#!/usr/bin/env python3
"""
Predict embedding throughput and cost for a vectorforge embedder config.

Reads the dataset, model, text rendering, and per-text cutoff straight from
the embedder YAML; only the GPU + parallelism + simulation knobs come from
the CLI. Anything pulled from the YAML can be overridden via flags.

Usage:
  vf throughput-predict configs/embedder/wikipedia_openai.yaml --gpu h100
  vf throughput-predict configs/embedder/fineweb-cc-main-2025-26.yaml \\
      --gpu b200 --num-gpus 8
  vf throughput-predict <config> --cutoff 256 512 1024 --sample 50000
  vf throughput-predict <config> --gpu a100 --output predictions.json
"""

import argparse
import logging
import os

import yaml

from vectorforge.throughput import GPU_TABLE, run_prediction


def _derive_from_config(config: dict) -> dict:
    """Pull throughput-predict inputs out of an embedder YAML."""
    source = config.get("source") or {}
    dense = config.get("dense_embedder") or {}
    multivector = config.get("multivector_embedder") or {}
    sparse = config.get("sparse_embedder") or {}

    dataset = source.get("dataset_name")
    if not dataset:
        raise ValueError("config: source.dataset_name is required")

    # huggingface source has an HF dataset config; huggingface_parquet does not.
    hf_config = source.get("config")

    split = source.get("split", "train")
    column = source.get("text_field")
    template = source.get("text_template")
    if template:
        # text_template wins over text_field, matching _build_text_extractor.
        column = None

    # Pick the model used for tokenization + parameter count. Dense wins;
    # fall back to multivector then sparse for runs that omit dense.
    embedder_for_model = dense or multivector or sparse
    model = embedder_for_model.get("model") if embedder_for_model else None

    # Per-text truncation is the cutoff used in the padding simulation.
    cutoff = embedder_for_model.get("max_tokens") if embedder_for_model else None

    batch_size = embedder_for_model.get("batch_size") if embedder_for_model else None

    total_rows = source.get("total_rows_override")

    return {
        "dataset": dataset,
        "hf_config": hf_config,
        "split": split,
        "column": column,
        "template": template,
        "model": model,
        "cutoff": cutoff,
        "batch_size": batch_size,
        "total_rows": total_rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict embedding throughput and cost from a vectorforge embedder config",
    )
    parser.add_argument("config", help="Path to embedder YAML config")

    # GPU + cost knobs (not in the YAML)
    parser.add_argument("--gpu", default="a10g",
                        help=f"GPU key. One of: {', '.join(GPU_TABLE.keys())}")
    parser.add_argument("--gpu-scale", type=float, default=1.0,
                        help="Multiplier on effective TFLOPS (e.g. 0.85 for thermal headroom)")
    parser.add_argument("--rate", type=float, default=None,
                        help="$/hr override (default: GPU_TABLE entry)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Parallel GPUs for wall-clock estimate (default: single)")
    parser.add_argument("--overhead", type=float, default=1.2,
                        help="Cost overhead multiplier (default: 1.2)")

    # Overrides for config-derived values
    parser.add_argument("--cutoff", type=int, nargs="+", default=None,
                        help="Override dense_embedder.max_tokens. Accepts multiple to sweep.")
    parser.add_argument("--model", default=None,
                        help="Override the model used for tokenization + param count")
    parser.add_argument("--hf-config", default=None,
                        help="Override the HuggingFace dataset config (e.g. 20231101.en)")
    parser.add_argument("--split", default=None, help="Override source.split")
    parser.add_argument("--column", default=None, help="Override source.text_field")
    parser.add_argument("--template", default=None, help="Override source.text_template")
    parser.add_argument("--total-rows", type=int, default=None,
                        help="Override source.total_rows_override / HF metadata")
    parser.add_argument("--params", type=int, default=None,
                        help="Override the model parameter count (skips the AutoConfig lookup)")

    # Simulation knobs
    parser.add_argument("--sample", type=int, default=100_000,
                        help="Number of rows to tokenize for the empirical distribution")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override dense_embedder.batch_size for padding simulation")
    parser.add_argument("--num-batches", type=int, default=10_000,
                        help="Number of synthetic batches to draw in the simulation")

    parser.add_argument("--output", default=None,
                        help="Write JSON results + plot to this path")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not os.path.isfile(args.config):
        parser.error(f"Config not found: {args.config}")

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    try:
        derived = _derive_from_config(config)
    except ValueError as e:
        parser.error(str(e))

    # CLI flags win over YAML-derived values
    cutoffs = args.cutoff or ([derived["cutoff"]] if derived["cutoff"] else None)
    if not cutoffs:
        parser.error(
            "No cutoff: the config has no dense/multivector/sparse embedder with "
            "max_tokens, and --cutoff was not given. Pass --cutoff N (or multiple Ns)."
        )

    model = args.model or derived["model"]
    if not model:
        parser.error(
            "No model: the config has no dense/multivector/sparse embedder with a "
            "model field, and --model was not given."
        )

    column = args.column if args.column is not None else derived["column"]
    template = args.template if args.template is not None else derived["template"]
    if args.column or args.template:
        # Explicit CLI override resets the other side to avoid ambiguity.
        if args.column:
            template = None
        if args.template:
            column = None

    if not column and not template:
        parser.error(
            "No text rendering: the config has no source.text_field or "
            "source.text_template, and neither --column nor --template was given."
        )

    try:
        run_prediction(
            dataset=derived["dataset"],
            model=model,
            cutoffs=cutoffs,
            hf_config=args.hf_config if args.hf_config is not None else derived["hf_config"],
            split=args.split or derived["split"],
            column=column,
            template=template,
            sample=args.sample,
            gpu=args.gpu,
            gpu_scale=args.gpu_scale,
            rate=args.rate,
            batch_size=args.batch_size or derived["batch_size"] or 64,
            num_batches=args.num_batches,
            total_rows=args.total_rows if args.total_rows is not None else derived["total_rows"],
            total_rows_source="config" if (args.total_rows is None and derived["total_rows"]) else None,
            num_gpus=args.num_gpus,
            overhead=args.overhead,
            params=args.params,
            output=args.output,
        )
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
