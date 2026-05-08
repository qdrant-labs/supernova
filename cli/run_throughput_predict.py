#!/usr/bin/env python3
"""
Predict embedding throughput and cost for a vectorforge embedder config.

Reads the dataset, model, text rendering, and per-text cutoff straight from
the embedder YAML; only the GPU + parallelism + simulation knobs come from
the CLI. Anything pulled from the YAML can be overridden via flags.
"""

import logging
import os

import click
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


@click.command(
    name="throughput-predict", help="Predict embedding throughput + cost from a config."
)
@click.argument("config")
# GPU + cost knobs (not in the YAML)
@click.option(
    "--gpu",
    default="a10g",
    show_default=True,
    help=f"GPU key. One of: {', '.join(GPU_TABLE.keys())}",
)
@click.option(
    "--gpu-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Multiplier on effective TFLOPS (e.g. 0.85 for thermal headroom).",
)
@click.option(
    "--rate", type=float, default=None, help="$/hr override (default: GPU_TABLE entry)."
)
@click.option(
    "--num-gpus",
    type=int,
    default=None,
    help="Parallel GPUs for wall-clock estimate (default: single).",
)
@click.option(
    "--overhead",
    type=float,
    default=1.2,
    show_default=True,
    help="Cost overhead multiplier.",
)
# Overrides for config-derived values
@click.option(
    "--cutoff",
    type=int,
    multiple=True,
    default=(),
    help="Override dense_embedder.max_tokens. Repeat for sweep: --cutoff 256 --cutoff 512.",
)
@click.option(
    "--model",
    "model_override",
    default=None,
    help="Override the model used for tokenization + param count.",
)
@click.option(
    "--hf-config",
    "hf_config_override",
    default=None,
    help="Override the HuggingFace dataset config (e.g. 20231101.en).",
)
@click.option("--split", "split_override", default=None, help="Override source.split.")
@click.option(
    "--column", "column_override", default=None, help="Override source.text_field."
)
@click.option(
    "--template",
    "template_override",
    default=None,
    help="Override source.text_template.",
)
@click.option(
    "--total-rows",
    type=int,
    default=None,
    help="Override source.total_rows_override / HF metadata.",
)
@click.option(
    "--params",
    type=int,
    default=None,
    help="Override the model parameter count (skips the AutoConfig lookup).",
)
# Simulation knobs
@click.option(
    "--sample",
    type=int,
    default=100_000,
    show_default=True,
    help="Number of rows to tokenize for the empirical distribution.",
)
@click.option(
    "--batch-size",
    "batch_size_override",
    type=int,
    default=None,
    help="Override dense_embedder.batch_size for padding simulation.",
)
@click.option(
    "--num-batches",
    type=int,
    default=10_000,
    show_default=True,
    help="Number of synthetic batches to draw in the simulation.",
)
@click.option("--output", default=None, help="Write JSON results + plot to this path.")
@click.option("-v", "--verbose", is_flag=True)
def throughput_predict(
    config,
    gpu,
    gpu_scale,
    rate,
    num_gpus,
    overhead,
    cutoff,
    model_override,
    hf_config_override,
    split_override,
    column_override,
    template_override,
    total_rows,
    params,
    sample,
    batch_size_override,
    num_batches,
    output,
    verbose,
):
    """Predict embedding throughput and cost from a vectorforge embedder config."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )

    if not os.path.isfile(config):
        raise click.UsageError(f"Config not found: {config}")

    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    try:
        derived = _derive_from_config(cfg)
    except ValueError as e:
        raise click.UsageError(str(e))

    # CLI flags win over YAML-derived values
    cutoffs = list(cutoff) or ([derived["cutoff"]] if derived["cutoff"] else None)
    if not cutoffs:
        raise click.UsageError(
            "No cutoff: the config has no dense/multivector/sparse embedder with "
            "max_tokens, and --cutoff was not given. Pass --cutoff N (or multiple Ns)."
        )

    model = model_override or derived["model"]
    if not model:
        raise click.UsageError(
            "No model: the config has no dense/multivector/sparse embedder with a "
            "model field, and --model was not given."
        )

    column = column_override if column_override is not None else derived["column"]
    template = (
        template_override if template_override is not None else derived["template"]
    )
    if column_override or template_override:
        # Explicit CLI override resets the other side to avoid ambiguity.
        if column_override:
            template = None
        if template_override:
            column = None

    if not column and not template:
        raise click.UsageError(
            "No text rendering: the config has no source.text_field or "
            "source.text_template, and neither --column nor --template was given."
        )

    try:
        run_prediction(
            dataset=derived["dataset"],
            model=model,
            cutoffs=cutoffs,
            hf_config=hf_config_override
            if hf_config_override is not None
            else derived["hf_config"],
            split=split_override or derived["split"],
            column=column,
            template=template,
            sample=sample,
            gpu=gpu,
            gpu_scale=gpu_scale,
            rate=rate,
            batch_size=batch_size_override or derived["batch_size"] or 64,
            num_batches=num_batches,
            total_rows=total_rows if total_rows is not None else derived["total_rows"],
            total_rows_source="config"
            if (total_rows is None and derived["total_rows"])
            else None,
            num_gpus=num_gpus,
            overhead=overhead,
            params=params,
            output=output,
        )
    except ValueError as e:
        raise click.UsageError(str(e))


if __name__ == "__main__":
    throughput_predict()
