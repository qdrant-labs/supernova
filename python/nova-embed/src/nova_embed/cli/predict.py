"""`nova embed predict <config>` — throughput + cost prediction, no GPU needed.

Everything dataset/model-shaped comes from the embed config itself (including
which entries fuse into one forward pass); only GPU, cost, and simulation
knobs live on the CLI.
"""

import logging

import click

from nova_embed.predict import GPU_TABLE


@click.command(
    name="predict",
    help="Predict embedding throughput + cost from a config (no GPU needed).",
)
@click.argument("config")
# GPU + cost knobs (not in the YAML)
@click.option(
    "--gpu",
    default="a10g",
    show_default=True,
    help=f"GPU key. One of: {', '.join(GPU_TABLE)}",
)
@click.option(
    "--gpu-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Multiplier on effective TFLOPS (e.g. 0.85 for thermal headroom).",
)
@click.option(
    "--rate", type=float, default=None, help="$/hr override (default: GPU table entry)."
)
@click.option(
    "--num-gpus",
    type=int,
    default=None,
    help="Parallel GPUs for the wall-clock estimate (default: single).",
)
@click.option(
    "--overhead",
    type=float,
    default=1.2,
    show_default=True,
    help="Cost overhead multiplier.",
)
# Overrides for config/model-derived values
@click.option(
    "--cutoff",
    type=int,
    default=None,
    help="Token cutoff for the padding simulation "
    "(default: each model's positional limit).",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override the per-pass batch size from the config entries.",
)
@click.option(
    "--total-rows",
    type=int,
    default=None,
    help="Override the dataset row count (skips the source metadata sweep).",
)
@click.option(
    "--params",
    type=int,
    default=None,
    help="Override the model parameter count (single-model configs only).",
)
# Simulation knobs
@click.option(
    "--sample",
    type=int,
    default=100_000,
    show_default=True,
    help="Rows to tokenize for the empirical token distribution.",
)
@click.option(
    "--num-batches",
    type=int,
    default=10_000,
    show_default=True,
    help="Synthetic batches drawn in the padding simulation.",
)
@click.option("--plot", default=None, help="Write token-distribution plot(s) to this path.")
@click.option("--output", default=None, help="Write the full JSON results to this path.")
@click.option("-v", "--verbose", is_flag=True)
def predict(
    config,
    gpu,
    gpu_scale,
    rate,
    num_gpus,
    overhead,
    cutoff,
    batch_size,
    total_rows,
    params,
    sample,
    num_batches,
    plot,
    output,
    verbose,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(message)s"
    )

    from nova_embed.config import load_config
    from nova_embed.predict import run_prediction

    import nova_embed.sources  # noqa: F401 — registration side-effects

    cfg = load_config(config)
    try:
        run_prediction(
            cfg,
            config_label=config,
            gpu=gpu,
            gpu_scale=gpu_scale,
            rate=rate,
            num_gpus=num_gpus,
            overhead=overhead,
            cutoff=cutoff,
            sample=sample,
            batch_size=batch_size,
            num_batches=num_batches,
            total_rows=total_rows,
            params=params,
            plot=plot,
            output=output,
        )
    except ValueError as e:
        raise click.UsageError(str(e))
