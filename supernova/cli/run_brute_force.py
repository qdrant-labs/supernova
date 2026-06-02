#!/usr/bin/env python3
"""
Brute-force nearest-neighbor search for recall evaluation.

Single-worker mode (--local): one GPU instance exhaustively searches the
full corpus. Use for small corpora or testing.

Distributed mode: use `nova brute-force-dist` to split the corpus across N
GPU workers. Each worker runs with --num-jobs N and outputs a partial result
file. Merge with `nova brute-force-merge` when all workers finish.
"""

import logging

from pathlib import Path

import click
import yaml

from supernova.cli.skypilot_utils import (
    CUDA_IMAGE_IDS,
    build_env_dict,
    launch_single_job,
    make_run_dir,
)
from supernova.destinations import S3Destination, parse_destination
from supernova.eval.brute_force import (
    DEFAULT_K,
    DistanceMetric,
    merge_results,
    run_brute_force,
)
from supernova.utils import get_bucket_region

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE_TYPE = "g4dn.2xlarge"  # 1× T4 GPU, 32GB RAM, 25Gbps
DEFAULT_ACCELERATOR = "T4:1"


def launch_on_ec2(
    corpus_uri: str,
    queries_filename: str,
    k: int,
    metric: DistanceMetric,
    dense_column: str,
    output: str,
    instance_type: str,
    on_demand: bool,
    dry_run: bool,
):
    dest = parse_destination(corpus_uri)
    if not isinstance(dest, S3Destination):
        raise click.UsageError(
            f"EC2 launch is supported for s3:// corpora only. For {corpus_uri}, use --local."
        )
    region = get_bucket_region(dest.bucket)
    click.echo(f"Bucket region: {region}")

    worker_flags = (
        f"--queries {queries_filename} -k {k} "
        f"--metric {metric.value} --dense-column {dense_column} "
        f"--output {output} --local"
    )

    run_dir = make_run_dir("brute-force")

    image_id = CUDA_IMAGE_IDS.get(region)
    if image_id is None:
        click.echo(
            f"Warning: no CUDA AMI configured for {region!r}. Known: {list(CUDA_IMAGE_IDS)}"
        )

    resources = {
        "cloud": "aws",
        "region": region,
        "instance_type": instance_type,
        "accelerators": DEFAULT_ACCELERATOR,
        "use_spot": not on_demand,
    }
    if image_id:
        resources["image_id"] = image_id

    job_yaml = {
        "name": "nova-brute-force",
        "resources": resources,
        "file_mounts": {"/app": "."},
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra eval",
        "run": f"cd /app && uv run nova brute-force {corpus_uri} {worker_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    click.echo("=" * 60)
    click.echo("supernova brute-force plan")
    click.echo("=" * 60)
    click.echo(f"  Corpus URI:  {corpus_uri}")
    click.echo(f"  Queries:     {queries_filename}")
    click.echo(f"  K:           {k}")
    click.echo(f"  Metric:      {metric.value}")
    click.echo(
        f"  Instance:    {instance_type}  ({'on-demand' if on_demand else 'spot'})"
    )
    click.echo(f"  Output:      {dest.eval_uri(output)}")
    click.echo(f"  Run dir:     {run_dir}")
    click.echo("=" * 60)

    if dry_run:
        click.echo(f"\n[dry run] Job config: {job_path}")
        click.echo(f"To run manually: sky jobs launch -y {job_path}")
        return

    launch_single_job(job_path, build_env_dict())
    click.echo(f"\nOutput will be at {dest.eval_uri(output)}")
    click.echo("Monitor: sky jobs logs")
    click.echo("Cancel:  sky jobs cancel -a")


_METRIC_CHOICES = [m.value for m in DistanceMetric]


@click.command(
    name="brute-force",
    help="Exhaustive nearest-neighbor search for recall eval (single GPU).",
)
@click.argument("corpus_uri")
@click.option(
    "--queries",
    default="queries_1000.parquet",
    show_default=True,
    help="Queries parquet filename within {corpus}/eval/.",
)
@click.option(
    "-k",
    "k",
    type=int,
    default=DEFAULT_K,
    show_default=True,
    help="Neighbors to retrieve per query.",
)
@click.option(
    "--metric",
    type=click.Choice(_METRIC_CHOICES, case_sensitive=False),
    default=DistanceMetric.COSINE.value,
    show_default=True,
    help="Distance metric.",
)
@click.option(
    "--dense-column",
    default="dense_embedding",
    show_default=True,
    help="Dense embedding column name.",
)
@click.option(
    "--output",
    default=None,
    help="Output filename (default: brute_force_<queries_stem>_k<K>.parquet).",
)
@click.option("--local", is_flag=True, help="Run in-process instead of launching EC2.")
@click.option(
    "--num-jobs",
    type=int,
    default=None,
    help="Total parallel workers (used by brute-force-dist).",
)
@click.option(
    "--job-rank",
    type=int,
    default=None,
    help="This worker's rank (0-indexed; defaults to $SKYPILOT_JOB_RANK).",
)
@click.option(
    "--instance-type",
    default=DEFAULT_INSTANCE_TYPE,
    show_default=True,
    help="EC2 instance type.",
)
@click.option("--on-demand", is_flag=True, help="Use on-demand instead of spot.")
@click.option(
    "--dry-run", is_flag=True, help="Print plan and write job config, don't launch."
)
def brute_force(
    corpus_uri,
    queries,
    k,
    metric,
    dense_column,
    output,
    local,
    num_jobs,
    job_rank,
    instance_type,
    on_demand,
    dry_run,
):
    """Brute-force nearest-neighbor search for recall evaluation."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger("supernova").setLevel(logging.INFO)

    try:
        parse_destination(corpus_uri)
    except ValueError as e:
        raise click.UsageError(str(e))

    metric_enum = DistanceMetric(metric)
    queries_stem = Path(queries).stem
    if output is None:
        output = f"brute_force_{queries_stem}_k{k}.parquet"

    if local or num_jobs:
        run_brute_force(
            corpus_uri=corpus_uri,
            queries_filename=queries,
            k=k,
            metric=metric_enum,
            dense_column=dense_column,
            output=output,
            num_jobs=num_jobs,
            job_rank=job_rank,
        )
    else:
        launch_on_ec2(
            corpus_uri=corpus_uri,
            queries_filename=queries,
            k=k,
            metric=metric_enum,
            dense_column=dense_column,
            output=output,
            instance_type=instance_type,
            on_demand=on_demand,
            dry_run=dry_run,
        )


@click.command(
    name="brute-force-merge",
    help="Merge partial results from a distributed brute-force run.",
)
@click.argument("corpus_uri")
@click.option(
    "--queries",
    default="queries_1000.parquet",
    show_default=True,
    help="Queries parquet filename within {corpus}/eval/.",
)
@click.option(
    "-k",
    "k",
    type=int,
    default=DEFAULT_K,
    show_default=True,
    help="Neighbors retrieved per query.",
)
@click.option(
    "--output",
    default=None,
    help="Output filename (default: brute_force_<queries_stem>_k<K>.parquet).",
)
def brute_force_merge(corpus_uri, queries, k, output):
    """Merge partial brute-force results from a distributed run."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger("supernova").setLevel(logging.INFO)

    try:
        parse_destination(corpus_uri)
    except ValueError as e:
        raise click.UsageError(str(e))

    queries_stem = Path(queries).stem
    if output is None:
        output = f"brute_force_{queries_stem}_k{k}.parquet"

    merge_results(
        corpus_uri=corpus_uri,
        queries_filename=queries,
        k=k,
        output=output,
    )


if __name__ == "__main__":
    brute_force()
