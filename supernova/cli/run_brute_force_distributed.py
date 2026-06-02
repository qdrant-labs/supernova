#!/usr/bin/env python3
"""
Distributed brute-force nearest-neighbor search via a SkyPilot GPU pool.

Splits the corpus across N GPU workers. Each worker prefetches its assigned
files to local NVMe, runs GPU similarity search, and saves a partial top-K
result under {corpus_uri}/eval/_bf_partial_*. Run `nova brute-force-merge` when
all workers finish.

Today this command provisions AWS GPU instances, so it requires an s3://
corpus URI (we co-locate the workers in the bucket's region). For an hf://
corpus, run `nova brute-force --local` from a high-bandwidth machine.
"""

import logging
import math
from pathlib import Path

import click
import yaml

from supernova.cli.run_brute_force import DEFAULT_ACCELERATOR, DEFAULT_INSTANCE_TYPE
from supernova.cli.skypilot_utils import (
    CUDA_IMAGE_IDS,
    build_env_dict,
    launch_pool_and_jobs,
    make_run_dir,
    print_monitor,
)
from supernova.destinations import (
    S3Destination,
    discover_corpus_parquets,
    parse_destination,
)
from supernova.eval.brute_force import DEFAULT_K, DistanceMetric, partial_subkey
from supernova.utils import get_bucket_region

logger = logging.getLogger(__name__)

DEFAULT_NUM_JOBS = 50

_METRIC_CHOICES = [m.value for m in DistanceMetric]


@click.command(
    name="brute-force-dist", help="Distributed brute-force via SkyPilot GPU pool."
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
    help="Neighbors per query.",
)
@click.option(
    "--metric",
    type=click.Choice(_METRIC_CHOICES, case_sensitive=False),
    default=DistanceMetric.COSINE.value,
    show_default=True,
)
@click.option("--dense-column", default="dense_embedding", show_default=True)
@click.option(
    "--num-jobs",
    type=int,
    default=DEFAULT_NUM_JOBS,
    show_default=True,
    help="Number of GPU workers.",
)
@click.option("--output", default=None, help="Final merged output filename.")
@click.option(
    "--instance-type",
    default=DEFAULT_INSTANCE_TYPE,
    show_default=True,
    help="EC2 instance type per worker.",
)
@click.option("--on-demand", is_flag=True, help="Use on-demand instead of spot.")
@click.option(
    "--pool-name", default=None, help="SkyPilot pool name (default: auto-generated)."
)
@click.option(
    "--dry-run", is_flag=True, help="Print plan and write configs, don't launch."
)
def brute_force_dist(
    corpus_uri,
    queries,
    k,
    metric,
    dense_column,
    num_jobs,
    output,
    instance_type,
    on_demand,
    pool_name,
    dry_run,
):
    """Distributed brute-force nearest-neighbor search via SkyPilot GPU pool."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    try:
        dest = parse_destination(corpus_uri)
    except ValueError as e:
        raise click.UsageError(str(e))
    if not isinstance(dest, S3Destination):
        raise click.UsageError(
            f"brute-force-dist provisions AWS GPU workers, so it needs an s3:// corpus. "
            f"For {corpus_uri}, run `nova brute-force --local` instead."
        )

    metric_enum = DistanceMetric(metric)
    queries_stem = Path(queries).stem
    output_eff = output or f"brute_force_{queries_stem}_k{k}.parquet"

    region = get_bucket_region(dest.bucket)
    image_id = CUDA_IMAGE_IDS.get(region)
    if image_id is None:
        click.echo(
            f"Warning: no CUDA AMI configured for {region!r}. Known: {list(CUDA_IMAGE_IDS)}"
        )

    corpus_uris = discover_corpus_parquets(dest)
    files_per_worker = math.ceil(len(corpus_uris) / num_jobs)

    pool_name_eff = pool_name or f"nova-bf-{queries_stem}"
    run_dir = make_run_dir("brute-force-dist")

    resources = {
        "cloud": "aws",
        "region": region,
        "instance_type": instance_type,
        "accelerators": DEFAULT_ACCELERATOR,
        "use_spot": not on_demand,
    }
    if image_id:
        resources["image_id"] = image_id

    worker_flags = (
        f"--queries {queries} -k {k} "
        f"--metric {metric_enum.value} --dense-column {dense_column} "
        f"--num-jobs {num_jobs} --local"
    )

    pool_yaml = {
        "pool": {
            "min_workers": num_jobs,
            "max_workers": num_jobs,
        },
        "resources": resources,
        "file_mounts": {"/app": "."},
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra eval",
    }
    job_yaml = {
        "name": f"nova-bf-{queries_stem}",
        "resources": resources,
        "run": f"cd /app && uv run nova brute-force {corpus_uri} {worker_flags}",
    }

    pool_path = run_dir / "pool.yaml"
    job_path = run_dir / "job.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    partial_uri = dest.eval_uri(partial_subkey(queries_stem, k))
    merge_cmd = (
        f"nova brute-force-merge {corpus_uri} --queries {queries} "
        f"-k {k} --output {output_eff}"
    )

    click.echo("=" * 60)
    click.echo("supernova brute-force-dist plan")
    click.echo("=" * 60)
    click.echo(f"  Corpus URI:     {corpus_uri}")
    click.echo(f"  Queries:        {queries}")
    click.echo(f"  K:              {k}")
    click.echo(f"  Metric:         {metric_enum.value}")
    click.echo(f"  Workers:        {num_jobs}")
    click.echo(f"  Files/worker:   ~{files_per_worker} (of {len(corpus_uris)} total)")
    click.echo(
        f"  Instance:       {instance_type}  ({'on-demand' if on_demand else 'spot'})"
    )
    click.echo(f"  Region:         {region}")
    click.echo(f"  Pool:           {pool_name_eff}")
    click.echo(f"  Partial output: {partial_uri}/")
    click.echo(f"  Final output:   {dest.eval_uri(output_eff)}")
    click.echo(f"  Run dir:        {run_dir}")
    click.echo("=" * 60)
    click.echo("\nWhen all workers finish, run:")
    click.echo(f"  {merge_cmd}")
    click.echo("")

    if dry_run:
        click.echo(f"[dry run] Pool config: {pool_path}")
        click.echo(f"[dry run] Job config:  {job_path}")
        return

    launch_pool_and_jobs(
        pool_name_eff, pool_path, job_path, num_jobs, build_env_dict()
    )

    click.echo(f"Submitted {num_jobs} workers to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)
    click.echo(f"\nWhen done: {merge_cmd}")


if __name__ == "__main__":
    brute_force_dist()
