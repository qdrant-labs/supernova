#!/usr/bin/env python3
"""
Sample N rows from an embedded corpus as eval query vectors.

Default mode launches a single high-bandwidth EC2 instance in the corpus's
region via SkyPilot (S3 only). Use --local to run the full pipeline
in-process (also what the EC2 job calls).
"""

import logging

import click
import yaml

from cli.skypilot_utils import build_env_flags, make_run_dir, launch_single_job
from vectorforge.destinations import S3Destination, parse_destination
from vectorforge.eval.generate_queries import generate_queries as _generate_queries
from vectorforge.utils import get_bucket_region

logger = logging.getLogger(__name__)

# "n"-family = enhanced networking (25-100Gbps). S3 reads that take
# minutes locally finish in seconds in-region.
DEFAULT_INSTANCE_TYPE = "r5n.2xlarge"  # 8 vCPU, 64GB RAM, 25Gbps


def launch_on_ec2(
    corpus_uri: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    instance_type: str,
    on_demand: bool,
    dry_run: bool,
    prefetch: bool = False,
):
    dest = parse_destination(corpus_uri)
    if not isinstance(dest, S3Destination):
        # In-region EC2 launch only makes sense for S3 (we co-locate the
        # instance with the bucket). For HF corpora, run --local from a
        # high-bandwidth machine instead.
        raise click.UsageError(
            f"EC2 launch is supported for s3:// corpora only. For {corpus_uri}, "
            "use --local to run in-process."
        )
    region = get_bucket_region(dest.bucket)
    click.echo(f"Bucket region: {region}")

    worker_flags = f"-n {n} --seed {seed} --output {output} --local"
    if columns:
        worker_flags += " " + " ".join(f"--columns {c}" for c in columns)
    if prefetch:
        worker_flags += " --prefetch"

    run_dir = make_run_dir("generate-queries")

    job_yaml = {
        "name": "vf-generate-queries",
        "resources": {
            "cloud": "aws",
            "region": region,
            "instance_type": instance_type,
            "use_spot": not on_demand,
        },
        "file_mounts": {"/app": "."},
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync",
        "run": f"cd /app && uv run vf generate-queries {corpus_uri} {worker_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    click.echo("=" * 60)
    click.echo("vectorforge generate-queries plan")
    click.echo("=" * 60)
    click.echo(f"  Corpus URI:  {corpus_uri}")
    click.echo(f"  Queries:     {n}  (seed={seed})")
    click.echo(f"  Region:      {region}")
    click.echo(
        f"  Instance:    {instance_type}  ({'on-demand' if on_demand else 'spot'})"
    )
    click.echo(f"  Columns:     {columns or 'all'}")
    click.echo(
        f"  Fetch mode:  {'prefetch (download-first)' if prefetch else 'range requests'}"
    )
    click.echo(f"  Output:      {dest.eval_uri(output)}")
    click.echo(f"  Run dir:     {run_dir}")
    click.echo("=" * 60)

    if dry_run:
        click.echo(f"\n[dry run] Job config: {job_path}")
        click.echo(f"To run manually: sky jobs launch -y {job_path}")
        return

    launch_single_job(job_path, build_env_flags())

    click.echo(f"\nOutput will be at {dest.eval_uri(output)}")
    click.echo("Monitor: sky jobs logs")
    click.echo("Cancel:  sky jobs cancel -a")


@click.command(
    name="generate-queries",
    help="Sample N rows as eval queries (launches EC2; --local to run here).",
)
@click.argument("corpus_uri")
@click.option(
    "-n", "--num-queries", "num_queries", type=int, default=1000, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--columns",
    multiple=True,
    default=(),
    help="Columns to fetch (default: all). Repeat for each: "
    "--columns dense_embedding --columns sparse_embedding",
)
@click.option(
    "--output", default=None, help="Output filename (default: queries_<n>.parquet)."
)
@click.option(
    "--local",
    is_flag=True,
    help="Run the full pipeline in-process instead of launching EC2.",
)
@click.option(
    "--prefetch",
    is_flag=True,
    help="Download each parquet fully before reading (better for large row groups).",
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
def generate_queries(
    corpus_uri,
    num_queries,
    seed,
    columns,
    output,
    local,
    prefetch,
    instance_type,
    on_demand,
    dry_run,
):
    """Sample N eval query rows from an embedded corpus."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    try:
        parse_destination(corpus_uri)
    except ValueError as e:
        raise click.UsageError(str(e))

    if output is None:
        output = f"queries_{num_queries}.parquet"

    columns_list = list(columns) if columns else None

    if local:
        _generate_queries(
            corpus_uri=corpus_uri,
            n=num_queries,
            seed=seed,
            columns=columns_list,
            output=output,
            prefetch=prefetch,
        )
    else:
        launch_on_ec2(
            corpus_uri=corpus_uri,
            n=num_queries,
            seed=seed,
            columns=columns_list,
            output=output,
            instance_type=instance_type,
            on_demand=on_demand,
            dry_run=dry_run,
            prefetch=prefetch,
        )


if __name__ == "__main__":
    generate_queries()
