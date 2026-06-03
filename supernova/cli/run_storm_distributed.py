#!/usr/bin/env python3
"""Dispatch a distributed load test ("storm") via a SkyPilot pool.

Unlike embed/load, this is **replicated, not partitioned**: every worker runs
the *same* `nova storm` profile, so total offered load ≈ num_workers ×
per-worker concurrency. Defaults to on-demand instances — spot preemption
mid-run would corrupt the measurement window.
"""

import json
import logging
import re
from pathlib import Path

import click
import yaml

from supernova.cli.skypilot_utils import (
    build_env_dict,
    build_worker_setup,
    config_mount,
    launch_pool_and_jobs,
    make_run_dir,
    print_dry_run,
    print_monitor,
    worker_run,
)

logger = logging.getLogger(__name__)

DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory": 8,
    "cloud": "aws",
    "use_spot": False,  # on-demand: a preempted load generator skews the results
}


@click.command(
    name="storm-dist", help="Distributed load test via SkyPilot pool (replicated)."
)
@click.argument("config")
@click.option(
    "--dry-run", is_flag=True, help="Generate configs and print plan, don't launch."
)
@click.option(
    "--num-workers", type=int, default=None, help="Override dispatch.num_workers."
)
@click.option(
    "--pool-name", default=None, help="SkyPilot pool name (default: auto-generated)."
)
@click.option(
    "--spot",
    is_flag=True,
    help="Use spot instances (default: on-demand; preemption skews a load test).",
)
def storm_dist(config, dry_run, num_workers, pool_name, spot):
    """Dispatch a replicated storm across a SkyPilot pool."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    from supernova.cli.run_loader import resolve_config

    with open(config) as f:
        cfg = resolve_config(yaml.safe_load(f))

    dispatch = cfg.get("dispatch", {})
    num_workers_eff = num_workers or dispatch.get("num_workers", 4)
    run_name = dispatch.get("run_name", Path(config).stem)
    pool_name_eff = pool_name or f"nova-storm-{run_name}"

    resources = dict(cfg.get("resources") or DEFAULT_RESOURCES)
    resources["use_spot"] = bool(spot)

    run_dir = make_run_dir(pool_name_eff)
    cfg_mounts, remote_cfg = config_mount(run_dir, config)

    click.echo("=" * 60)
    click.echo("supernova distributed storm plan (replicated load)")
    click.echo("=" * 60)
    click.echo(f"  Target:       {cfg['target'].get('type', 'qdrant')} @ {cfg['target'].get('url', '?')}")
    click.echo(f"  Workers:      {num_workers_eff}  (each runs the full load profile)")
    click.echo(f"  Per-worker:   concurrency={cfg.get('load', {}).get('concurrency', 32)}")
    click.echo(f"  Instances:    {'spot' if spot else 'on-demand'}")
    click.echo(f"  Run dir:      {run_dir}")
    click.echo("=" * 60)

    pool_yaml = {
        "pool": {"min_workers": num_workers_eff, "max_workers": num_workers_eff},
        "resources": resources,
        "file_mounts": cfg_mounts,
        "setup": build_worker_setup("storm"),
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    # Replication: every worker runs the SAME full profile (no --num-jobs/rank).
    job_yaml = {
        "name": f"storm-{run_name}",
        "resources": resources,
        "run": worker_run(f"storm {remote_cfg}"),
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    manifest = {
        "run_id": run_dir.name,
        "config": config,
        "num_workers": num_workers_eff,
        "pool_name": pool_name_eff,
        "mode": "storm",
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if dry_run:
        print_dry_run(pool_name_eff, num_workers_eff, pool_path, job_path)
        return

    # Forward exactly the secrets the config references via ${VAR} — vendor-agnostic,
    # so a non-Qdrant target's creds get forwarded without a hardcoded list.
    referenced = sorted(set(re.findall(r"\$\{(\w+)\}", Path(config).read_text())))
    envs = build_env_dict(referenced)
    logger.info("Creating pool '%s' with %d workers...", pool_name_eff, num_workers_eff)
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_workers_eff, envs)

    click.echo(f"\nLaunched {num_workers_eff} storm workers on pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)
    click.echo(
        "\nTODO: each worker prints its own p50/p95/p99. For fleet-wide numbers, "
        "have workers write raw latency samples to S3 / a TSDB and add a "
        "`storm-merge` step that combines the distributions (never average p99s)."
    )


if __name__ == "__main__":
    storm_dist()