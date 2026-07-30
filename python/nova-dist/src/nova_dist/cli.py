"""`nova dist <tool>` — orchestrate embed / load / storm across a SkyPilot fleet.

This is dispatched to like any other tool: `nova dist embed ...` execs
`nova-dist embed ...`. Resources + worker install come from a SkyPilot YAML
(`--resources`); this module only injects the per-rank `run:` and launches.

  nova dist embed <config> --resources sky.yaml --num-jobs N
  nova dist load  <config> --resources sky.yaml --num-jobs N   # prepare + fan-out
  nova dist load  <config> --finalize                          # after workers finish
  nova dist storm <config> --resources sky.yaml --num-jobs N   # replicated
"""

from __future__ import annotations

import os
import shutil
import subprocess

from pathlib import Path

import click
import yaml

from nova_dist import sky


def _resolve_binary(name: str) -> str:
    """
    Find a tool binary on the controller: `$NOVA_<NAME>_BIN`, then PATH.
    """
    env = f"NOVA_{name.removeprefix('nova-').upper().replace('-', '_')}_BIN"
    return os.environ.get(env) or shutil.which(name) or name


def _fanout(
    tool: str,
    config: str,
    resources: str,
    num_jobs: int,
    pool_name: str | None, dry_run: bool, run_cmd: str,
    env_extra: list[str],
    extra_file_mounts: dict[str, str] | None = None,
    extra_runtime_envs: dict[str, str] | None = None,
    dry_run_extra=None,
) -> None:
    """
    Stage the config, generate pool+job YAMLs from the user's resources YAML,
    and launch `num_jobs` ranked jobs.
    """
    sky_cfg, source = sky.resolve_resources(tool, resources)
    pool = pool_name or f"nova-{tool}-{Path(config).stem}"
    run_dir = sky.make_run_dir(pool)
    file_mounts, remote_cfg = sky.stage_config(run_dir, config)
    if extra_file_mounts:
        file_mounts.update(extra_file_mounts)

    # `run_cmd` is a template referencing {cfg} and {n}.
    cmd = run_cmd.format(cfg=remote_cfg, n=num_jobs)
    pool_path, job_path = sky.write_pool_and_job(run_dir, sky_cfg, cmd, file_mounts, num_jobs)

    click.echo(f"tool={tool}  config={config}  num_jobs={num_jobs}  pool={pool}")
    click.echo(f"resources: {source}")
    click.echo(f"run: {cmd}")
    click.echo(f"staged: {run_dir}")

    if dry_run:
        sky.print_dry_run(pool, num_jobs, pool_path, job_path)
        if dry_run_extra:
            dry_run_extra()
        return

    envs = sky.forward_env(config, env_extra)
    if extra_runtime_envs:
        envs.update(extra_runtime_envs)
    sky.launch_pool_and_jobs(pool, pool_path, job_path, num_jobs, envs)
    sky.print_monitor(pool)


def _run_local(binary: str, args: list[str]) -> None:
    """
    Run a tool phase on the controller (e.g. load prepare/finalize).
    """
    exe = _resolve_binary(binary)
    click.echo(f"local: {exe} {' '.join(args)}")
    subprocess.run([exe, *args], check=True)


def _stage_catalog_for_workers(config: str, catalog_source: str, remote_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Copy a local catalog parquet into a run-local staging dir and return:
      - file mounts (remote_dir -> local_dir)
      - runtime envs (FINEWEB_S3_CATALOG -> remote parquet path)
    """
    src = Path(catalog_source)
    if not src.exists():
        raise click.UsageError(f"catalog file not found: {src}")
    run_dir = sky.make_run_dir(f"catalog-{Path(config).stem}")
    local_dir = run_dir / "catalog"
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / src.name
    shutil.copy(src, dst)
    mounted_dir = remote_dir.rstrip("/") or "/catalog"
    remote_path = f"{mounted_dir}/{src.name}"
    click.echo(f"catalog: local={src} -> worker={remote_path} (env FINEWEB_S3_CATALOG)")
    return {mounted_dir: str(local_dir)}, {"FINEWEB_S3_CATALOG": remote_path}


def _embed_partition_preview(config: str, num_jobs: int) -> None:
    """
    Best-effort: ask the locally-installed nova-embed to inspect the source and
    print how the dataset partitions across `num_jobs` workers. Skips with a hint
    if nova-embed isn't on the controller (it's only required on the workers).
    """
    exe = _resolve_binary("nova-embed")
    click.echo("\ndata partition (nova-embed --dry-run):")
    try:
        subprocess.run([exe, config, "--num-jobs", str(num_jobs), "--dry-run"], check=False)
    except FileNotFoundError:
        click.echo(
            "  (nova-embed not installed on this controller — run `make embed` to "
            "see the per-worker file/row estimate)"
        )


def _stage_storm_query_source(
    *,
    config: str,
    query_source_path: str,
    remote_dir: str,
) -> tuple[str, dict[str, str]]:
    """
    Stage a local parquet file/dir for nova-storm workers and rewrite
    `query.source.uri` in a generated config copy.
    """
    src = Path(query_source_path).expanduser().resolve()
    if not src.exists():
        raise click.UsageError(f"storm query source not found: {src}")

    run_dir = sky.make_run_dir(f"storm-src-{Path(config).stem}")
    mount_local = run_dir / "query_source_mount"
    mount_local.mkdir(parents=True, exist_ok=True)
    dst = mount_local / src.name
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy(src, dst)

    cfg_data = yaml.safe_load(Path(config).read_text())
    if not isinstance(cfg_data, dict):
        raise click.UsageError("storm config must be a YAML mapping")
    query = cfg_data.get("query")
    if not isinstance(query, dict):
        raise click.UsageError("storm config is missing `query:` block")
    source = query.get("source")
    if not isinstance(source, dict):
        raise click.UsageError("storm config is missing `query.source:` block")

    mounted_dir = remote_dir.rstrip("/") or "/storm-workload"
    remote_uri = f"{mounted_dir}/{src.name}"
    source["uri"] = remote_uri

    staged_cfg = run_dir / "storm_staged_query_source.yaml"
    staged_cfg.write_text(yaml.safe_dump(cfg_data, sort_keys=False))
    click.echo(f"storm query source: local={src} -> worker={remote_uri}")
    return str(staged_cfg), {mounted_dir: str(mount_local)}


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """
    Orchestrate a tool across a SkyPilot fleet (embed / load / storm).
    """


# Shared options for the fan-out commands.
def _common(fn):
    fn = click.argument("config")(fn)
    fn = click.option("--resources", default=None, help="SkyPilot YAML override (resources/setup/envs). Defaults to ~/.nova/skypilot/<tool>.yaml, then built-in defaults.")(fn)
    fn = click.option("--num-jobs", type=int, required=True, help="Number of parallel jobs/workers.")(fn)
    fn = click.option("--pool-name", default=None, help="SkyPilot pool name (default: nova-<tool>-<config>).")(fn)
    fn = click.option("--dry-run", is_flag=True, help="Generate the pool/job YAMLs and print the plan; don't launch.")(fn)
    return fn


@main.command()
@_common
def embed(config, resources, num_jobs, pool_name, dry_run):
    """
    Embed a dataset across a GPU pool. Each rank embeds its slice.
    """
    # nova-embed auto-reads $SKYPILOT_JOB_RANK, so it just needs --num-jobs.
    _fanout(
        "embed", config, resources, num_jobs, pool_name, dry_run,
        run_cmd="nova-embed {cfg} --num-jobs {n}",
        env_extra=["HF_TOKEN", "OPENAI_API_KEY", "HF_HUB_ENABLE_HF_TRANSFER"],
        dry_run_extra=lambda: _embed_partition_preview(config, num_jobs),
    )


@main.command()
@click.argument("config")
@click.option("--resources", default=None, help="SkyPilot YAML override. Defaults to ~/.nova/skypilot/load.yaml, then built-in defaults.")
@click.option("--num-jobs", type=int, default=None, help="Number of parallel load workers.")
@click.option("--pool-name", default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--finalize", is_flag=True, help="Re-enable + await indexing on the controller (run after all workers finish). Does not launch a fleet.")
@click.option("--catalog", "catalog_path", default=None, help="Local parquet catalog file to stage to workers and expose as FINEWEB_S3_CATALOG.")
@click.option("--catalog-remote-dir", default="/catalog", show_default=True, help="Remote mount directory for staged catalog file(s).")
@click.option("--build-catalog-input", default=None, help="Build catalog locally from this parquet root before launching workers.")
@click.option("--build-catalog-output", default=None, help="Output parquet path for --build-catalog-input.")
@click.option("--build-catalog-resume", is_flag=True, help="Resume catalog build when used with --build-catalog-input.")
def load(
    config,
    resources,
    num_jobs,
    pool_name,
    dry_run,
    finalize,
    catalog_path,
    catalog_remote_dir,
    build_catalog_input,
    build_catalog_output,
    build_catalog_resume,
):
    """
    Load pre-embedded parquet across a pool.

    Lifecycle: this command runs `prepare` on the controller, then fans out N
    `load` workers (each takes its file slice). When they've all finished, run
    `nova dist load <config> --finalize` to build the index.
    """
    if finalize:
        _run_local("nova-load", ["finalize", config])
        return
    if num_jobs is None:
        raise click.UsageError("--num-jobs is required (unless --finalize).")
    if build_catalog_input and not build_catalog_output:
        raise click.UsageError("--build-catalog-output is required with --build-catalog-input.")

    if build_catalog_input and not dry_run:
        args = [
            "catalog-build",
            "--input",
            build_catalog_input,
            "--output",
            build_catalog_output,
        ]
        if build_catalog_resume:
            args.append("--resume")
        _run_local("nova-load", args)

    catalog_source = build_catalog_output if build_catalog_input else catalog_path
    extra_mounts = None
    extra_envs = None
    if catalog_source:
        extra_mounts, extra_envs = _stage_catalog_for_workers(
            config=config,
            catalog_source=catalog_source,
            remote_dir=catalog_remote_dir,
        )

    # 1. Master creates the collection + defers indexing.
    if not dry_run:
        _run_local("nova-load", ["prepare", config])

    # 2. Fan out the load workers. The Rust loader doesn't auto-read the rank,
    #    so pass it explicitly from the env SkyPilot sets per job.
    _fanout(
        "load", config, resources, num_jobs, pool_name, dry_run,
        run_cmd="nova-load load {cfg} --num-jobs {n} --job-rank $SKYPILOT_JOB_RANK",
        env_extra=["QDRANT_URL", "QDRANT_API_KEY"],
        extra_file_mounts=extra_mounts,
        extra_runtime_envs=extra_envs,
    )

    if not dry_run:
        click.echo("\nwhen all workers finish, build the index:")
        click.echo(f"  nova dist load {config} --finalize")


@main.command()
@_common
@click.option(
    "--stage-query-source",
    default=None,
    help="Stage this local parquet file/dir to workers and rewrite query.source.uri to the mounted path.",
)
@click.option(
    "--query-source-remote-dir",
    default="/storm-workload",
    show_default=True,
    help="Remote mount directory used with --stage-query-source.",
)
def storm(
    config,
    resources,
    num_jobs,
    pool_name,
    dry_run,
    stage_query_source,
    query_source_remote_dir,
):
    """
    Load-test a vector store with `num_jobs` replicated workers (not sliced).
    """
    cfg_for_fanout = config
    extra_mounts = None
    if stage_query_source:
        cfg_for_fanout, extra_mounts = _stage_storm_query_source(
            config=config,
            query_source_path=stage_query_source,
            remote_dir=query_source_remote_dir,
        )

    _fanout(
        "storm",
        cfg_for_fanout,
        resources,
        num_jobs,
        pool_name,
        dry_run,
        run_cmd="nova-storm {cfg}",
        env_extra=["QDRANT_URL", "QDRANT_API_KEY"],
        extra_file_mounts=extra_mounts,
    )


@main.command()
@click.argument("config")
def sweep(config) -> None:
    """
    Fan a `nova sweep` out across a SkyPilot fleet — not implemented yet.

    Multi-target distribution (bin-packing slices across N SkyPilot-launched
    Qdrant instances, and `nova-sweep --num-jobs`/`--job-rank` to pick which
    target's slices a process owns) hasn't been built — see the `nova sweep`
    plan's "Explicitly deferred" section. This stub exists so `nova dist
    sweep <config>` fails loudly and specifically instead of the dispatcher's
    generic "unknown command", or the subcommand being silently absent.
    """
    click.echo(
        "error: nova dist sweep is not implemented yet — run nova-sweep directly "
        "by hand in the meantime",
        err=True,
    )
    raise SystemExit(1)


@main.group()
def bf() -> None:
    """
    Brute-force ground truth: fan out `compute`, then `merge` on the controller.
    """


@bf.command("compute")
@_common
def bf_compute(config, resources, num_jobs, pool_name, dry_run):
    """
    Search the corpus across a GPU pool — each rank takes a slice of the files.
    """
    _fanout(
        "bf", config, resources, num_jobs, pool_name, dry_run,
        run_cmd="nova-bf compute {cfg} --num-jobs {n} --job-rank $SKYPILOT_JOB_RANK",
        env_extra=["HF_TOKEN"],
    )
    if not dry_run:
        click.echo("\nwhen all workers finish, merge the partials:")
        click.echo(f"  nova dist bf merge {config}")


@bf.command("merge")
@click.argument("config")
def bf_merge(config):
    """
    Merge per-rank partial results into each search's own top-K parquet (runs on the controller).
    """
    _run_local("nova-bf", ["merge", config])


if __name__ == "__main__":
    main()
