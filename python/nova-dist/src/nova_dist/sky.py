"""SkyPilot orchestration helpers for `nova dist`.

The user supplies a SkyPilot YAML (`--resources`) that owns the hard part:
`resources:` (cloud, accelerators, spot…), `setup:` (how to install the tool on
a worker), and any `envs:`. We add only the pool wrapper, the staged config, and
the per-rank `run:` command, then launch via the SkyPilot SDK.

`sky` is imported lazily inside each function — it pulls in a lot at import time,
and `nova dist --help` / `--dry-run` shouldn't pay for it.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

from datetime import datetime
from pathlib import Path

import yaml

# Always forwarded to workers (workers re-read secrets from the env; nothing is
# written to their disk). Tool-specific extras are added per command.
AWS_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
]


def nova_home() -> Path:
    """
    Root for local run state (staged configs, generated yamls). `$NOVA_HOME`
    overrides; defaults to `~/.nova`. Deliberately outside any repo.
    """
    return Path(os.environ.get("NOVA_HOME", Path.home() / ".nova"))


def make_run_dir(name: str) -> Path:
    run_dir = nova_home() / "runs" / f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def referenced_env_vars(config_path: str | Path) -> list[str]:
    """
    Env vars the workload config names via `${VAR}` — forward exactly these
    (vendor-agnostic: a differently-named secret is picked up automatically).
    """
    return sorted(set(re.findall(r"\$\{(\w+)\}", Path(config_path).read_text())))


def forward_env(config_path: str, extra: list[str] | None = None) -> dict[str, str]:
    """
    Env vars to attach to the launch: AWS creds + everything the config
    references + per-tool baselines, filtered to those actually set.
    """
    wanted = AWS_ENV_VARS + referenced_env_vars(config_path) + (extra or [])
    return {v: os.environ[v] for v in dict.fromkeys(wanted) if os.environ.get(v)}


def load_resources_yaml(path: str) -> dict:
    """
    Read the user's SkyPilot YAML. We use its `resources` / `setup` / `envs`;
    a `run:` in it is ignored (we inject the per-rank command).
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "resources" not in data:
        raise ValueError(f"{path}: a SkyPilot resources YAML must define `resources:`")
    if data.get("run"):
        # Harmless but a footgun — make it explicit that we override it.
        data.pop("run")
    return data


def stage_config(run_dir: Path, config_path: str) -> tuple[dict, str]:
    """
    Copy the (raw, unresolved) workload config into the run dir and return
    `(file_mounts, remote_path)` mounting it at `/cfg` on the worker.
    """
    src = Path(config_path)
    staged = run_dir / "cfg"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, staged / src.name)
    return {"/cfg": str(staged)}, f"/cfg/{src.name}"


def write_pool_and_job(
    run_dir: Path,
    sky_cfg: dict,
    run_cmd: str,
    file_mounts: dict,
    num_jobs: int,
) -> tuple[Path, Path]:
    """
    Generate the pool + job YAMLs from the user's resources YAML.

    Pool task carries resources + setup + the staged config (visible at install
    time). Job task carries resources + envs + the injected `run:` (pool-submitted
    jobs must restate resources but must NOT set setup/file_mounts).
    """
    resources = sky_cfg["resources"]
    pool_yaml = {
        "pool": {"min_workers": num_jobs, "max_workers": num_jobs},
        "resources": resources,
        "file_mounts": file_mounts,
    }
    if sky_cfg.get("setup"):
        pool_yaml["setup"] = sky_cfg["setup"]
    job_yaml = {
        "name": run_dir.name,
        "resources": resources,
        "run": run_cmd,
    }
    if sky_cfg.get("envs"):
        job_yaml["envs"] = sky_cfg["envs"]

    pool_path = run_dir / "pool.yaml"
    job_path = run_dir / "job.yaml"
    pool_path.write_text(yaml.dump(pool_yaml, sort_keys=False))
    job_path.write_text(yaml.dump(job_yaml, sort_keys=False))
    return pool_path, job_path


def launch_pool_and_jobs(
    pool_name: str, pool_path: Path, job_path: Path, num_jobs: int, envs: dict
) -> None:
    """
    Apply the pool, then submit `num_jobs` ranked jobs to it. Each job sees
    `$SKYPILOT_JOB_RANK` in `[0, num_jobs)`.
    """
    import sky

    pool_task = sky.Task.from_yaml(str(pool_path))
    if envs:
        pool_task.update_envs(envs)
    sky.stream_and_get(
        sky.jobs.pool_apply(pool_task, pool_name, mode=sky.serve.UpdateMode.ROLLING),
        output_stream=sys.stdout,
    )

    job_task = sky.Task.from_yaml(str(job_path))
    if envs:
        job_task.update_envs(envs)
    sky.stream_and_get(
        sky.jobs.launch(job_task, pool=pool_name, num_jobs=num_jobs),
        output_stream=sys.stdout,
    )


def print_dry_run(pool_name: str, num_jobs: int, pool_path: Path, job_path: Path) -> None:
    print(f"\n[dry run] would create pool '{pool_name}' and submit {num_jobs} job(s)")
    print(f"  pool: {pool_path}")
    print(f"  job:  {job_path}")
    print("\nrun manually with the sky CLI:")
    print(f"  sky jobs pool apply -p {pool_name} {pool_path}")
    print(f"  sky jobs launch -p {pool_name} --num-jobs {num_jobs} {job_path}")


def print_monitor(pool_name: str) -> None:
    print("\nmonitor:   sky jobs queue")
    print("logs:      sky jobs logs <job-id>")
    print(f"tear down: sky jobs pool down {pool_name}")
