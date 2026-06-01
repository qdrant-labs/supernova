"""
Shared SkyPilot helpers used across all vf *-dist and EC2-launch commands.

All launches go through the SkyPilot Python SDK (sky.jobs.launch /
sky.jobs.pool_apply) and stream their progress to stdout via
sky.stream_and_get. ``sky`` is imported lazily inside each helper because
the package pulls in a lot at import time.
"""

import os
import sys

from datetime import datetime
from pathlib import Path

# Base AWS credential vars forwarded to every SkyPilot job.
AWS_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
]

# Deep-learning AMIs with CUDA pre-installed, keyed by AWS region.
# Required for GPU workers — without an explicit image_id SkyPilot picks a
# plain Ubuntu AMI that has no GPU drivers.
CUDA_IMAGE_IDS = {
    "us-east-1": "ami-0038d79e7270bb987",
    "us-west-2": "ami-08a03808395c1b31f",
    "us-east-2": "ami-0a28b3d7e7c9192a7",
}


def build_env_dict(extra_vars: list[str] | None = None) -> dict[str, str]:
    """
    Collect env vars to forward to a SkyPilot job, as a dict suitable for
    ``sky.Task.update_envs(...)``.

    Always forwards AWS credential vars. Pass extra_vars for tool-specific
    secrets (HF_TOKEN, QDRANT_URL, OPENAI_API_KEY, etc.).
    """
    envs: dict[str, str] = {}
    for var in AWS_ENV_VARS + (extra_vars or []):
        val = os.environ.get(var)
        if val:
            envs[var] = val
    return envs


def make_run_dir(name: str) -> Path:
    """Create runs/{timestamp}_{name}/ and return the path."""
    run_dir = Path("runs") / f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def launch_single_job(job_path: Path, envs: dict[str, str]) -> None:
    """Submit a single SkyPilot job (no pool) and stream its progress."""
    import sky

    task = sky.Task.from_yaml(str(job_path))
    if envs:
        task.update_envs(envs)

    request_id = sky.jobs.launch(task)
    sky.stream_and_get(request_id, output_stream=sys.stdout)


def launch_pool_and_jobs(
    pool_name: str,
    pool_path: Path,
    job_path: Path,
    num_jobs: int,
    envs: dict[str, str],
) -> None:
    """
    Apply a pool config and submit ``num_jobs`` parallel jobs to that pool.

    Env vars are attached to both the pool task (visible during setup) and
    the job task (visible at run time) to match the prior ``--env`` flag
    behavior, which forwarded them to both phases.
    """
    import sky

    pool_task = sky.Task.from_yaml(str(pool_path))
    if envs:
        pool_task.update_envs(envs)
    pool_req = sky.jobs.pool_apply(
        pool_task, pool_name, mode=sky.serve.UpdateMode.ROLLING
    )
    sky.stream_and_get(pool_req, output_stream=sys.stdout)

    job_task = sky.Task.from_yaml(str(job_path))
    if envs:
        job_task.update_envs(envs)
    job_req = sky.jobs.launch(job_task, pool=pool_name, num_jobs=num_jobs)
    sky.stream_and_get(job_req, output_stream=sys.stdout)


def launch_single_job_to_pool(
    pool_name: str,
    job_path: Path,
    envs: dict[str, str],
) -> None:
    """Submit a single SkyPilot job to an existing pool (does not create a pool)."""
    import sky

    task = sky.Task.from_yaml(str(job_path))
    if envs:
        task.update_envs(envs)
    request_id = sky.jobs.launch(task, pool=pool_name)
    sky.stream_and_get(request_id, output_stream=sys.stdout)


def print_dry_run(
    pool_name: str, num_jobs: int, pool_path: Path, job_path: Path
) -> None:
    """Print the dry-run summary for a pool + jobs launch."""
    print(f"\n[dry run] Would create pool '{pool_name}' and submit {num_jobs} jobs")
    print(f"  Pool config: {pool_path}")
    print(f"  Job config:  {job_path}")
    print("\nTo run manually:")
    print(f"  sky jobs pool apply -p {pool_name} {pool_path}")
    print(f"  sky jobs launch -p {pool_name} --num-jobs {num_jobs} {job_path}")


def print_monitor(pool_name: str) -> None:
    """Print standard pool monitor / logs / teardown instructions."""
    print(f"Monitor:   sky jobs pool status {pool_name}")
    print(f"Logs:      sky jobs pool logs {pool_name}")
    print(f"Tear down: sky jobs pool down {pool_name}")
