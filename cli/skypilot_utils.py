"""
Shared SkyPilot helpers used across all vf *-dist and EC2-launch commands.
"""

import os
import subprocess

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


def build_env_flags(extra_vars: list[str] | None = None) -> list[str]:
    """
    Build --env KEY=VAL flags for sky jobs commands from the current environment.

    Always forwards AWS credential vars. Pass extra_vars for tool-specific
    secrets (HF_TOKEN, QDRANT_URL, OPENAI_API_KEY, etc.).
    """
    flags = []
    for var in AWS_ENV_VARS + (extra_vars or []):
        val = os.environ.get(var)
        if val:
            flags.extend(["--env", f"{var}={val}"])
    return flags


def make_run_dir(name: str) -> Path:
    """Create runs/{timestamp}_{name}/ and return the path."""
    run_dir = Path("runs") / f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def launch_pool_and_jobs(
    pool_name: str,
    pool_path: Path,
    job_path: Path,
    num_jobs: int,
    env_flags: list[str],
) -> None:
    """
    Create a SkyPilot pool and submit num_jobs parallel jobs to it.
    """
    subprocess.run(
        ["sky", "jobs", "pool", "apply", "-p", pool_name, str(pool_path), *env_flags],
        check=True,
    )
    subprocess.run(
        [
            "sky",
            "jobs",
            "launch",
            "-p",
            pool_name,
            "--num-jobs",
            str(num_jobs),
            "-y",
            str(job_path),
            *env_flags,
        ],
        check=True,
    )


def launch_single_job(job_path: Path, env_flags: list[str]) -> None:
    """Submit a single SkyPilot job (no pool)."""
    subprocess.run(
        ["sky", "jobs", "launch", "-y", str(job_path), *env_flags], check=True
    )


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
