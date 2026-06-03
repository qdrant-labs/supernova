"""
Shared SkyPilot helpers used across all nova *-dist and EC2-launch commands.

All launches go through the SkyPilot Python SDK (sky.jobs.launch /
sky.jobs.pool_apply) and stream their progress to stdout via
sky.stream_and_get. ``sky`` is imported lazily inside each helper because
the package pulls in a lot at import time.
"""

import os
import shutil
import sys

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
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


# --- Worker bootstrap -------------------------------------------------------
#
# Workers install a *pinned, named* supernova rather than mounting the
# controller's working directory. This is the fix for issue #7: the old
# `file_mounts {/app: "."}` + `uv sync` bootstrap assumed `nova *-dist` was
# always launched from the repo root. Naming an artifact (a PyPI version)
# instead of shipping a directory decouples "where I launched from" from
# "what code the worker runs".

# The worker venv lives at a fixed path so the pool's `setup` phase (which
# installs supernova) and the job's `run` phase (which invokes it) agree on
# where the `nova` entrypoint is — the two run in separate shells on the same node.
WORKER_VENV = "$HOME/.nova-venv"


def worker_version() -> str:
    """Installed supernova version.

    Workers pin to this exact version so the controller and its workers always
    run identical code.
    """
    try:
        return version("supernova")
    except PackageNotFoundError as exc:  # pragma: no cover - dev-only guard
        raise RuntimeError(
            "supernova is not installed in this environment, so distributed "
            "workers can't be pinned to a matching version. Install it first "
            "(e.g. `uv sync`) before launching a distributed run."
        ) from exc


def worker_install_spec(extra: str | None = None) -> str:
    """pip/uv install target for workers.

    ``extra`` is the optional-dependency group (e.g. "load", "embed"); pass
    None for commands that only need base deps (e.g. generate-queries).

    Default: the pinned PyPI release matching the controller's own version.
    Override with ``NOVA_WORKER_INSTALL_SPEC`` to test code that isn't on PyPI
    yet — a git ref or a (presigned) wheel URL. A literal ``{extra}`` in the
    override is replaced with this job's extra, e.g.::

        NOVA_WORKER_INSTALL_SPEC='supernova[{extra}] @ git+https://github.com/qdrant-labs/supernova@<sha>'
    """
    bracket = f"[{extra}]" if extra else ""
    override = os.environ.get("NOVA_WORKER_INSTALL_SPEC")
    if override:
        return override.replace("{extra}", extra or "")
    return f"supernova{bracket}=={worker_version()}"


def build_worker_setup(extra: str | None = None) -> str:
    """SkyPilot ``setup`` script: install uv, then install supernova[extra]
    into a fixed venv. ``extra`` may be None for base-only commands.
    Replaces the old mount-CWD + ``uv sync`` bootstrap."""
    spec = worker_install_spec(extra)
    return (
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        f"uv venv {WORKER_VENV} && "
        f'uv pip install --python {WORKER_VENV}/bin/python "{spec}"'
    )


def worker_run(argv: str) -> str:
    """SkyPilot ``run`` command invoking the installed ``nova`` entrypoint."""
    return f"{WORKER_VENV}/bin/nova {argv}"


def config_mount(run_dir: Path, config_path: str) -> tuple[dict[str, str], str]:
    """Stage a config for a worker; return ``(file_mounts, remote_path)``.

    The config is copied verbatim (unresolved) into the run dir and mounted at
    ``/cfg`` on the worker. We ship the *raw* config — secrets travel as
    forwarded env vars and ``nova`` resolves ``${VAR}`` at run time — so no
    secret is ever written to the worker's disk.
    """
    src = Path(config_path)
    staged_dir = run_dir / "cfg"
    staged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, staged_dir / src.name)
    return {"/cfg": str(staged_dir)}, f"/cfg/{src.name}"
