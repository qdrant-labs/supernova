"""
Shared SkyPilot helpers used across all nova *-dist and EC2-launch commands.

All launches go through the SkyPilot Python SDK (sky.jobs.launch /
sky.jobs.pool_apply) and stream their progress to stdout via
sky.stream_and_get. ``sky`` is imported lazily inside each helper because
the package pulls in a lot at import time.
"""

import os
import re
import shutil
import sys

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

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


def referenced_env_vars(config_path: str | Path) -> list[str]:
    """Env vars the config references via ``${VAR}`` — forward exactly these to
    workers. Vendor-agnostic: a non-Qdrant target or a differently-named secret
    is picked up without a hardcoded provider list. Pair with a per-tool baseline
    (e.g. HF_TOKEN) for auth that's read from the env but never named in the YAML.
    """
    return sorted(set(re.findall(r"\$\{(\w+)\}", Path(config_path).read_text())))


def nova_home() -> Path:
    """Root for nova's local state — run metadata, and a home for future caches.

    Defaults to ``~/.nova``; override with ``$NOVA_HOME``. Deliberately outside
    any project directory so an installed ``nova`` writes to a stable location
    no matter where it's invoked, and so SkyPilot's file-mount staging never
    runs ``git ls-files`` inside a repo subtree (which trips a git bug).
    """
    return Path(os.environ.get("NOVA_HOME", Path.home() / ".nova"))


def load_resource_defaults() -> dict:
    """User-level SkyPilot resource defaults, so the fleet spec lives in one
    place instead of every run's config.

    Read from ``$NOVA_RESOURCES_FILE`` if set, else ``$NOVA_HOME/resources.yaml``;
    returns ``{}`` when absent. Shape: an optional ``all:`` base merged under
    every tool, plus per-tool sections, each a SkyPilot resources dict::

        all:   {cloud: aws}
        load:  {cpus: 8}
        storm: {cpus: 4}
        embed: {accelerators: A10G:1, instance_type: g5.4xlarge}
    """
    override = os.environ.get("NOVA_RESOURCES_FILE")
    path = Path(override) if override else nova_home() / "resources.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_resources(
    tool: str,
    config_resources: dict | None = None,
    overrides: dict | None = None,
    builtin: dict | None = None,
) -> dict:
    """Resolve a SkyPilot ``resources`` dict by layering, low → high:

      built-in tool default  <  ``~/.nova`` file (``all:`` then ``<tool>:``)
        <  per-run config ``resources:``  <  CLI flag overrides

    Each layer key-merges over the previous, so you set only what differs — your
    standard fleet lives in the file and a config/flag tweaks a key. ``overrides``
    values that are ``None`` (an unset flag) are ignored so they don't clobber a
    lower layer.
    """
    defaults = load_resource_defaults()
    merged = dict(builtin or {})
    merged.update(defaults.get("all") or {})
    merged.update(defaults.get(tool) or {})
    merged.update(config_resources or {})
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return merged


def make_run_dir(name: str) -> Path:
    """Create ``~/.nova/runs/{timestamp}_{name}/`` and return the path."""
    run_dir = (
        nova_home() / "runs" / f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_{name}"
    )
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
# Pin the worker's Python: AMIs often ship an older system Python (e.g. 3.10),
# but supernova needs >=3.11. `uv venv --python` downloads a managed CPython if
# the requested version isn't already present. 3.12 has mature wheels across the
# full ML stack (torch, FlagEmbedding, fastembed, ...).
WORKER_PYTHON = "3.12"


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
    None for commands that only need base deps.

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
        f"uv venv --python {WORKER_PYTHON} {WORKER_VENV} && "
        f'uv pip install --python {WORKER_VENV}/bin/python "{spec}"'
    )


def worker_run(argv: str) -> str:
    """SkyPilot ``run`` command invoking the installed ``nova`` entrypoint."""
    return f"{WORKER_VENV}/bin/nova {argv}"


# --- Rust subcommand workers ------------------------------------------------
#
# storm/load are standalone Rust binaries, not Python. A worker just needs the
# binary on disk — no venv, no `nova` dispatcher, no Python extras. We ship a
# *prebuilt* binary (built once in CI, see .github/workflows/rust-binaries.yml)
# and the worker downloads it: `curl` in seconds, no Rust toolchain, no on-worker
# compile (which took minutes per fresh provision — the old `cargo install` path).

REPO_URL = "https://github.com/qdrant-labs/supernova"

# Where downloaded worker binaries land. Invoked by absolute path (no PATH/venv
# setup), so the `run` phase and `setup` phase agree without sourcing anything.
RUST_BIN_DIR = "$HOME/.nova/bin"


def rust_binary_url(binary: str) -> str:
    """Download URL for a prebuilt Rust worker binary (linux x86_64).

    Default: the GitHub Release asset matching the controller's version — the
    `rust-binaries.yml` workflow attaches `nova-load` / `nova-storm` to each `v*`
    release. Override with ``NOVA_RUST_BIN_URL`` to point at un-released code; a
    literal ``{binary}`` is replaced with the crate name. For a dev branch that's
    the rolling ``dev-<branch>`` pre-release the same workflow publishes on push::

        NOVA_RUST_BIN_URL='https://github.com/qdrant-labs/supernova/releases/download/dev-my-feature/{binary}'
    """
    override = os.environ.get("NOVA_RUST_BIN_URL")
    if override:
        return override.replace("{binary}", binary)
    return f"{REPO_URL}/releases/download/v{worker_version()}/{binary}"


def build_rust_worker_setup(binary: str) -> str:
    """SkyPilot ``setup`` for a Rust subcommand worker: download the prebuilt
    binary and mark it executable. No toolchain, no compile — seconds, not the
    minutes a from-source `cargo install` cost on every fresh provision (#16).

    ``-f`` makes curl fail loudly on a 404 (e.g. CI hasn't published this
    version/branch yet) instead of writing an error page to the binary path.
    """
    url = rust_binary_url(binary)
    dest = f"{RUST_BIN_DIR}/{binary}"
    return (
        f'mkdir -p "{RUST_BIN_DIR}" && '
        f"curl -fsSL '{url}' -o \"{dest}\" && chmod +x \"{dest}\""
    )


def rust_worker_run(binary: str, argv: str) -> str:
    """SkyPilot ``run`` command invoking the downloaded Rust binary by path."""
    return f'"{RUST_BIN_DIR}/{binary}" {argv}'


def resolve_binary(name: str) -> str:
    """Locate a nova subcommand binary on **this** machine: ``$NOVA_<NAME>_BIN``
    override, then PATH.

    Used controller-side when a Python orchestrator must invoke a Rust tool
    locally — e.g. ``load-dist`` running ``nova-load --setup-only`` for the
    one-time collection setup before it launches workers.
    """
    env = f"NOVA_{name.removeprefix('nova-').upper().replace('-', '_')}_BIN"
    override = os.environ.get(env)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(
        f"`{name}` not found on PATH (needed for distributed control-plane ops). "
        f"Build it (`cargo build --release -p {name} --features full`) and set {env} "
        f"to the binary, or `cargo install --git {REPO_URL} {name} --features full`."
    )


def config_mount(run_dir: Path, config_path: str) -> tuple[dict[str, str], str]:
    """
    Stage a config for a worker; return ``(file_mounts, remote_path)``.

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
