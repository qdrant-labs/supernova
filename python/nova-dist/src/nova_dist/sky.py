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
    # Forward AWS creds for cross-cloud runs (e.g. GCP workers reading S3), plus
    # vars referenced by the workload config and tool-specific extras.
    wanted = AWS_ENV_VARS + referenced_env_vars(config_path) + (extra or [])
    return {v: os.environ[v] for v in dict.fromkeys(wanted) if os.environ.get(v)}


_REPO = "https://github.com/qdrant-labs/supernova"


def _rust_worker_setup(binary: str) -> str:
    """
    Install a prebuilt Rust binary on a worker (seconds), falling back to a
    source compile if no release asset matches the arch (e.g. before the first
    release). Installs to `/usr/local/bin` so it's on `PATH` in both SkyPilot's
    `setup` and `run` shells (they're separate).
    """
    return (
        "set -e\n"
        'case "$(uname -m)" in\n'
        "  x86_64) t=x86_64-unknown-linux-gnu ;;\n"
        "  aarch64|arm64) t=aarch64-unknown-linux-gnu ;;\n"
        "  *) t= ;;\n"
        "esac\n"
        f'if [ -n "$t" ] && curl -fsSL "{_REPO}/releases/latest/download/{binary}-$t" -o /tmp/{binary}; then\n'
        f"  sudo install -m 0755 /tmp/{binary} /usr/local/bin/{binary}\n"
        "else\n"
        f'  echo "no prebuilt {binary} for $(uname -m) / latest release — building from source"\n'
        "  command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\n"
        '  . "$HOME/.cargo/env"\n'
        f"  cargo install --git {_REPO} {binary}\n"
        f'  sudo install -m 0755 "$HOME/.cargo/bin/{binary}" /usr/local/bin/{binary}\n'
        "fi"
    )


def _python_worker_setup(binary: str, pip_spec: str) -> str:
    """
    Install a Python tool (`binary`, from `pip_spec`) on a worker. Works on BOTH
    a root CUDA container and a non-root GPU VM:
      - `set -e` so failures abort setup loudly (visible in setup logs) instead of
        surfacing later as `<binary>: command not found`;
      - `sudo` only when not root;
      - apt-get curl/git if missing (a minimal CUDA container lacks both; a real
        VM AMI has them, so this no-ops there);
      - install the console script to ~/.local/bin (pinned via UV_TOOL_BIN_DIR),
        then symlink into /usr/local/bin — which IS on PATH in the separate run
        shell (an `export PATH` here would not be).
    """
    return (
        "set -e\n"
        'SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"\n'
        # ffmpeg: newer `datasets` (pulled in by sentence-transformers) imports
        # torchcodec, which dlopens libavutil/libavcodec at import time — a GPU
        # AMI without FFmpeg makes the whole ST backend unimportable.
        "command -v curl >/dev/null && command -v git >/dev/null && command -v ffmpeg >/dev/null || "
        "($SUDO apt-get update && $SUDO apt-get install -y curl git ffmpeg)\n"
        "curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        # --python pins the tool env: unpinned, uv grabs the newest CPython
        # (3.14+), where missing wheels/support markers make the resolver
        # backtrack to prehistoric transitive versions that crash at import.
        f"UV_TOOL_BIN_DIR=\"$HOME/.local/bin\" uv tool install --python 3.12 '{pip_spec}'\n"
        f'$SUDO ln -sf "$HOME/.local/bin/{binary}" /usr/local/bin/{binary}'
    )


# Built-in fallbacks per tool: resources + the worker-install `setup` + envs. Used
# when neither --resources nor ~/.nova/skypilot/<tool>.yaml is present, so a
# first-time `nova dist <tool>` works with zero extra files. An override (file or
# flag) is shallow-merged OVER these by top-level key — set only what you want to
# change (e.g. just `setup:` for a dev build keeps the default `resources:`).
# These mirror the copy-and-tweak examples in configs/skypilot/.
#
# GPU image: a SkyPilot catalog GPU AMI alias, NOT `docker:nvidia/cuda`. The
# docker image looked appealing (version/region-agnostic) but the GPU is not
# visible inside the container in practice — torch falls back to CPU. This AMI is
# a real GPU VM with the driver + CUDA preinstalled (torch.cuda works). It is
# catalog-version-specific: if your SkyPilot errors on the tag, grep
# ~/.sky/catalogs/*/aws/images.csv for the current gpu tag (and override per-tool
# in ~/.nova/skypilot/<tool>.yaml).
_GPU_IMAGE = "skypilot:custom-gpu-ubuntu-cuda13"

DEFAULTS: dict[str, dict] = {
    "embed": {
        "resources": {
            "cloud": "aws",
            "accelerators": "A10G:1",
            "use_spot": False,
            "disk_size": 150,
            "image_id": _GPU_IMAGE,
        },
        "setup": _python_worker_setup(
            "nova-embed",
            f"nova-embed[embed] @ git+{_REPO}@master#subdirectory=python/nova-embed",
        ),
        "envs": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    },
    "load": {
        "resources": {"cloud": "aws", "cpus": "8+", "use_spot": False, "disk_size": 100},
        "setup": _rust_worker_setup("nova-load"),
    },
    "storm": {
        "resources": {"cloud": "aws", "cpus": "4+", "use_spot": False},
        "setup": _rust_worker_setup("nova-storm"),
    },
    "bf": {
        # GPU brute force on the same real-GPU AMI as embed (see _GPU_IMAGE note).
        "resources": {
            "cloud": "aws",
            "accelerators": "A10G:1",
            "use_spot": False,
            "disk_size": 150,
            "image_id": _GPU_IMAGE,
        },
        "setup": _python_worker_setup(
            "nova-bf",
            f"nova-bf[compute] @ git+{_REPO}@master#subdirectory=python/nova-bf",
        ),
    },
}

# Top-level keys an override may replace. `run:` is intentionally not here — we
# always inject the per-rank command.
#
# `file_mounts` lets an override stage extra local files onto workers (e.g. a
# locally-built binary, to skip an in-`setup` compile) — merged under the
# run's own mounts (staged config, `--catalog`), which win on path collision.
_MERGE_KEYS = ("resources", "setup", "envs", "file_mounts")


def skypilot_dir() -> Path:
    """
    Where per-tool default SkyPilot YAMLs live. `$NOVA_SKYPILOT_DIR` overrides;
    defaults to `$NOVA_HOME/skypilot`.
    """
    override = os.environ.get("NOVA_SKYPILOT_DIR")
    return Path(override) if override else nova_home() / "skypilot"


def resolve_resources(tool: str, flag_path: str | None) -> tuple[dict, str]:
    """
    Resolve the SkyPilot spec for `tool`, layering low → high:

        built-in DEFAULTS  <  ~/.nova/skypilot/<tool>.yaml  <  --resources flag

    Only the top-level keys present in an override (`resources`/`setup`/`envs`/
    `file_mounts`) replace the default's value for that key — so a file with
    just `setup:` keeps the default `resources:`, and vice versa. Returns
    `(spec, source)` where `source` describes where the override came from (for
    logging).
    """
    if tool not in DEFAULTS:
        raise ValueError(f"no built-in defaults for tool {tool!r}")
    spec = {k: v for k, v in DEFAULTS[tool].items()}  # shallow copy of top level

    if flag_path:
        override_path: Path | None = Path(flag_path)
        if not override_path.exists():  # an explicit flag must exist
            raise FileNotFoundError(f"--resources file not found: {flag_path}")
        source = str(override_path)
    else:
        candidate = skypilot_dir() / f"{tool}.yaml"
        override_path = candidate if candidate.exists() else None
        source = str(candidate) if override_path else "built-in defaults"

    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        for k in _MERGE_KEYS:
            if k in override:
                spec[k] = override[k]
    return spec, source


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
    merged_mounts = {**sky_cfg.get("file_mounts", {}), **file_mounts}
    pool_yaml = {
        "pool": {"min_workers": num_jobs, "max_workers": num_jobs},
        "resources": resources,
        "file_mounts": merged_mounts,
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
