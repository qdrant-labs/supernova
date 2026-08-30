"""Run-identity metadata for the embed manifest: code, host, model revisions.

The manifest (see `embedders/runner.py`) already records what the pipeline was
CONFIGURED to do. These are the facts it cannot get from the config — which
build of the code ran, which machine ran it, and which revision of a model's
weights the vectors actually came from.

Everything here is best-effort and returns partial or empty: a manifest is a
record OF a run that already happened, so nothing in it may raise.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess

logger = logging.getLogger(__name__)


def _workspace_version(start: str) -> str | None:
    """The supernova workspace version from the root `Cargo.toml`, when this is
    a checkout rather than an installed wheel.

    nova-embed's own `[project] version` is a static nobody bumps, so the
    workspace version is the only release-shaped number the toolset has — and
    the git tags carry it, so it lines up with `git_describe`.
    """
    import tomllib

    path = os.path.abspath(start)
    for _ in range(8):  # src/nova_embed → … → repo root is 4 up; 8 is slack
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
        candidate = os.path.join(path, "Cargo.toml")
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "rb") as fh:
                ws = tomllib.load(fh).get("workspace", {})
        except Exception:  # noqa: BLE001 - an unreadable Cargo.toml is not fatal
            continue
        # A crate's own Cargo.toml carries no `[workspace]`; keep walking up
        # rather than concluding there is no workspace above it.
        version = ws.get("version") or ws.get("package", {}).get("version")
        if version:
            return version
    return None


def code_versions() -> dict:
    """WHICH CODE produced these embeddings: supernova version, git revision,
    and the versions of the libraries that decide what a vector IS.

    The library versions are not boilerplate here the way they might be
    elsewhere: the model runtime is the model's behaviour. A transformers or
    sentence-transformers major bump can change tokenization, pooling defaults
    or break a backend's path outright, and the output parquet looks identical
    either way. When a collection built from these vectors later behaves oddly,
    this block is the difference between bisecting and guessing.

    `git_dirty` is scoped to the nova-embed package directory, not the whole
    monorepo — an unrelated edit elsewhere says nothing about the code that ran,
    while an uncommitted edit HERE means the sha alone is a lie.
    """
    info: dict = {}
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        supernova = _workspace_version(pkg_dir)
        if supernova:
            info["supernova_version"] = supernova
    except Exception:  # noqa: BLE001
        pass
    try:
        from importlib.metadata import version

        info["nova_embed_version"] = version("nova-embed")
    except Exception:
        pass
    info["python_version"] = platform.python_version()
    # Only the ones actually loaded: a run that never imported vllm should not
    # claim a vllm version, and importing it here to find out would cost
    # seconds and GPU memory for a field nobody asked for.
    for mod in ("torch", "transformers", "sentence_transformers", "vllm", "fastembed", "pyarrow"):
        try:
            import sys

            loaded = sys.modules.get(mod)
            if loaded is not None and getattr(loaded, "__version__", None):
                info[f"{mod}_version"] = loaded.__version__
        except Exception:
            pass
    try:
        def _git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", pkg_dir, *args],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()

        info["git_commit"] = _git("rev-parse", "HEAD")
        # e.g. "v0.0.12-9-g659c42c-dirty" — the release-relative form of the
        # sha above, and the line a human reads first.
        info["git_describe"] = _git("describe", "--tags", "--always", "--dirty")
        # Fleet jobs deploy a branch (`@master`), so the branch is what someone
        # would check out to reproduce this.
        info["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        info["git_dirty"] = bool(_git("status", "--porcelain", "-uno", "--", pkg_dir))
    except Exception:
        pass  # not a checkout, or no git — the versions above still stand
    return info


def job_identity() -> dict:
    """Which launch and which host produced this manifest.

    A 50-rank embed run leaves 50 manifests, and the reason to open one is that
    something about that rank looks wrong — at which point you need its log,
    which lives under the SkyPilot task/cluster id.
    """
    info: dict = {}
    for var in (
        "SKYPILOT_TASK_ID",
        "SKYPILOT_CLUSTER_NAME",
        "SKYPILOT_JOB_ID",
        "SKYPILOT_JOB_RANK",
        "SKYPILOT_NODE_RANK",
        "SKYPILOT_NUM_NODES",
    ):
        value = os.environ.get(var)
        if value:  # absent off-SkyPilot; a missing key beats a null
            info[var.lower()] = value
    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        pass
    info["pid"] = os.getpid()
    return info


# Config keys whose VALUE is a credential. `backend_kwargs` is "every unknown
# key in the entry", and some backends take secrets that way — `openai` accepts
# `api_key` exactly like it accepts `batch_size`. The manifest is uploaded next
# to the embeddings, so recording those verbatim would publish the key to the
# bucket. Matched on the key name, over-inclusively: redacting a harmless
# `cache_key` costs a provenance field nobody reads; leaking one real key is
# a credential rotation.
_SECRET_EXACT = frozenset({"api_key", "token", "password", "secret", "credentials"})
_SECRET_SUFFIXES = ("_key", "_token", "_secret", "_password")
REDACTED = "[redacted]"


def redact(kwargs: dict | None) -> dict | None:
    """`kwargs` with every credential-shaped value replaced by `[redacted]`.

    The KEY is kept: "this run passed an api_key" is useful provenance, and
    losing it would hide that the entry was configured differently from one
    that took its credential from the environment. Recurses into nested dicts
    (vLLM's engine kwargs) and leaves `None` alone — an absent secret is not
    a secret.
    """
    if not kwargs:
        return kwargs
    out: dict = {}
    for key, value in kwargs.items():
        lowered = str(key).lower()
        if isinstance(value, dict):
            out[key] = redact(value)
        elif value is not None and (
            lowered in _SECRET_EXACT or lowered.endswith(_SECRET_SUFFIXES)
        ):
            out[key] = REDACTED
        else:
            out[key] = value
    return out


def hf_revision(model: str | None, revision: str | None = None) -> str | None:
    """The commit sha of the model repo these weights were loaded from.

    `model` is a bare Hub id like `Alibaba-NLP/gte-multilingual-base`, which is
    a MOVING reference: the repo owner can push new weights, a new tokenizer or
    a changed pooling config, and every later run silently embeds into a
    different space while the config, the manifest and the parquet all stay
    byte-identical. The sha is what makes an embedding run reproducible, and
    what a query-side embedder has to match to search the collection built
    here.

    Read from the local Hub cache (`snapshots/<sha>`) rather than the network:
    by the time this is called the weights are loaded, so the cache is warm and
    the answer is what was USED — not what the Hub happens to serve now, which
    is the very thing that can have moved. Returns None for a local path, a
    non-Hub backend (an API model has no revision), or a cold/absent cache.

    `revision` is the entry's PIN (sha, tag or branch) and must be passed when
    the config has one: resolving without it asks the cache for `main`, which
    is a different snapshot than the pinned one the run actually loaded — so
    the manifest would confidently name the wrong weights, or find no cached
    `main` at all and report nothing. Either way it inverts the point of the
    field.
    """
    if not model or os.path.exists(model):  # a local directory is not a Hub repo
        return None
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=model, revision=revision, local_files_only=True
        )
        sha = os.path.basename(os.path.normpath(path))
        # The cache lays snapshots out as `<repo>/snapshots/<40-hex sha>`; a
        # branch-named symlink or an unexpected layout is not a revision.
        return sha if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha) else None
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a run
        logger.debug("could not resolve the Hub revision for %s: %s", model, exc)
        return None
