"""Run manifest: what a brute-force run was, what it ran on, and how it went.

A result parquet's schema metadata (see `results.provenance`) answers "what is
this ground truth?" — corpus, queries, metric, k, precision, tie-break. It
deliberately says nothing about the RUN: how many ranks, which corpus files
this worker took, how long it took, what it ran on, what it cost. Those are
properties of an execution, not of the rows, and they are what someone asks
months later when re-running, attributing cost, or explaining why one GT took
2 hours and another 12.

So each phase also writes one JSON manifest next to its output, in the same
shape nova-embed's `_manifest.json` uses (source / destination / created_at /
compute / settings / counts / timing / output_files) plus the nova-bf-specific
blocks — `searches`, `sharding`, `tiebreak` — that make a GT run reproducible.

Writing it is best-effort: a manifest is a record OF the outputs, so failing to
write one must never fail a run whose outputs already landed (a multi-hour
sharded GT run is not worth losing to an S3 hiccup on a 4 KB JSON PUT).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import sys
import subprocess

from datetime import datetime, timezone

from nova_bf.config import BruteForceConfig, SearchSpec
from nova_bf.io import ParquetFile, Store
from nova_bf.results import queries_stem

logger = logging.getLogger(__name__)

# Bump when a field's MEANING changes (not when one is added) — consumers that
# parse manifests across runs need to tell a rename from a new key.
MANIFEST_VERSION = 1


def manifest_name(
    cfg: BruteForceConfig,
    phase: str,
    job_rank: int | None = None,
    num_jobs: int | None = None,
) -> str:
    """Where this run's manifest goes, under `cfg.output.path`.

    Keyed by the queries stem for the same reason `result_name`/`partial_dir`
    are: several configs can share one output prefix, and two runs' manifests
    overwriting each other would be worse than having none. A sharded compute
    run writes one manifest PER RANK (each describes only its own slice, files,
    and timings), collected in a directory rather than scattered across the
    output root next to the result parquets.
    """
    stem = f"_bf_manifest_{queries_stem(cfg.queries.path)}_{phase}"
    if job_rank is None or num_jobs is None:
        return f"{stem}.json"
    width = max(3, len(str(num_jobs - 1)))
    return f"{stem}/rank{job_rank:0{width}d}.json"


def detect_compute(device: str | None = None) -> dict:
    """Best-effort hardware fingerprint: instance type, region, GPU, torch.

    So a run's output is self-describing for cost attribution and
    reproducibility — "31 ranks, 2h" means nothing without knowing they were
    g5.8xlarge. Mirrors nova-embed's `_detect_compute` (AWS IMDSv2 + torch), so
    the two toolsets' manifests describe hardware the same way.

    Returns `{}` off-cloud / without torch and never raises: short timeouts,
    every exception swallowed.
    """
    info: dict = {}
    try:
        import urllib.request

        tok = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(tok, timeout=0.5).read().decode()

        def _imds(path: str) -> str:
            req = urllib.request.Request(
                f"http://169.254.169.254/latest/meta-data/{path}",
                headers={"X-aws-ec2-metadata-token": token},
            )
            return urllib.request.urlopen(req, timeout=0.5).read().decode()

        info["instance_type"] = _imds("instance-type")
        info["region"] = _imds("placement/region")
        info["availability_zone"] = _imds("placement/availability-zone")
    except Exception:
        pass
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            # The denominator for this run's peak usage (see `gpu_peak`): a
            # peak is only actionable next to the card it had to fit in.
            info["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        pass
    if device is not None:
        # What scoring ACTUALLY ran on. Not redundant with `gpu`: a CUDA box
        # whose driver failed at import time still lists a GPU here while
        # having scored on the CPU, and that gap explains an odd wall time.
        info["device"] = device
    return info


def _workspace_version(start: str) -> str | None:
    """The supernova workspace version from the root `Cargo.toml`, if this is a
    checkout rather than an installed wheel.

    nova-bf's own `[project] version` is a static `0.0.1` nobody bumps, so the
    workspace version is the only release-shaped number the toolset has — and
    it is what the git tags carry (`v0.0.12`), so it lines up with
    `git_describe` below. Walks up from the package directory; `None` when
    there is no Cargo.toml above it.
    """
    import tomllib

    path = os.path.abspath(start)
    for _ in range(8):  # src/nova_bf → … → repo root is 4 up; 8 is slack
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
        # `[workspace] version` today; `[workspace.package] version` is the
        # more common spelling, so accept either rather than break on a move.
        # A crate's own Cargo.toml carries neither: keep walking up rather than
        # concluding there is no workspace above it.
        version = ws.get("version") or ws.get("package", {}).get("version")
        if version:
            return version
    return None


def kernel_usage(prune_launches: int) -> dict:
    """Which GPU fast paths actually RAN, not which ones were permitted.

    Each entry carries:

      permitted    the kill switch, i.e. what was allowed
      launches     how many times it actually ran
      unavailable  why it cannot run, or None
    """
    from nova_bf import merge_triton, topk_triton

    return {
        "prune": {
            "permitted": not os.environ.get("NOVA_BF_NO_PRUNE"),
            # Prune has no kernel and cannot decline at runtime; this counts the
            # slices where a threshold was actually applied.
            "launches": prune_launches,
            "unavailable": None,
        },
        "fold_kernel": merge_triton.usage(),
        "topk_kernel": topk_triton.usage(),
    }


def code_versions() -> dict:
    """WHICH CODE produced this ground truth: the supernova version and git
    commit, plus the versions of the libraries whose numerics it depends on.

    The scoring and tie-break kernels are the ground truth — a change to the
    top-K merge or the tie-break packing changes which of two equal-scoring
    docs wins, and nothing in a result file records the revision that decided
    it. The package version alone cannot: it is a static `0.0.1` that nobody
    bumps per commit, so the commit sha is the only real answer.

    `git_dirty` is scoped to the nova-bf package directory, not the whole
    repo — an unrelated edit elsewhere in the monorepo says nothing about the
    code that ran, while an uncommitted edit HERE means the sha above is a
    lie by itself.

    Everything is best-effort and returns partial: a wheel install is not a git
    checkout, and git may not exist on the box at all.
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

        info["nova_bf_version"] = version("nova-bf")
    except Exception:
        pass
    info["python_version"] = platform.python_version()
    for mod in ("numpy", "pyarrow"):
        try:
            info[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            pass
    try:
        def _git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", pkg_dir, *args],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()

        # Only trust git if the repository it answers about actually CONTAINS
        # this package.
        toplevel = os.path.realpath(_git("rev-parse", "--show-toplevel"))
        if os.path.commonpath([toplevel, os.path.realpath(pkg_dir)]) != toplevel:
            raise RuntimeError("git toplevel does not contain the nova-bf package")

        info["git_commit"] = _git("rev-parse", "HEAD")
        # e.g. "v0.0.12-9-g659c42c-dirty" — the release-relative form of the
        # sha above, and the line a human reads first.
        info["git_describe"] = _git("describe", "--tags", "--always", "--dirty")
        # Fleet jobs deploy a branch (`@master`), so the branch is what someone
        # would have to check out to reproduce this.
        info["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        info["git_dirty"] = bool(_git("status", "--porcelain", "-uno", "--", pkg_dir))
    except Exception:
        pass  # not a checkout, or no git — the versions above still stand
    return info


def job_identity() -> dict:
    """Which launch and which host produced this manifest.

    A 64-rank run leaves 64 manifests, and the reason to open one is that
    something about that rank looks wrong — at which point you need its log,
    which lives under the SkyPilot task/cluster id. Without these the manifest
    is an orphan: it says a rank did something odd, and nothing says where to
    go read about it.
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


def corpus_fingerprint(files: list[ParquetFile]) -> dict:
    """A hash of the corpus file list, IN THE ORDER the run indexed it.

    This is an id-scheme fingerprint, not a data checksum. With no
    `corpus.id_column`, hit ids are `make_point_id(file_key, row)` and the
    global file index also drives the ordinal tie-break — so the ORDER of this
    list is load-bearing. Add, remove or rename one corpus file and every later
    file's index shifts: the ground truth silently stops joining to a
    collection loaded from the other list, and recall reads near zero with
    nothing in either artifact to explain it. Comparing this hash is the cheap
    way to prove two runs saw the same corpus, and it catches
    `corpus.include`/`exclude` drift between ranks for free.

    Hashed over the loader `key` (the s3 object key / absolute local path),
    because that is exactly what `make_point_id` consumes — not `read_path`,
    which carries the bucket on s3 and would differ for the same logical
    corpus. `first`/`last` are carried in the clear so a human can eyeball
    which corpus this was without resolving a hash.
    """
    keys = [f.key for f in files]
    return {
        "files": len(keys),
        "sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "first": keys[0] if keys else None,
        "last": keys[-1] if keys else None,
    }


def search_entry(spec: SearchSpec) -> dict:
    """The nova-bf-specific core of a manifest: what this search actually was.

    `filter` is dumped in full (not just a `filtered: true` flag, which is all
    the parquet metadata can afford) because the filter IS the search for a
    filtered GT — recall numbers from two runs are comparable only if their
    predicates were identical, and a YAML edit between runs is otherwise
    invisible in the artifacts.
    """
    return {
        "name": spec.name,
        "vector_type": spec.vector_type,
        "metric": spec.metric,
        "k": spec.k,
        "filter": (
            None if spec.filter is None
            else spec.filter.model_dump(mode="json", exclude_defaults=True)
        ),
        "rows": (
            None if spec.rows is None
            else spec.rows.model_dump(mode="json")
        ),
    }


def source_block(cfg: BruteForceConfig) -> dict:
    """Corpus + queries: the inputs, and the columns actually read from them."""
    return {
        "corpus": {
            "path": cfg.corpus.path,
            "include": cfg.corpus.include,
            "exclude": cfg.corpus.exclude,
            "id_column": cfg.corpus.id_column,
            "dense_column": cfg.corpus.dense_column,
            "sparse_column": cfg.corpus.sparse_column,
            "multivector_column": cfg.corpus.multivector_column,
        },
        "queries": {
            "path": cfg.queries.path,
            "id_column": cfg.queries.id_column,
            "payload_fields": list(cfg.queries.payload_fields),
            "dense_column": cfg.queries.dense_column,
            "sparse_column": cfg.queries.sparse_column,
            "multivector_column": cfg.queries.multivector_column,
        },
    }


def base_manifest(cfg: BruteForceConfig, phase: str, device: str | None = None) -> dict:
    """The fields every phase's manifest carries, in a stable key order."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source_block(cfg),
        "destination": cfg.output.path,
        "compute": detect_compute(device),
        "code": code_versions(),
        "job": job_identity(),
        # Run-level knobs, resolved. CLI overrides (`--io-workers` etc.) are
        # merged in by the caller, so this is what RAN, not what the YAML said.
        "params": cfg.params.model_dump(mode="json"),
        # Which tie-break rule decided exact ties — the same value stamped into
        # the parquet metadata, repeated here so a manifest stands alone.
        "tiebreak": cfg.params.tiebreak,
    }


def gpu_peak(device: str | None) -> dict:
    """How much GPU memory this run actually needed, at its high-water mark.

    Sizing a fleet is the recurring question here — `dense_batch_size`,
    `multivector_token_budget` and the query matrix all trade against one
    card's memory, and the failure mode is an OOM hours into a run on an
    instance one size too small. The logs say what was configured; only this
    says what it cost. `allocated` is what nova-bf's tensors held, `reserved`
    is what the caching allocator took from the driver — reserved is the one
    to compare against `gpu_total_bytes`, since that is what actually has to
    fit.

    `{}` on CPU or without torch; never raises.
    """
    if device != "cuda":
        return {}
    try:
        import torch

        return {
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:  # noqa: BLE001
        return {}


def host_peak() -> dict:
    """Peak resident set size for this process, in bytes.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB here, macOS reports bytes. Normalize so the field
        # is comparable with `peak_gpu_allocated_bytes` and cannot be misread
        # by a factor of 1024.
        scale = 1 if sys.platform == "darwin" else 1024
        return {"peak_host_rss_bytes": int(peak) * scale}
    except Exception:  # noqa: BLE001
        return {}


def write(store: Store, filename: str, payload: dict) -> str | None:
    """Write `payload` as JSON to `store/filename`. Never raises.

    Returns the path written, or None if writing failed (logged as a warning,
    not an error: the run's real outputs are already on disk by the time this
    is called, and losing the manifest must not fail them).
    """
    try:
        # default=str so an unexpected value (a date in a filter dump, say)
        # degrades to its string form instead of sinking the whole manifest.
        data = json.dumps(payload, indent=2, default=str).encode()
        path = store.write_bytes(filename, data)
        logger.info("wrote run manifest %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - a manifest must never fail a run
        logger.warning("could not write run manifest %s: %s", filename, exc)
        return None
