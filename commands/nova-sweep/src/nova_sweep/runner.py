"""Per-slice execution, where a slice is a combination of data layout,
index variant, and search parameters. For each `Slice`, resolve the
insert-vs-reuse-vs-error decision, walk `index_variants` via `nova-load
reindex`, walk `searches` via `nova-storm --json`, and accumulate report rows.

Backend-neutral: every store-specific detail (collection existence, generated
`nova-load`/`nova-storm` configs) is delegated to a `SweepBackend` looked up
by `cfg.target.type` (see `nova_sweep.backends`) — this module has no
knowledge of Qdrant or any other concrete backend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

from pathlib import Path

import click
import yaml

from nova_sweep.backends import get_backend
from nova_sweep.config import SweepConfig
from nova_sweep.report import build_row
from nova_sweep.slices import Slice

log = logging.getLogger("nova_sweep")

_REINDEX_TIMING_RE = re.compile(r"effective_seconds=(\d+(?:\.\d+)?)")


class SweepAbort(Exception):
    """A step-1 collection collision — aborts the whole run,
    unlike a mid-slice subprocess failure, which only aborts
    that slice's remaining points."""


def _resolve_binary(name: str) -> str:
    """`$NOVA_<NAME>_BIN`, then PATH — same override convention as `nova-dist`."""
    env = f"NOVA_{name.removeprefix('nova-').upper().replace('-', '_')}_BIN"
    return os.environ.get(env) or shutil.which(name) or name


def _run(binary: str, args: list[str]) -> tuple[bool, str, str]:
    """Run a tool subprocess to completion. Never raises on a non-zero exit —
    the caller decides whether that's an abort or a recorded
    error row. Returns `(ok, stdout, stderr)`.
    """
    exe = _resolve_binary(binary)
    log.info("run: %s %s", exe, " ".join(args))
    proc = subprocess.run([exe, *args], capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()


def _resolve_reindex_seconds(*, stdout: str, stderr: str, fallback_seconds: float) -> float:
    """Use nova-load's internally-reported effective completion time when present.

    Wall-clock includes the required green-hold validation window, but the
    reported sweep metric should stop at the start of the hold window that
    eventually proved stable.
    """
    for text in (stderr, stdout):
        match = _REINDEX_TIMING_RE.search(text)
        if match:
            return float(match.group(1))
    return fallback_seconds


def _write_yaml(data: dict, tmpdir: Path, name: str) -> str:
    path = tmpdir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def collection_exists(cfg: SweepConfig, collection_name: str) -> bool:
    """Thin wrapper over the backend's own check — kept as a module-level
    function since `cli.py`'s `--dry-run` calls it directly."""
    return get_backend(cfg.target.type).collection_exists(cfg, collection_name)


def _resolve_insert_action(cfg: SweepConfig, slc: Slice, exists: bool, skip_insert: bool) -> str:
    """Returns `"recreate"` | `"load"` | `"skip"` — never raises itself; the
    collision case is signaled by returning `None` so the caller can build a
    full error message with slice context."""
    if cfg.target.recreate == "always":
        return "recreate"
    if exists and not skip_insert:
        return None
    if exists:
        return "skip"
    return "load"


def run_sweep(cfg: SweepConfig, slices: list[Slice], *, skip_insert: bool, cleanup: bool) -> list[dict]:
    backend = get_backend(cfg.target.type)
    rows: list[dict] = []
    inserted_collections: list[str] = []

    for slc in slices:
        exists = backend.collection_exists(cfg, slc.collection_name)
        action = _resolve_insert_action(cfg, slc, exists, skip_insert)
        if action is None:
            raise SweepAbort(
                f"collection '{slc.collection_name}' (data_layout "
                f"'{slc.data_layout_name}') already exists. Pass --skip-insert "
                "to reuse it as-is, or set `target.recreate: always` in the "
                "config to force a fresh reload."
            )

        click.echo(f"[{slc.data_layout_name}] collection={slc.collection_name} action={action}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            if action != "skip":
                load_cfg = backend.build_load_config(cfg, slc, recreate=(action == "recreate"))
                cfg_path = _write_yaml(load_cfg, tmp_path, "load")
                ok, stdout, stderr = _run("nova-load", ["run", cfg_path])
                output = stderr or stdout
                inserted_collections.append(slc.collection_name)
                if not ok:
                    log.warning("[%s] insert failed: %s", slc.data_layout_name, output)
                    rows.extend(
                        build_row(
                            data_layout=slc.data_layout, data_layout_name=slc.data_layout_name,
                            collection_name=slc.collection_name, index_variant=variant, search=search,
                            summary=None, reindex_seconds=0.0, search_seconds=0.0,
                            ok=False, error=f"insert failed: {output}",
                        )
                        for variant in slc.index_variants
                        for search in slc.searches
                    )
                    continue  # next slice — nothing under a failed insert can be measured

            for variant in slc.index_variants:
                reindex_cfg = backend.build_reindex_config(cfg, slc, variant)
                cfg_path = _write_yaml(reindex_cfg, tmp_path, f"reindex_{variant['_name']}")
                t0 = time.monotonic()
                ok, stdout, stderr = _run("nova-load", ["reindex", cfg_path])
                wall_seconds = time.monotonic() - t0
                reindex_seconds = _resolve_reindex_seconds(
                    stdout=stdout, stderr=stderr, fallback_seconds=wall_seconds
                )
                output = stderr or stdout
                if not ok:
                    log.warning(
                        "[%s/%s] reindex failed: %s", slc.data_layout_name, variant["_name"], output
                    )
                    rows.extend(
                        build_row(
                            data_layout=slc.data_layout, data_layout_name=slc.data_layout_name,
                            collection_name=slc.collection_name, index_variant=variant, search=search,
                            summary=None, reindex_seconds=reindex_seconds, search_seconds=0.0,
                            ok=False, error=f"reindex failed: {output}",
                        )
                        for search in slc.searches
                    )
                    continue  # next index_variant

                for search in slc.searches:
                    storm_cfg = backend.build_storm_config(cfg, slc, search)
                    cfg_path = _write_yaml(
                        storm_cfg, tmp_path, f"storm_{variant['_name']}_{search['_name']}"
                    )
                    t0 = time.monotonic()
                    ok, stdout, stderr = _run("nova-storm", [cfg_path, "--json"])
                    search_seconds = time.monotonic() - t0

                    summary, error = None, None
                    if ok:
                        try:
                            summary = json.loads(stdout)
                        except json.JSONDecodeError as e:
                            ok, error = False, f"failed to parse nova-storm --json output: {e}"
                    else:
                        output = stderr or stdout
                        error = f"storm failed: {output}"

                    rows.append(
                        build_row(
                            data_layout=slc.data_layout, data_layout_name=slc.data_layout_name,
                            collection_name=slc.collection_name, index_variant=variant, search=search,
                            summary=summary, reindex_seconds=reindex_seconds,
                            search_seconds=search_seconds, ok=ok, error=error,
                        )
                    )

    if cleanup:
        # Only collections THIS run inserted into — never a `--skip-insert`-reused
        # one, which pre-existed and wasn't this run's to delete.
        for name in inserted_collections:
            delete_cfg = backend.build_delete_config(cfg, name)
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg_path = _write_yaml(delete_cfg, Path(tmpdir), "delete")
                ok, stdout, stderr = _run("nova-load", ["delete", cfg_path])
                output = stderr or stdout
                if not ok:
                    log.warning("cleanup: failed to delete '%s': %s", name, output)

    return rows
