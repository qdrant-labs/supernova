"""Unit tests for `nova_sweep.runner`'s pure, backend-neutral pieces: the
insert-action decision and reindex timing extraction. No subprocess, no live
store. Generated `nova-load`/`nova-storm` config shapes are backend-specific
and covered in `test_qdrant_backend.py` instead — the runner itself only
calls through `SweepBackend`, it doesn't build configs.
"""

from __future__ import annotations

from nova_sweep.config import CorpusConfig, OutputConfig, QueriesConfig, SweepConfig
from nova_sweep.runner import _resolve_reindex_seconds, _resolve_insert_action
from nova_sweep.slices import build_slices


def _cfg(recreate="never", **axes) -> SweepConfig:
    return SweepConfig(
        corpus=CorpusConfig(path="/tmp/corpus", dense_column="dense_embedding"),
        queries=QueriesConfig(
            uri="/tmp/q.parquet", column="dense_embedding", ground_truth_column="hit_ids"
        ),
        target={"url": "http://localhost:6334", "recreate": recreate},
        output=OutputConfig(path="/tmp/out"),
        **axes,
    )


def _one_slice(cfg: SweepConfig):
    return build_slices(cfg, "mysweep")[0]


# --- insert-action resolution -----------------------------------------------


def test_recreate_always_wins_regardless_of_existence_or_flag():
    cfg = _cfg(recreate="always")
    slc = _one_slice(cfg)
    assert _resolve_insert_action(cfg, slc, exists=True, skip_insert=False) == "recreate"
    assert _resolve_insert_action(cfg, slc, exists=False, skip_insert=False) == "recreate"


def test_existing_collection_without_skip_insert_is_a_collision():
    cfg = _cfg(recreate="never")
    slc = _one_slice(cfg)
    assert _resolve_insert_action(cfg, slc, exists=True, skip_insert=False) is None


def test_existing_collection_with_skip_insert_is_skipped():
    cfg = _cfg(recreate="never")
    slc = _one_slice(cfg)
    assert _resolve_insert_action(cfg, slc, exists=True, skip_insert=True) == "skip"


def test_new_collection_loads_regardless_of_skip_insert():
    cfg = _cfg(recreate="never")
    slc = _one_slice(cfg)
    assert _resolve_insert_action(cfg, slc, exists=False, skip_insert=False) == "load"
    assert _resolve_insert_action(cfg, slc, exists=False, skip_insert=True) == "load"


def test_reindex_seconds_uses_effective_timing_from_logs():
    seconds = _resolve_reindex_seconds(
        stdout="",
        stderr="2026-07-03T00:00:00Z INFO nova_load: reindex timing: effective_seconds=12.345",
        fallback_seconds=17.0,
    )
    assert seconds == 12.345


def test_reindex_seconds_falls_back_to_wall_clock_without_marker():
    seconds = _resolve_reindex_seconds(stdout="", stderr="plain log output", fallback_seconds=17.0)
    assert seconds == 17.0
