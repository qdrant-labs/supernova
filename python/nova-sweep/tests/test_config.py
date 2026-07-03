"""Unit tests for `nova_sweep.config`'s `target.type` dispatch: old configs
that omit `type` must keep resolving to the Qdrant backend (backward
compatibility), and an unknown `type` must fail loudly rather than silently
falling back.
"""

from __future__ import annotations

import pytest

from nova_sweep.backends.qdrant import QdrantTargetConfig
from nova_sweep.config import CorpusConfig, OutputConfig, QueriesConfig, SweepConfig


def _cfg(target: object) -> SweepConfig:
    return SweepConfig(
        corpus=CorpusConfig(path="/tmp/corpus"),
        queries=QueriesConfig(uri="/tmp/q.parquet", column="dense_embedding"),
        target=target,
        output=OutputConfig(path="/tmp/out"),
    )


def test_target_without_type_defaults_to_qdrant():
    cfg = _cfg({"url": "http://localhost:6334"})
    assert isinstance(cfg.target, QdrantTargetConfig)
    assert cfg.target.type == "qdrant"
    assert cfg.target.url == "http://localhost:6334"


def test_target_with_explicit_qdrant_type():
    cfg = _cfg({"type": "qdrant", "url": "http://localhost:6334", "recreate": "always"})
    assert isinstance(cfg.target, QdrantTargetConfig)
    assert cfg.target.recreate == "always"


def test_target_with_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown sweep target type"):
        _cfg({"type": "milvus", "url": "http://localhost:19530"})


def test_target_that_is_not_a_mapping_raises_a_clear_error():
    with pytest.raises(ValueError, match="`target:` must be a mapping"):
        _cfg(None)
    with pytest.raises(ValueError, match="`target:` must be a mapping"):
        _cfg("qdrant")


def test_target_with_explicit_null_type_defaults_to_qdrant():
    """YAML's `type:` (no value) and an omitted `type:` key both parse to
    `None`/absent respectively — both must default to qdrant, not just the
    omitted-key case."""
    cfg = _cfg({"type": None, "url": "http://localhost:6334"})
    assert isinstance(cfg.target, QdrantTargetConfig)
    assert cfg.target.type == "qdrant"


def test_target_with_non_string_type_raises_a_clear_error():
    with pytest.raises(ValueError, match="`target.type` must be a string"):
        _cfg({"type": [1, 2], "url": "http://localhost:6334"})
    with pytest.raises(ValueError, match="`target.type` must be a string"):
        _cfg({"type": 5, "url": "http://localhost:6334"})
