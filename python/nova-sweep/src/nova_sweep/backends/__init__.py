"""Sweep target backend registry, dispatched on `target.type` — mirrors the
tagged-enum shape of `nova-load`'s `VectorStoreConfig` and `nova-storm`'s
`TargetConfig`. Qdrant, Milvus, and Elasticsearch are registered; a future
backend means adding a module like `qdrant.py` and a line in `_REGISTRY`, not
touching `runner.py`.
"""

from __future__ import annotations

from nova_sweep.backends.base import SweepBackend, TargetConfigBase
from nova_sweep.backends.elastic import ElasticBackend, ElasticTargetConfig
from nova_sweep.backends.milvus import MilvusBackend, MilvusTargetConfig
from nova_sweep.backends.qdrant import QdrantBackend, QdrantTargetConfig

_REGISTRY: dict[str, tuple[type[TargetConfigBase], SweepBackend]] = {
    "qdrant": (QdrantTargetConfig, QdrantBackend()),
    "milvus": (MilvusTargetConfig, MilvusBackend()),
    "elastic": (ElasticTargetConfig, ElasticBackend()),
}


def _lookup(backend_type: str) -> tuple[type[TargetConfigBase], SweepBackend]:
    entry = _REGISTRY.get(backend_type)
    if entry is None:
        raise ValueError(
            f"unknown sweep target type '{backend_type}'; available: {sorted(_REGISTRY)}"
        )
    return entry


def parse_target(raw: object) -> TargetConfigBase:
    """Dispatch a raw `target:` mapping to its backend's config model, on
    `type:` — required; omitting it (or setting it to YAML `null`) is an
    error rather than an implicit default."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"`target:` must be a mapping (e.g. `target: {{url: ...}}`), got {raw!r}"
        )
    data = dict(raw)
    backend_type = data.get("type")
    if backend_type is None:
        raise ValueError(f"`target.type` is required; available: {sorted(_REGISTRY)}")
    if not isinstance(backend_type, str):
        raise ValueError(f"`target.type` must be a string, got {backend_type!r}")
    data["type"] = backend_type
    config_cls, _ = _lookup(backend_type)
    return config_cls.model_validate(data)


def get_backend(backend_type: str) -> SweepBackend:
    _, backend = _lookup(backend_type)
    return backend


__all__ = ["TargetConfigBase", "SweepBackend", "parse_target", "get_backend"]
