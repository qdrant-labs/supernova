"""Backend adapter interface — the extension point for a future sweep target,
mirroring `nova-load`'s `VectorStore`/`VectorStoreConfig` and `nova-storm`'s
`QueryTarget`/`TargetConfig` split (see AGENTS.md). Qdrant
(`nova_sweep.backends.qdrant`) is the only concrete implementation today.

A backend owns three things: its own `target:` config fields (subclassing
`TargetConfigBase`), the one live pre-flight check (`collection_exists`), and
translation of sweep params into `nova-load`/`nova-storm` configs. The runner
(`runner.py`) only ever calls through this interface — it has no
backend-specific knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from nova_sweep.config import SweepConfig
    from nova_sweep.slices import Slice

# The one named dense vector every generated nova-load/nova-storm config
# uses today — a nova-sweep-level convention, not backend-specific, but
# backends are what actually stamp it into generated configs.
VECTOR_NAME = "dense"


class TargetConfigBase(BaseModel):
    """Fields every backend's target config carries, regardless of `type`."""

    model_config = ConfigDict(extra="forbid")

    type: str
    # never (default) | always. `never` means a
    # pre-existing same-named collection is a hard stop unless `--skip-insert`
    # is passed on the CLI; `always` unconditionally deletes + reloads.
    recreate: Literal["never", "always"] = "never"


def datasource_type(path: str) -> str:
    return "s3" if path.startswith("s3://") else "local"


def base_load_config(cfg: "SweepConfig") -> dict:
    """`datasource`/`vectors` skeleton shared by every backend's generated
    nova-load config — only `vectorstore` (added by the backend) varies.
    `datasource`/`vectors` are required by nova-load's shared `LoadConfig`
    schema but unused by `reindex`/`delete` (only `vectorstore.collection_name`
    matters there — see that command's own doc comment in
    `crates/nova-load/src/lib.rs`); building this once keeps every generated
    config valid without duplicating the dummy fields per backend."""
    return {
        "datasource": {"type": datasource_type(cfg.corpus.path), "path": cfg.corpus.path},
        "vectors": {"dense": {"type": "dense", "column": cfg.corpus.dense_column}},
    }


class SweepBackend(ABC):
    """A sweep target backend: the collection-existence check plus every
    generated `nova-load`/`nova-storm` config the runner needs, for one
    `target.type`."""

    @abstractmethod
    def collection_exists(self, cfg: "SweepConfig", collection_name: str) -> bool: ...

    @abstractmethod
    def build_load_config(self, cfg: "SweepConfig", slc: "Slice", *, recreate: bool) -> dict: ...

    @abstractmethod
    def build_reindex_config(self, cfg: "SweepConfig", slc: "Slice", variant: dict) -> dict: ...

    @abstractmethod
    def build_delete_config(self, cfg: "SweepConfig", collection_name: str) -> dict: ...

    @abstractmethod
    def build_storm_config(self, cfg: "SweepConfig", slc: "Slice", search: dict) -> dict: ...
