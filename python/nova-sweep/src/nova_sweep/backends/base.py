"""Backend adapter interface — the extension point for a future sweep target,
mirroring `nova-load`'s `VectorStore`/`VectorStoreConfig` and `nova-storm`'s
`QueryTarget`/`TargetConfig` split (see AGENTS.md). Qdrant, Milvus, and
Elasticsearch (`nova_sweep.backends.{qdrant,milvus,elastic}`) are the concrete
implementations.

A backend owns three things: its own `target:` config fields (subclassing
`TargetConfigBase`), the one live pre-flight check (`collection_exists`), and
translation of sweep params into `nova-load`/`nova-storm` configs. The runner
(`runner.py`) only ever calls through this interface — it has no
backend-specific knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

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


def overlay_layout_vectors(load_cfg: dict, slc: "Slice") -> None:
    """Overlay a data_layout's `vectors.dense.*` overrides (distance/datatype/…)
    onto `load_cfg`'s vectors block, guarding a malformed `vectors` axis with a
    clear message. Shared by the fresh-load AND reindex paths for every backend:
    the metric (`distance`) MUST be carried into reindex too, or nova-load
    defaults it — silently corrupting the rebuilt index's metric on Milvus, or
    tripping the similarity-immutability check on Elastic."""
    layout_vectors_block = slc.data_layout.get("vectors", {})
    if not isinstance(layout_vectors_block, dict):
        raise ValueError(
            f"data_layouts: `vectors` must expand to a mapping (did you mean "
            f"`vectors.dense.<field>: [...]`?), got {layout_vectors_block!r} "
            f"for data_layout '{slc.data_layout_name}'"
        )
    layout_vectors = layout_vectors_block.get("dense", {})
    if not isinstance(layout_vectors, dict):
        raise ValueError(
            f"data_layouts: `vectors.dense` must expand to a mapping of field "
            f"overrides (e.g. `vectors.dense.distance: [...]`), got "
            f"{layout_vectors!r} for data_layout '{slc.data_layout_name}'"
        )
    load_cfg["vectors"]["dense"].update(layout_vectors)


def layout_structural_params(slc: "Slice") -> dict:
    """The data_layout's structural params beyond `vectors`/`_name` — the
    backend-specific create-time / index knobs (Qdrant `shard_number` & friends,
    Milvus `index_type`/`index_params`, Elastic `index_options`). Applied at
    fresh load AND re-asserted as the *base* of a reindex (which then overlays
    its own variant params on top), so a data_layout's index config survives the
    per-variant reindex instead of being reset to the backend default."""
    return {k: v for k, v in slc.data_layout.items() if k not in ("vectors", "_name")}


def apply_corpus_layout(load_cfg: dict, slc: "Slice") -> dict:
    """Fresh-load helper (`build_load_config`): overlay the data_layout's
    `vectors.dense.*` overrides, stamp the supernova `id_expression` (so ground
    truth lines up with `nova bf`'s point-id derivation), and return the
    *remaining* (structural) data_layout params for the caller to place — flat on
    the vectorstore block (Milvus/Elastic) or nested under `vectorstore.params`
    (Qdrant)."""
    overlay_layout_vectors(load_cfg, slc)
    load_cfg["datasource"]["id_expression"] = "vf_point_id(filename, file_row_number)"
    return layout_structural_params(slc)


def build_storm_query(
    cfg: "SweepConfig", search: dict, search_param_keys: set[str]
) -> tuple[dict, dict]:
    """Split one `searches` grid entry into the `query` and `load` blocks every
    backend's nova-storm config shares. `top_k` sets `query.top_k`; keys in
    `search_param_keys` (the backend's own vocabulary) go to
    `query.search_params`; everything else routes to nova-storm's
    backend-neutral `load:` block. `query.source` is built from `cfg.queries`.
    Only the `target` block is backend-specific, so each backend assembles that
    itself. Returns `(query, load)`."""
    search_params: dict[str, Any] = {}
    load: dict[str, Any] = {}
    top_k = 10
    for key, value in search.items():
        if key == "_name":
            continue
        if key == "top_k":
            top_k = value
        elif key in search_param_keys:
            search_params[key] = value
        else:
            load[key] = value

    source: dict[str, Any] = {
        "uri": cfg.queries.uri,
        "column": cfg.queries.column,
        "limit": cfg.queries.limit,
    }
    if cfg.queries.ground_truth_column:
        source["ground_truth_column"] = cfg.queries.ground_truth_column

    query: dict[str, Any] = {"vector_name": VECTOR_NAME, "top_k": top_k, "source": source}
    if search_params:
        query["search_params"] = search_params
    return query, load


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
