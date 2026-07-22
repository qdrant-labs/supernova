"""Qdrant sweep backend — the only concrete implementation today. Owns the
`target.type: qdrant` config fields plus every Qdrant-specific translation of
sweep params into `nova-load`/`nova-storm` configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from qdrant_client import QdrantClient

from nova_sweep.backends.base import VECTOR_NAME, SweepBackend, TargetConfigBase, base_load_config

if TYPE_CHECKING:
    from nova_sweep.config import SweepConfig
    from nova_sweep.slices import Slice

# Search-entry keys that route to `query.search_params` instead of `load.*` —
# Qdrant's own search-param vocabulary, not a generic sweep concept.
_SEARCH_PARAM_KEYS = {"hnsw_ef", "exact", "quantization"}


class QdrantTargetConfig(TargetConfigBase):
    type: Literal["qdrant"] = "qdrant"

    url: str
    api_key: str | None = None


class QdrantBackend(SweepBackend):
    def collection_exists(self, cfg: "SweepConfig", collection_name: str) -> bool:
        """nova-load's own CLI has no bare existence query (`inspect`
        deliberately never connects to the store) — this is the one place
        `nova-sweep` talks to Qdrant directly rather than through a
        `nova-load`/`nova-storm` subprocess.

        `prefer_grpc=True`: `target.url` is the same gRPC endpoint (typically
        port 6334) the Rust tools connect to via `qdrant_client::Qdrant::from_url`
        — the Python client defaults to REST, which would need a second port.
        """
        target: QdrantTargetConfig = cfg.target
        client = QdrantClient(
            url=target.url, api_key=target.api_key, prefer_grpc=True
        )
        try:
            return client.collection_exists(collection_name)
        finally:
            client.close()

    def _vectorstore_block(self, cfg: "SweepConfig", collection_name: str) -> dict:
        target: QdrantTargetConfig = cfg.target
        return {
            "type": "qdrant",
            "collection_name": collection_name,
            "url": target.url,
            **({"api_key": target.api_key} if target.api_key else {}),
        }

    def build_load_config(self, cfg: "SweepConfig", slc: "Slice", *, recreate: bool) -> dict:
        load_cfg = base_load_config(cfg)
        load_cfg["vectorstore"] = self._vectorstore_block(cfg, slc.collection_name)
        # Overlay this data_layout's `vectors.dense.*` overrides (distance/datatype/size/...).
        # `vectors`/`vectors.dense` must expand to mappings — guarded explicitly
        # so a typo'd axis (e.g. `data_layouts: {vectors: [...]}`, missing the
        # `.dense.<field>` suffix) fails with a clear message instead of an
        # opaque AttributeError/ValueError from `dict.get`/`dict.update`.
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
        # The recommended id_expression for supernova-produced corpora — matches
        # `nova bf`'s own point-id derivation, so ground truth lines up.
        load_cfg["datasource"]["id_expression"] = "vf_point_id(filename, file_row_number)"
        # Every other data_layout key is a collection-wide create-time param
        # (shard_number, replication_factor, write_consistency_factor,
        # on_disk_payload — see QdrantParams in
        # crates/nova-load/src/stores/qdrant.rs) that lives under
        # vectorstore.params, sibling to `recreate`. `vectors`/`_name` are
        # handled separately above.
        layout_params = {k: v for k, v in slc.data_layout.items() if k not in ("vectors", "_name")}
        load_cfg["vectorstore"]["params"] = {**layout_params, "recreate": recreate}
        return load_cfg

    def build_reindex_config(self, cfg: "SweepConfig", slc: "Slice", variant: dict) -> dict:
        reindex_cfg = base_load_config(cfg)
        reindex_cfg["vectorstore"] = self._vectorstore_block(cfg, slc.collection_name)
        reindex_cfg["vectorstore"]["params"] = {k: v for k, v in variant.items() if k != "_name"}
        return reindex_cfg

    def build_delete_config(self, cfg: "SweepConfig", collection_name: str) -> dict:
        delete_cfg = base_load_config(cfg)
        delete_cfg["vectorstore"] = self._vectorstore_block(cfg, collection_name)
        return delete_cfg

    def build_storm_config(self, cfg: "SweepConfig", slc: "Slice", search: dict) -> dict:
        target: QdrantTargetConfig = cfg.target
        search_params: dict[str, Any] = {}
        load: dict[str, Any] = {}
        top_k = 10
        for key, value in search.items():
            if key in ("_name",):
                continue
            if key == "top_k":
                top_k = value
            elif key in _SEARCH_PARAM_KEYS:
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

        return {
            "target": {
                "type": "qdrant",
                "url": target.url,
                **({"api_key": target.api_key} if target.api_key else {}),
                "collection_name": slc.collection_name,
            },
            "query": query,
            "load": load,
        }
