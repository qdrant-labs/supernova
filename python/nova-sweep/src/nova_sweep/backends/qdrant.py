"""Qdrant sweep backend. Owns the `target.type: qdrant` config fields plus
every Qdrant-specific translation of sweep params into `nova-load`/`nova-storm`
configs. Unlike the Milvus/Elastic backends (flat vectorstore blocks), Qdrant
nests structural params under `vectorstore.params`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qdrant_client import QdrantClient

from nova_sweep.backends.base import (
    SweepBackend,
    TargetConfigBase,
    apply_corpus_layout,
    base_load_config,
    build_storm_query,
)

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
        # Overlay `vectors.dense.*` + stamp the supernova id_expression, and get
        # back the remaining structural data_layout params. Unlike Milvus/Elastic
        # (flat), Qdrant nests those under `vectorstore.params` — collection-wide
        # create-time params (shard_number, replication_factor,
        # write_consistency_factor, on_disk_payload — see QdrantParams in
        # crates/nova-load/src/stores/qdrant.rs), sibling to `recreate`.
        layout_params = apply_corpus_layout(load_cfg, slc)
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
        query, load = build_storm_query(cfg, search, _SEARCH_PARAM_KEYS)

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
