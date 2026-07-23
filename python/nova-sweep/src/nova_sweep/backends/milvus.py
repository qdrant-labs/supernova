"""Milvus sweep backend. Owns the `target.type: milvus` config fields plus the
Milvus-specific translation of sweep params into `nova-load`/`nova-storm`
configs.

Two things differ from Qdrant. (1) Milvus's nova-load `vectorstore` block is
FLAT — `index_type`/`index_params`/etc. sit directly on it, not under a
`params:` sub-block — so data_layout/index-variant params are merged flat.
(2) Its search-param vocabulary is `{ef, nprobe}` (HNSW / IVF), routed to
`query.search_params`. The metric comes from the top-level `vectors.dense.distance`,
like every backend. The collection-existence pre-flight uses Milvus's REST API
(same endpoint as gRPC in 2.4+) via stdlib `urllib`, so no `pymilvus` dep.
"""

from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING, Any, Literal

from pydantic import model_validator

from nova_sweep.backends.base import (
    SweepBackend,
    TargetConfigBase,
    apply_corpus_layout,
    base_load_config,
    build_storm_query,
    layout_structural_params,
    overlay_layout_vectors,
)

if TYPE_CHECKING:
    from nova_sweep.config import SweepConfig
    from nova_sweep.slices import Slice

# Search-entry keys that route to `query.search_params` (Milvus's own vocabulary:
# `ef` for HNSW, `nprobe` for IVF) rather than the top-level `load.*` block.
_SEARCH_PARAM_KEYS = {"ef", "nprobe"}


class MilvusTargetConfig(TargetConfigBase):
    type: Literal["milvus"] = "milvus"

    url: str
    username: str | None = None
    password: str | None = None

    @model_validator(mode="after")
    def _credentials_paired(self) -> "MilvusTargetConfig":
        """Milvus REST bearer auth is `username:password` — one without the
        other would silently send no auth (nova-sweep's pre-flight) or surface
        as a late 401, and nova-load's own milvus backend rejects it outright.
        Fail here, at config-parse time, with a clear message instead."""
        if bool(self.username) != bool(self.password):
            raise ValueError(
                "milvus target: `username` and `password` must be set together"
            )
        return self


class MilvusBackend(SweepBackend):
    def collection_exists(self, cfg: "SweepConfig", collection_name: str) -> bool:
        """Pre-flight existence check over Milvus's REST API (list collections),
        the one place nova-sweep talks to the store directly rather than through
        a nova-load/nova-storm subprocess. REST (not `pymilvus`) keeps the check
        dependency-free and consistent with how the nova-load/nova-storm milvus
        backends already use REST."""
        target: MilvusTargetConfig = cfg.target
        base = target.url.rstrip("/")
        req = urllib.request.Request(
            f"{base}/v2/vectordb/collections/list",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if target.username and target.password:
            req.add_header("Authorization", f"Bearer {target.username}:{target.password}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        if body.get("code") != 0:
            raise RuntimeError(f"milvus collections/list failed: {body}")
        return collection_name in (body.get("data") or [])

    def _vectorstore_block(self, cfg: "SweepConfig", collection_name: str) -> dict:
        target: MilvusTargetConfig = cfg.target
        block: dict[str, Any] = {
            "type": "milvus",
            "collection_name": collection_name,
            "url": target.url,
        }
        if target.username:
            block["username"] = target.username
        if target.password:
            block["password"] = target.password
        return block

    def build_load_config(self, cfg: "SweepConfig", slc: "Slice", *, recreate: bool) -> dict:
        load_cfg = base_load_config(cfg)
        load_cfg["vectorstore"] = self._vectorstore_block(cfg, slc.collection_name)
        layout_params = apply_corpus_layout(load_cfg, slc)
        # Milvus vectorstore fields are FLAT (index_type/index_params/id_max_length
        # sit directly on the block), unlike Qdrant's `params:` sub-block.
        load_cfg["vectorstore"].update({**layout_params, "recreate": recreate})
        return load_cfg

    def build_reindex_config(self, cfg: "SweepConfig", slc: "Slice", variant: dict) -> dict:
        reindex_cfg = base_load_config(cfg)
        reindex_cfg["vectorstore"] = self._vectorstore_block(cfg, slc.collection_name)
        # Carry the data_layout's distance into the reindex vectors block: Milvus
        # rebuilds the index with the metric from `vectors.dense.distance`, so
        # omitting it defaults the rebuilt index to COSINE and silently corrupts
        # a non-cosine collection's metric.
        overlay_layout_vectors(reindex_cfg, slc)
        # Base the rebuild on the data_layout's own index_type/index_params, THEN
        # overlay the variant's. Milvus reindex always rebuilds the index from
        # config (defaulting a missing index_type to AUTOINDEX), so without this
        # base an index set via data_layouts is silently discarded on reindex.
        reindex_cfg["vectorstore"].update(layout_structural_params(slc))
        reindex_cfg["vectorstore"].update({k: v for k, v in variant.items() if k != "_name"})
        return reindex_cfg

    def build_delete_config(self, cfg: "SweepConfig", collection_name: str) -> dict:
        delete_cfg = base_load_config(cfg)
        delete_cfg["vectorstore"] = self._vectorstore_block(cfg, collection_name)
        return delete_cfg

    def build_storm_config(self, cfg: "SweepConfig", slc: "Slice", search: dict) -> dict:
        target: MilvusTargetConfig = cfg.target
        query, load = build_storm_query(cfg, search, _SEARCH_PARAM_KEYS)

        target_block: dict[str, Any] = {
            "type": "milvus",
            "url": target.url,
            "collection_name": slc.collection_name,
        }
        if target.username:
            target_block["username"] = target.username
        if target.password:
            target_block["password"] = target.password

        return {"target": target_block, "query": query, "load": load}
