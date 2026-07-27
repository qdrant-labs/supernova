"""Elasticsearch sweep backend. Owns the `target.type: elastic` config fields
plus the ES-specific translation of sweep params into `nova-load`/`nova-storm`
configs.

Differences from Qdrant. (1) ES's nova-load `vectorstore` block is FLAT and uses
`index_name` (not `collection_name`) — the swept collection name is mapped to the
index name. `index_options` (the HNSW mapping params) sit flat on the block.
(2) Its search-param vocabulary is `{num_candidates}` (must be >= top_k), routed
to `query.search_params`. (3) `distance`/`similarity` is fixed at index creation
and CANNOT be changed by `reindex` — so a distance override must be a
`data_layouts` axis (fresh load), never an `index_variants` axis (the nova-load
reindex will error otherwise). The existence pre-flight uses a plain HTTP
HEAD via stdlib `urllib`, so no `elasticsearch` python dep.
"""

from __future__ import annotations

import base64
import ssl
import urllib.error
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

# ES's search-param vocabulary — the per-shard HNSW candidate pool.
_SEARCH_PARAM_KEYS = {"num_candidates"}


class ElasticTargetConfig(TargetConfigBase):
    type: Literal["elastic"] = "elastic"

    url: str
    username: str | None = None
    password: str | None = None
    # base64 `id:api_key` — alternative to username/password.
    api_key: str | None = None
    # DEV ONLY: skip TLS cert validation (a default ES 8 dev node self-signs).
    tls_insecure: bool = False

    @model_validator(mode="after")
    def _credentials_paired(self) -> "ElasticTargetConfig":
        """Basic auth needs both `username` and `password` — one without the
        other silently sends no auth (nova-sweep's pre-flight falls through to
        no header) and surfaces as a late 401. Use `api_key` for token auth
        instead. Fail here, at config-parse time, with a clear message."""
        if bool(self.username) != bool(self.password):
            raise ValueError(
                "elastic target: `username` and `password` must be set together "
                "(use `api_key` for token auth)"
            )
        return self


class ElasticBackend(SweepBackend):
    def _auth_header(self, target: "ElasticTargetConfig") -> dict[str, str]:
        if target.api_key:
            return {"Authorization": f"ApiKey {target.api_key}"}
        if target.username and target.password:
            token = base64.b64encode(f"{target.username}:{target.password}".encode()).decode()
            return {"Authorization": f"Basic {token}"}
        return {}

    def collection_exists(self, cfg: "SweepConfig", collection_name: str) -> bool:
        """Existence pre-flight via a plain `HEAD /<index>` (200 = exists,
        404 = not) — stdlib `urllib`, no `elasticsearch` python dep. Here the
        swept collection name IS the ES index name."""
        target: ElasticTargetConfig = cfg.target
        base = target.url.rstrip("/")
        req = urllib.request.Request(
            f"{base}/{collection_name}", headers=self._auth_header(target), method="HEAD"
        )
        ctx = ssl._create_unverified_context() if target.tls_insecure else None
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def _vectorstore_block(self, cfg: "SweepConfig", collection_name: str) -> dict:
        target: ElasticTargetConfig = cfg.target
        # The swept collection name maps to ES's `index_name`.
        block: dict[str, Any] = {"type": "elastic", "index_name": collection_name, "url": target.url}
        if target.username:
            block["username"] = target.username
        if target.password:
            block["password"] = target.password
        if target.api_key:
            block["api_key"] = target.api_key
        if target.tls_insecure:
            block["tls_insecure"] = True
        return block

    def build_load_config(self, cfg: "SweepConfig", slc: "Slice", *, recreate: bool) -> dict:
        load_cfg = base_load_config(cfg)
        load_cfg["vectorstore"] = self._vectorstore_block(cfg, slc.collection_name)
        layout_params = apply_corpus_layout(load_cfg, slc)
        # ES vectorstore fields are FLAT (e.g. `index_options`), unlike Qdrant's
        # `params:` sub-block.
        load_cfg["vectorstore"].update({**layout_params, "recreate": recreate})
        return load_cfg

    def build_reindex_config(self, cfg: "SweepConfig", slc: "Slice", variant: dict) -> dict:
        reindex_cfg = base_load_config(cfg)
        reindex_cfg["vectorstore"] = self._vectorstore_block(cfg, slc.collection_name)
        # Carry the data_layout's distance into the reindex vectors block: ES
        # derives `similarity` from `vectors.dense.distance` and rejects an
        # in-place similarity change, so omitting it defaults to cosine and makes
        # reindexing a non-cosine index fail the similarity-immutability check.
        overlay_layout_vectors(reindex_cfg, slc)
        # Base the reindex on the data_layout's own index_options, THEN overlay
        # the variant's — same as Milvus, so the two flat backends behave alike
        # and a data_layout's index config isn't dropped when a variant omits it.
        reindex_cfg["vectorstore"].update(layout_structural_params(slc))
        reindex_cfg["vectorstore"].update({k: v for k, v in variant.items() if k != "_name"})
        return reindex_cfg

    def build_delete_config(self, cfg: "SweepConfig", collection_name: str) -> dict:
        delete_cfg = base_load_config(cfg)
        delete_cfg["vectorstore"] = self._vectorstore_block(cfg, collection_name)
        return delete_cfg

    def build_storm_config(self, cfg: "SweepConfig", slc: "Slice", search: dict) -> dict:
        target: ElasticTargetConfig = cfg.target
        query, load = build_storm_query(cfg, search, _SEARCH_PARAM_KEYS)

        target_block: dict[str, Any] = {
            "type": "elastic",
            "url": target.url,
            "index_name": slc.collection_name,
        }
        if target.username:
            target_block["username"] = target.username
        if target.password:
            target_block["password"] = target.password
        if target.api_key:
            target_block["api_key"] = target.api_key
        if target.tls_insecure:
            target_block["tls_insecure"] = True

        return {"target": target_block, "query": query, "load": load}
