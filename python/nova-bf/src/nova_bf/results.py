"""Shared result schema + output naming for compute and merge."""

from __future__ import annotations

import pyarrow as pa

from nova_bf.config import BruteForceConfig

# Reserved output columns; everything else in a result row is carried payload.
RESERVED = ("query_id", "hit_ids", "hit_scores")


def queries_stem(queries_path: str) -> str:
    base = queries_path.rstrip("/").split("/")[-1]
    return base[:-8] if base.endswith(".parquet") else base


def result_name(cfg: BruteForceConfig) -> str:
    return f"bf_{queries_stem(cfg.queries.path)}_k{cfg.params.k}.parquet"


def partial_dir(cfg: BruteForceConfig) -> str:
    return f"_bf_partial_{queries_stem(cfg.queries.path)}_k{cfg.params.k}"


def build_result_table(
    query_ids: list[str],
    payload: dict[str, list],
    hit_ids: list[list[str]],
    hit_scores: list[list[float]],
) -> pa.Table:
    data: dict = {"query_id": pa.array(query_ids, pa.string())}
    for col, vals in payload.items():
        data[col] = pa.array(vals)
    data["hit_ids"] = pa.array(hit_ids, pa.list_(pa.string()))
    data["hit_scores"] = pa.array(hit_scores, pa.list_(pa.float32()))
    return pa.table(data)
