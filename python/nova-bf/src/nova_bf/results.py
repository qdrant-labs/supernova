"""Shared result schema + output naming for compute and merge."""

from __future__ import annotations

import logging

import logging

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


<<<<<<< HEAD
def warn_if_short(short: int, total: int, k: int, logger: logging.Logger) -> None:
=======
def warn_if_short(short_count: int, total: int, k: int, logger: logging.Logger) -> None:
>>>>>>> refs/remotes/origin/master
    """Log if any query's FINAL top-K came out shorter than k. Not an error —
    hit_ids/hit_scores are already correctly truncated (see the `-inf` sentinel
    handling in compute.py) — just a signal that the corpus, or `filter` if one
    is configured, didn't have k matches for some queries, so this ground truth
    is smaller than requested rather than wrong.

<<<<<<< HEAD
    Takes pre-computed counts (not a hit_ids list) so the streaming `merge`, which
    never materializes a full hit_ids list, can tally `short` per batch and still
    share this one warning.
    """
    if short:
        logger.warning(
            "%d/%d quer%s returned fewer than k=%d hits — the corpus (after "
            "any `filter`) didn't have enough matches for them",
            short, total, "y" if short == 1 else "ies", k,
=======
    Takes pre-aggregated counts, not the raw `hit_ids` lists, so `merge`'s
    streaming batches (never all in memory at once — see `ParamsConfig.merge_batch_size`)
    can call this exactly like `compute`'s single in-memory list can.
    """
    if short_count:
        logger.warning(
            "%d/%d quer%s returned fewer than k=%d hits — the corpus (after "
            "any `filter`) didn't have enough matches for them",
            short_count, total, "y" if short_count == 1 else "ies", k,
>>>>>>> refs/remotes/origin/master
        )
