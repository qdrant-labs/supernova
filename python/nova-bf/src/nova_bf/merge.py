"""The `merge` phase: inter-worker reduce over per-rank partials.

Each partial holds, per query, the top-K over one worker's disjoint slice of the
corpus (stride partition → no overlapping hits). So merging is just: concatenate
each query's candidates across partials and keep the global top-K. Carried
payload is identical across partials, so we take it from whichever appears first.
"""

from __future__ import annotations

import logging

from collections import defaultdict

from tqdm import tqdm

from nova_bf.config import BruteForceConfig
from nova_bf.io import Store
from nova_bf.results import RESERVED, build_result_table, partial_dir, result_name

logger = logging.getLogger(__name__)


def run_merge(cfg: BruteForceConfig) -> str:
    k = cfg.params.k
    out = Store(cfg.output.path)
    partials = out.list_parquets(subpath=partial_dir(cfg))
    if not partials:
        raise RuntimeError(
            f"no partial results under {cfg.output.path}/{partial_dir(cfg)}/ — "
            "run `bf compute --num-jobs N` first"
        )
    logger.info("merging %d partials (k=%d)", len(partials), k)

    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    payloads: dict[str, dict] = {}
    payload_cols: list[str] = []

    for f in tqdm(partials, unit="file", desc="merge", dynamic_ncols=True):
        table = out.read_columns(f.read_path, None)
        if not payload_cols:
            payload_cols = [c for c in table.column_names if c not in RESERVED]
        for row in table.to_pylist():
            qid = row["query_id"]
            candidates[qid].extend(zip(row["hit_scores"], row["hit_ids"]))
            if qid not in payloads:
                payloads[qid] = {c: row[c] for c in payload_cols}

    query_ids = sorted(candidates)
    hit_ids, hit_scores = [], []
    payload: dict[str, list] = {c: [] for c in payload_cols}
    for qid in query_ids:
        top = sorted(candidates[qid], reverse=True)[:k]
        hit_scores.append([s for s, _ in top])
        hit_ids.append([h for _, h in top])
        for c in payload_cols:
            payload[c].append(payloads[qid][c])

    table = build_result_table(query_ids, payload, hit_ids, hit_scores)
    path = out.write(result_name(cfg), table)
    logger.info("wrote %s (%d queries)", path, len(query_ids))
    return path
