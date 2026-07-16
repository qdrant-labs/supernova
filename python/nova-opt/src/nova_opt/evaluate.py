"""Evaluation backends: how a candidate's missing artifact levels get built
and how one search operating point gets measured.

`LiveQdrantEvaluator` drives the same nova-load / nova-storm subprocesses as
nova-sweep (and generates the same config shapes — see
`python/nova-sweep/src/nova_sweep/backends/qdrant.py`), but level by level:
`layout` is a fresh `nova-load run`, `index` and `quant` are separate
`nova-load reindex` patches, `search` is one `nova-storm --json` run. That
split is what lets the optimizer pay only for the levels an artifact cache
miss actually requires.

`ReplayEvaluator` answers from a previously measured results table instead
(e.g. an exhaustive nova-sweep run) — the offline harness for comparing the
tuner and its baselines without touching a live cluster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nova_opt.space import QUANT_VARIANTS, Candidate, config_features

log = logging.getLogger("nova_opt")

# nova-sweep's convention: the one named dense vector every generated config uses
VECTOR_NAME = "dense"

# recall-training-data metric vocabulary (COSINE / IP / L2, plus lowercase
# aliases) -> nova-load's own `vectors.<name>.distance` vocabulary
_LOAD_DISTANCE = {
    "cosine": "cosine",
    "ip": "dot",
    "dot": "dot",
    "l2": "euclid",
    "euclidean": "euclid",
}


class EvalError(Exception):
    """A build or search step failed — recorded as an error trial, not a crash."""


@dataclass
class SearchOutcome:
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_recall: float | None
    seconds: float
    raw: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    @abstractmethod
    def build(self, cand: Candidate, levels: tuple[str, ...]) -> dict[str, float]:
        """Materialize the given non-search levels for `cand`, in order.
        Returns measured seconds per level. Raises `EvalError` on failure."""

    @abstractmethod
    def search(self, cand: Candidate) -> SearchOutcome:
        """Measure one search operating point against the (already built)
        artifact. Raises `EvalError` on failure."""


# ---------------------------------------------------------------------------
# live qdrant, via nova-load / nova-storm subprocesses


def _resolve_binary(name: str) -> str:
    """`$NOVA_<NAME>_BIN`, then PATH — same override convention as nova-sweep."""
    env = f"NOVA_{name.removeprefix('nova-').upper().replace('-', '_')}_BIN"
    return os.environ.get(env) or shutil.which(name) or name


def _run(binary: str, args: list[str]) -> tuple[bool, str, str]:
    exe = _resolve_binary(binary)
    log.info("run: %s %s", exe, " ".join(args))
    proc = subprocess.run([exe, *args], capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()


def _layout_hash(cand: Candidate) -> str:
    return hashlib.md5(repr(cand.layout_key).encode()).hexdigest()[:10]


class LiveQdrantEvaluator(Evaluator):
    def __init__(
        self,
        *,
        run_name: str,
        corpus_path: str,
        corpus_column: str,
        queries_uri: str,
        queries_column: str,
        ground_truth_column: str | None,
        queries_limit: int,
        url: str,
        api_key: str | None = None,
        distance: str | None = None,
        duration_s: float = 10.0,
    ):
        self.run_name = run_name
        self.corpus_path = corpus_path
        self.corpus_column = corpus_column
        self.queries_uri = queries_uri
        self.queries_column = queries_column
        self.ground_truth_column = ground_truth_column
        self.queries_limit = queries_limit
        self.url = url
        self.api_key = api_key
        self.distance = distance
        self.duration_s = duration_s

    # one collection per layout artifact; index/quant are reindex patches on it
    def collection_name(self, cand: Candidate) -> str:
        return f"{self.run_name}_{_layout_hash(cand)}"

    def _vectorstore(self, cand: Candidate) -> dict:
        return {
            "type": "qdrant",
            "collection_name": self.collection_name(cand),
            "url": self.url,
            **({"api_key": self.api_key} if self.api_key else {}),
        }

    def _base_load_config(self, cand: Candidate) -> dict:
        source_type = "s3" if self.corpus_path.startswith("s3://") else "local"
        dense: dict[str, Any] = {
            "type": "dense",
            "column": self.corpus_column,
            "datatype": cand.layout.dtype,
        }
        if self.distance:
            dense["distance"] = _LOAD_DISTANCE.get(
                self.distance.lower(), self.distance
            )
        return {
            "datasource": {
                "type": source_type,
                "path": self.corpus_path,
                # matches nova-bf's point-id derivation so ground truth lines up
                "id_expression": "vf_point_id(filename, file_row_number)",
            },
            "vectors": {VECTOR_NAME: dense},
            "vectorstore": self._vectorstore(cand),
        }

    def _load_params(self, cand: Candidate, level: str) -> dict:
        if level == "layout":
            return {
                "shard_number": cand.layout.shard_count,
                "on_disk_payload": cand.layout.on_disk_payload,
                "optimizers": {"default_segment_number": cand.layout.segments},
                "recreate": True,
            }
        if level == "index":
            params: dict[str, Any] = {
                "hnsw": {
                    "m": cand.index.m,
                    "ef_construct": cand.index.ef_construct,
                    "on_disk": cand.index.on_disk,
                }
            }
            if cand.index.indexing_threshold is not None:
                params["optimizers"] = {
                    "indexing_threshold": cand.index.indexing_threshold
                }
            return params
        if level == "quant":
            block = dict(QUANT_VARIANTS[cand.quant.variant].load_block)
            if block["type"] != "none":
                block["always_ram"] = cand.quant.always_ram
            return {"quantization": block}
        raise ValueError(f"not a buildable level: '{level}'")

    def _run_step(self, subcommand: str, cfg: dict, label: str) -> float:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"{label}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            t0 = time.monotonic()
            ok, stdout, stderr = _run("nova-load", [subcommand, str(path)])
            seconds = time.monotonic() - t0
        if not ok:
            raise EvalError(f"{label} failed: {stderr or stdout}")
        return seconds

    def build(self, cand: Candidate, levels: tuple[str, ...]) -> dict[str, float]:
        costs: dict[str, float] = {}
        for level in levels:
            if level == "search":
                continue
            cfg = self._base_load_config(cand)
            if level == "layout":
                cfg["vectorstore"]["params"] = self._load_params(cand, level)
                costs[level] = self._run_step("run", cfg, "load")
            else:
                cfg["vectorstore"]["params"] = self._load_params(cand, level)
                costs[level] = self._run_step("reindex", cfg, f"reindex_{level}")
        return costs

    def search(self, cand: Candidate) -> SearchOutcome:
        s = cand.search
        search_params: dict[str, Any] = {"hnsw_ef": s.ef_search}
        if s.rescore is not None:
            search_params["quantization"] = {"rescore": s.rescore}
        source: dict[str, Any] = {
            "uri": self.queries_uri,
            "column": self.queries_column,
            "limit": self.queries_limit,
        }
        if self.ground_truth_column:
            source["ground_truth_column"] = self.ground_truth_column
        storm_cfg = {
            "target": {
                "type": "qdrant",
                "url": self.url,
                **({"api_key": self.api_key} if self.api_key else {}),
                "collection_name": self.collection_name(cand),
            },
            "query": {
                "vector_name": VECTOR_NAME,
                "top_k": s.top_k,
                "source": source,
                "search_params": search_params,
            },
            "load": {
                "concurrency": s.concurrency,
                "batch_size": s.batch_size,
                "duration_s": self.duration_s,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "storm.yaml"
            path.write_text(yaml.safe_dump(storm_cfg, sort_keys=False))
            t0 = time.monotonic()
            ok, stdout, stderr = _run("nova-storm", [str(path), "--json"])
            seconds = time.monotonic() - t0
        if not ok:
            raise EvalError(f"storm failed: {stderr or stdout}")
        try:
            summary = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise EvalError(f"failed to parse nova-storm --json output: {e}") from e
        return SearchOutcome(
            qps=float(summary.get("qps", 0.0)),
            p50_ms=float(summary.get("p50_ms", 0.0)),
            p95_ms=float(summary.get("p95_ms", 0.0)),
            p99_ms=float(summary.get("p99_ms", 0.0)),
            mean_recall=summary.get("mean_recall"),
            seconds=seconds,
            raw=summary,
        )


# ---------------------------------------------------------------------------
# offline replay from a measured results table


# candidate feature -> replay-table column (the config_features naming, which
# both data.csv and stats-joined sweep exports share)
_REPLAY_MATCH_COLUMNS = (
    "number_of_segments",
    "hnsw_m",
    "ef_construct",
    "quantization_variant",
    "ef_search",
    "top_k",
    "rescore",
)


class ReplayEvaluator(Evaluator):
    """Looks candidates up in a table of already-measured runs. Matching is
    on the intersection of `_REPLAY_MATCH_COLUMNS` with the table's columns
    (a table without e.g. `rescore` just won't discriminate on it); a
    candidate with no matching row raises `EvalError`, which the optimizer
    records and moves past. Build "costs" come from per-level seconds
    columns (`layout_seconds`, `index_seconds`, `quant_seconds`) when
    present, else zero — replay time is free either way."""

    def __init__(self, table: pd.DataFrame, *, latency_column: str = "p95_ms",
                 qps_column: str = "qps"):
        self.table = table.reset_index(drop=True)
        self.latency_column = latency_column
        self.qps_column = qps_column
        for col in (qps_column, latency_column):
            if col not in table.columns:
                raise ValueError(f"replay table is missing required column '{col}'")
        self.match_columns = [
            c for c in _REPLAY_MATCH_COLUMNS if c in table.columns
        ]
        if not self.match_columns:
            raise ValueError(
                "replay table shares no candidate columns; expected some of "
                f"{_REPLAY_MATCH_COLUMNS}"
            )
        self.recall_column = next(
            (c for c in ("mean_recall", "mean_recall_at_k") if c in table.columns),
            None,
        )

    def _lookup(self, cand: Candidate) -> pd.Series:
        feats = config_features(
            cand,
            {
                "corpus_size": 0, "query_count": 0, "vector_dim": 1,
                "data_size_bytes": 1, "distance_metric": "",
            },
        )
        mask = pd.Series(True, index=self.table.index)
        for col in self.match_columns:
            want = feats[col]
            if want is None:
                mask &= self.table[col].isna()
            else:
                mask &= self.table[col] == want
        hits = self.table[mask]
        if hits.empty:
            raise EvalError(f"no replay row matches candidate {feats}")
        return hits.iloc[0]

    def build(self, cand: Candidate, levels: tuple[str, ...]) -> dict[str, float]:
        row = self._lookup(cand)
        return {
            level: float(row.get(f"{level}_seconds", 0.0) or 0.0)
            for level in levels
            if level != "search"
        }

    def search(self, cand: Candidate) -> SearchOutcome:
        row = self._lookup(cand)
        recall = float(row[self.recall_column]) if self.recall_column else None
        return SearchOutcome(
            qps=float(row[self.qps_column]),
            p50_ms=float(row.get("p50_ms", row[self.latency_column])),
            p95_ms=float(row[self.latency_column]),
            p99_ms=float(row.get("p99_ms", row[self.latency_column])),
            mean_recall=recall,
            seconds=float(row.get("search_seconds", 0.0) or 0.0),
            raw={},
        )
