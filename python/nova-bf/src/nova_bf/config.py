"""Typed brute-force config (pydantic), with `${VAR}` env expansion.

    corpus:   the embedded parquets to search over (local dir / s3:// prefix)
    queries:  the query embeddings (a parquet file or dir)
    output:   where results land (local dir / s3:// prefix)
    params:   run-level knobs shared by every search (io_workers, merge tuning, …)
    searches: one or more independent top-K searches to compute in this run —
              each its own k/metric/vector_type/filter (see SearchSpec). Always
              required, even for a single search, so a config always says
              explicitly what kind of search it's running.
"""

from __future__ import annotations

import re

from typing import Literal

import yaml

from pydantic import BaseModel, ConfigDict, model_validator

import os

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def expand_env(text: str) -> str:
    """Expand `${VAR}` / `${VAR:-default}` against the environment (on raw YAML)."""

    def repl(m: re.Match) -> str:
        name, _, default = m.group(1).partition(":-")
        val = os.environ.get(name)
        if val:
            return val
        if ":-" in m.group(1):
            return default
        raise ValueError(
            f"environment variable '{name}' referenced in config is not set; "
            f"set it or supply a default with ${{{name}:-...}}"
        )

    return _ENV_RE.sub(repl, text)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    dense_column: str = "dense_embedding"
    # Struct<indices: list<uint32>, values: list<float32>> column, read instead
    # of dense_column by a search whose `vector_type` is "sparse" —
    # same schema nova-embed writes and nova-load reads (see docs/embedding).
    sparse_column: str = "sparse_embedding"
    # If set, hit_ids are taken verbatim from this already-unique column (e.g.
    # fineweb's `id` = "<urn:uuid:...>") — transparent for public data, and
    # resolvable without reconstructing the loader's hashing. If unset (default),
    # hit_ids are derived via make_point_id(corpus_file_key, row), matching the
    # point ids the loader synthesizes at ingest. Unlike make_point_id (a pure
    # function of file+row, recomputed only for the final K hits), a real id column
    # lives in the data, so it's read alongside the dense column and kept in RAM
    # per file for the worker's slice — budget ~(slice_rows × id_size) of host mem.
    id_column: str | None = None
    # Restrict which `.parquet` under `path` are searched (both are regexes matched
    # against each object's full path with `re.search`). `path` globs recursively,
    # so use these to skip siblings you didn't mean to include — e.g. a `prepared/`
    # folder someone dropped next to the shards:
    #   include: '/\d{3}/'      # only shard dirs 000/, 001/, …
    #   exclude: '/prepared/'   # …or just drop the one you don't want
    # include is applied first (keep only matches), then exclude (drop matches).
    include: str | None = None
    exclude: str | None = None


class QueriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    dense_column: str = "dense_embedding"
    sparse_column: str = "sparse_embedding"
    # If set, use this column as the query id verbatim; otherwise derive
    # make_point_id(queries_file_key, row) — same scheme as the corpus.
    id_column: str | None = None
    # Columns to carry from the queries file into each output row.
    payload_fields: list[str] = []


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ParamsConfig(BaseModel):
    """Run-level knobs shared by every search in `BruteForceConfig.searches` —
    IO/merge tuning, not search semantics (those live on each `SearchSpec`)."""

    model_config = ConfigDict(extra="forbid")

    # Concurrent corpus-file readers (one thread per in-flight file). IMPORTANT:
    # pyarrow reads parquet with pre_buffer=True, which dispatches the actual S3
    # byte-fetches to a SHARED global IO thread pool of size `io_thread_count`
    # (default ~8). So raising io_workers past ~io_thread_count adds NO real S3
    # concurrency — it only piles up read_table calls, inflates per-file latency,
    # and holds more decoded arrays in RAM (each reader ≈ one file; io_workers ×
    # file_size must fit host memory or the box OOMs — that's what killed the
    # 96/128-worker runs on a 16 GB g5.xlarge). Keep it modest; the real S3
    # concurrency knob is io_thread_count below. When `searches` mixes dense AND
    # sparse specs, each in-flight file's reader decodes BOTH columns at once, so
    # the per-file RAM budget above is (dense_bytes + sparse_bytes), not just one.
    io_workers: int = 16
    # pyarrow's global IO thread pool size = the TRUE S3 fetch concurrency (see
    # io_workers). 0 → leave pyarrow's default (~8). Raise it (e.g. 32) to test
    # whether the IO pool, rather than the NIC, is the throughput ceiling: if
    # `bf-bench wall_mbps` climbs toward the instance's NIC baseline you were
    # pool-bound; if it stays flat you're network-bound. Applied via
    # pa.set_io_thread_count() once at startup.
    io_thread_count: int = 0
    # `merge` reduces the W per-rank partials in row-batches of this many queries,
    # streaming the result to disk so the full output never sits in RAM (that's what
    # let the old merge OOM at 1M queries). Peak host memory is ~(this × W × k)
    # candidate slots. None (default) → auto: sized so the working set stays near
    # ~20M candidate slots regardless of W and k, floored at 1 and capped at the
    # query count. Set it explicitly to trade memory for fewer, larger batches.
    merge_batch_size: int | None = None
    # `merge`: when the partials live on S3, first bulk-download them to local disk
    # (ranged reads, io_workers-many concurrent) and merge from there, instead of
    # streaming each batch over S3 in lockstep. Trades local disk (~sum of partial
    # sizes) for a one-shot parallel download at full NIC speed + latency-free reads
    # during the reduce — worth it on a beefy box with fast NVMe. With `searches`
    # listing several searches, every search's downloads share ONE pool of
    # io_workers threads (see merge.py's `_prefetch_all`) rather than each search
    # getting its own — a search with fewer/smaller partials frees its share of
    # the pool for a slower search instead of sitting on dedicated capacity. No
    # effect when the partials are already local. Default off (a laptop
    # controller may lack the disk).
    merge_prefetch: bool = False


# A single scalar payload value, as it would appear in a corpus column.
MatchValue = str | int | float | bool


class RangeCondition(BaseModel):
    """Numeric bounds, combinable like Qdrant's Range (e.g. `gte` + `lt` together)."""

    model_config = ConfigDict(extra="forbid")

    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> "RangeCondition":
        if all(v is None for v in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("range condition needs at least one of gt/gte/lt/lte")
        return self


class FilterCondition(BaseModel):
    """One field predicate: keyword-style equality (`match`) or numeric bounds (`range`).

    `match` takes a scalar (equality) or a list (matches any of them — Qdrant's
    MatchAny). Exactly one of `match`/`range` must be set.
    """

    model_config = ConfigDict(extra="forbid")

    # The corpus column this condition reads and matches against — this is the
    # only place a filtered field is named; nothing else needs to declare it,
    # so `compute` reads exactly (and only) the columns the filter references.
    field: str
    match: MatchValue | list[MatchValue] | None = None
    range: RangeCondition | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "FilterCondition":
        if (self.match is None) == (self.range is None):
            raise ValueError(
                f"filter condition on `{self.field}` must set exactly one of `match` or `range`"
            )
        return self


class Filter(BaseModel):
    """A corpus-side payload filter, shaped like Qdrant's own filter (must/should/must_not).

    Restricts which corpus points are eligible neighbors for every query in the
    run — it does not touch queries themselves, same as a Qdrant search filter.
    `must` = AND, `should` = OR-at-least-one, `must_not` = AND-NOT.
    """

    model_config = ConfigDict(extra="forbid")

    must: list[FilterCondition] = []
    should: list[FilterCondition] = []
    must_not: list[FilterCondition] = []

    def fields(self) -> set[str]:
        return {c.field for group in (self.must, self.should, self.must_not) for c in group}


class SearchSpec(BaseModel):
    """One independent top-K search to compute in a `nova-bf compute` run —
    its own vector_type, metric, k, corpus_batch_size and (optional) filter,
    scored and top-K'd independently of every other spec in
    `BruteForceConfig.searches` (NOT fused into one hybrid score). Multiple
    specs sharing a run still share corpus file IO/decode (see compute.py) —
    that sharing is the whole point of listing several here instead of
    running `nova-bf compute` once per spec.

    `name` becomes part of the output filename (see `nova_bf.results`), so
    every spec in a run needs a distinct one.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    k: int = 1000
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    vector_type: Literal["dense", "sparse"] = "dense"
    corpus_batch_size: int | None = None
    filter: Filter | None = None

    @model_validator(mode="after")
    def _no_sparse_euclidean(self) -> "SearchSpec":
        if self.vector_type == "sparse" and self.metric == "euclidean":
            raise ValueError(
                f"search '{self.name}': metric='euclidean' is not supported with "
                "vector_type='sparse' (sparse retrieval only ever uses dot/cosine) "
                "— use 'dot' or 'cosine'"
            )
        return self

    @model_validator(mode="after")
    def _name_is_filename_safe(self) -> "SearchSpec":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(
                f"search name '{self.name}' must be non-empty and match "
                "[A-Za-z0-9_-]+ (it becomes part of the output filename)"
            )
        return self


class BruteForceConfig(BaseModel):
    # allow extra top-level keys (e.g. a `resources:` block for `nova dist`).
    model_config = ConfigDict(extra="allow")

    corpus: CorpusConfig
    queries: QueriesConfig
    output: OutputConfig
    params: ParamsConfig = ParamsConfig()
    # One or more independent top-K searches to compute in ONE corpus
    # read/decode pass — e.g. dense-unfiltered AND sparse-filtered against the
    # same corpus in a single `nova-bf compute` run (see compute.py's per-file
    # vector_type fan-out). Always required, even for a single search — a
    # config always says explicitly which search(es) it runs.
    searches: list[SearchSpec]

    @model_validator(mode="after")
    def _validate_searches(self) -> "BruteForceConfig":
        if not self.searches:
            raise ValueError("`searches` must list at least one SearchSpec")
        names = [s.name for s in self.searches]
        if len(set(names)) != len(names):
            raise ValueError(f"`searches` names must be unique, got {names}")
        return self


def load_config(path: str) -> BruteForceConfig:
    with open(path) as f:
        raw = expand_env(f.read())
    return BruteForceConfig.model_validate(yaml.safe_load(raw))
