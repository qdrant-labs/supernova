"""Typed brute-force config (pydantic), with `${VAR}` env expansion.

    corpus:   the embedded parquets to search over (local dir / s3:// prefix)
    queries:  the query embeddings (a parquet file or dir)
    output:   where results land (local dir / s3:// prefix)
    params:   k, distance metric
    filter:   optional corpus-side payload predicate restricting eligible
              neighbors (see Filter) — each condition's `field` is itself the
              declaration of which corpus column to read for it.
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
    # of dense_column when params.metric's sibling `vector_type` is "sparse" —
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
    model_config = ConfigDict(extra="forbid")

    k: int = 1000
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    # "sparse" reads corpus.sparse_column/queries.sparse_column (a
    # struct<indices, values> column) instead of the dense_column, and scores
    # via sparse-corpus × dense-query-vocab matmul instead of a dense matmul.
    vector_type: Literal["dense", "sparse"] = "dense"
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
    # Score a corpus file against the queries in row-batches of this many rows,
    # instead of all at once. The per-file score matrix is (n_queries × rows), so a
    # big file (or many queries) can OOM the GPU; batching bounds it to
    # (n_queries × corpus_batch_size). None (default) → whole file in one matmul
    # (fastest, current behavior). A batch below k is pointless — it can't fill the
    # top-K and isn't smaller than the always-resident (n_q × k) state — so values
    # under k are raised to k with a warning.
    corpus_batch_size: int | None = None
    # `merge` reduces the W per-rank partials in row-batches of this many queries,
    # streaming the result to disk so the full output never sits in RAM (that's what
    # let the old merge OOM at 1M queries). Peak host memory is ~(this × W × k)
    # candidate slots. None (default) → auto: sized so the working set stays near
    # ~20M candidate slots regardless of W and k, floored at 1 and capped at the
    # query count. Set it explicitly to trade memory for fewer, larger batches.
    merge_batch_size: int | None = None
    # `merge`: when the partials live on S3, first bulk-download them to local disk
    # (io_workers threads, whole-object copies) and merge from there, instead of
    # streaming each batch over S3 in lockstep. Trades local disk (~sum of partial
    # sizes) for a one-shot parallel download at full NIC speed + latency-free reads
    # during the reduce — worth it on a beefy box with fast NVMe. No effect when the
    # partials are already local. Default off (a laptop controller may lack the disk).
    merge_prefetch: bool = False

    @model_validator(mode="after")
    def _no_sparse_euclidean(self) -> "ParamsConfig":
        if self.vector_type == "sparse" and self.metric == "euclidean":
            raise ValueError(
                "metric='euclidean' is not supported with vector_type='sparse' "
                "(sparse retrieval only ever uses dot/cosine) — use 'dot' or 'cosine'"
            )
        return self


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
        # "" is reserved for the single implicit spec BruteForceConfig.effective_specs()
        # synthesizes from the legacy flat params/filter shape — BruteForceConfig's own
        # validator rejects an empty name inside a user-supplied `searches` list, so a
        # blank name can only ever reach here via that synthesis, never from parsed YAML.
        if self.name and not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(
                f"search name '{self.name}' must match [A-Za-z0-9_-]+ (it becomes "
                "part of the output filename)"
            )
        return self


class BruteForceConfig(BaseModel):
    # allow extra top-level keys (e.g. a `resources:` block for `nova dist`).
    model_config = ConfigDict(extra="allow")

    corpus: CorpusConfig
    queries: QueriesConfig
    output: OutputConfig
    params: ParamsConfig = ParamsConfig()
    filter: Filter | None = None
    # Independent top-K searches to compute in ONE corpus read/decode pass — e.g.
    # dense-unfiltered AND sparse-filtered against the same corpus in a single
    # `nova-bf compute` run (see compute.py's per-file vector_type fan-out).
    # Omit (default) for today's single flat-params/filter behavior; the output
    # filename is then identical to a run with no `searches` at all.
    searches: list[SearchSpec] | None = None

    @model_validator(mode="after")
    def _validate_searches(self) -> "BruteForceConfig":
        if self.searches is None:
            return self
        if not self.searches:
            raise ValueError(
                "`searches` must not be empty — omit the key entirely for the "
                "single-search (`params`/`filter`) default"
            )
        names = [s.name for s in self.searches]
        if any(not n for n in names):
            raise ValueError("every entry in `searches` must set a non-empty `name`")
        if len(set(names)) != len(names):
            raise ValueError(f"`searches` names must be unique, got {names}")
        if self.filter is not None:
            raise ValueError(
                "top-level `filter` is ignored when `searches` is set — put each "
                "search's filter inside its own `searches[].filter` instead"
            )
        # Every field name shared between ParamsConfig and SearchSpec (currently
        # k/metric/vector_type/corpus_batch_size) has moved from `params` onto
        # each `searches[]` entry — a leftover non-default value here (e.g. a
        # config migrated from the legacy single-search shape without deleting
        # `params.k`) would otherwise be silently ignored: effective_specs()
        # never reads these fields once `searches` is set, so every entry that
        # doesn't repeat the value gets SearchSpec's own default instead, with
        # no error. Derived from the two models' actual field sets (not a
        # hand-maintained tuple) so a future field added to both under the same
        # name is caught automatically instead of needing a third place to stay
        # in sync. Compared against a real `ParamsConfig()` instance rather than
        # pydantic's `FieldInfo.default` — the latter silently becomes
        # `PydanticUndefined` (never equal to anything) if a field ever switches
        # to `default_factory=`, which would wrongly reject every config.
        default_params = ParamsConfig()
        shared_fields = sorted(set(ParamsConfig.model_fields) & set(SearchSpec.model_fields))
        stale_fields = [
            f for f in shared_fields
            if getattr(self.params, f) != getattr(default_params, f)
        ]
        if stale_fields:
            raise ValueError(
                f"`searches` is set, so params.{'/'.join(stale_fields)} "
                f"{'is' if len(stale_fields) == 1 else 'are'} ignored, not applied as a "
                "default — remove it from `params` (only io_workers/io_thread_count/"
                "merge_batch_size/merge_prefetch still apply there) and set it on each "
                "`searches[]` entry that needs it instead"
            )
        return self

    def effective_specs(self) -> list["SearchSpec"]:
        """The searches this run computes: `searches` if set, else a single
        spec synthesized from the legacy flat params/filter fields (name=""),
        so compute.py/merge.py never special-case the single-search path."""
        if self.searches is not None:
            return self.searches
        return [SearchSpec(
            name="",
            k=self.params.k,
            metric=self.params.metric,
            vector_type=self.params.vector_type,
            corpus_batch_size=self.params.corpus_batch_size,
            filter=self.filter,
        )]


def load_config(path: str) -> BruteForceConfig:
    with open(path) as f:
        raw = expand_env(f.read())
    return BruteForceConfig.model_validate(yaml.safe_load(raw))
