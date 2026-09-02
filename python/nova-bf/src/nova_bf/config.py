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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nova_bf.tokenize import tokenize
from nova_bf.dates import is_epoch_format, normalize_date_fields, parse_scalar_epoch_us

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
    # list<list<float32>> column (outer list = one doc, inner list = one D-dim
    # token vector) read instead of dense_column by a search whose
    # `vector_type` is "multivector" (ColBERT / late-interaction MaxSim) — the
    # same schema nova-embed writes (see nova_embed.storage.writer's
    # MULTIVECTOR_EMBEDDING_TYPE) and nova-load reads. Named to match the
    # dense_/sparse_ convention above; nova-embed itself names the column after
    # the output entry, so set this to that column when it differs.
    multivector_column: str = "multivector_embedding"
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
    # Payload columns that hold datetimes. Only a DECLARED field is treated as
    # a date — no type sniffing (a plain string column is never silently
    # reinterpreted). Each declared field is parsed to int64 epoch microseconds
    # right after a file is read (see nova_bf.dates / compute.py), so `range`
    # over it compares as numbers — value-for-value comparable to Qdrant's
    # DatetimeRange. Two accepted shapes:
    #   date_fields: [published_at]                 # rfc3339 (default)
    #   date_fields: {published_at: rfc3339,
    #                 crawl_day: "%Y%m%d",          # any strptime pattern
    #                 ingested_at: epoch_s}         # already-numeric epoch
    # A static `range` bound on a date field is written as an RFC-3339 string
    # in the search filter (e.g. `gte: "2013-01-01T00:00:00Z"`) and parsed to
    # epoch µs at config load; a string bound on a NON-date field is rejected.
    date_fields: list[str] | dict[str, str] = []


class QueriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    dense_column: str = "dense_embedding"
    sparse_column: str = "sparse_embedding"
    # list<list<float32>> query column for a multivector search — see
    # CorpusConfig.multivector_column.
    multivector_column: str = "multivector_embedding"
    # If set, use this column as the query id verbatim; otherwise derive
    # make_point_id(queries_file_key, row) — same scheme as the corpus.
    id_column: str | None = None
    # Columns to carry from the queries file into each output row.
    payload_fields: list[str] = []
    # Query columns that hold datetimes — same declaration/format rules as
    # `CorpusConfig.date_fields`. These are the columns a `range_from_query`
    # draws its per-query bounds from when the corpus field is a date; parsed to
    # epoch µs so each query's bound compares against the (also-µs) corpus date.
    date_fields: list[str] | dict[str, str] = []

    @model_validator(mode="after")
    def _payload_fields_are_not_reserved(self) -> "QueriesConfig":
        """A carried column cannot be named like an output column.
        """
        # Imported here, not at module scope: `results` imports this module.
        from nova_bf.results import RESERVED

        clash = [c for c in self.payload_fields if c in RESERVED]
        if clash:
            raise ValueError(
                f"queries.payload_fields may not use the reserved output column "
                f"name(s) {clash}; nova-bf writes {list(RESERVED)} itself, so "
                "these would be silently overwritten. Rename the column(s) in the "
                "queries file, or drop them from payload_fields."
            )
        return self


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
    # file_size must fit host memory or the box OOMs. Keep it modest; the real S3
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
    # PyArrow CPU threads used for Parquet decoding/decompression, separate from
    # the I/O thread pool. 0 uses os.cpu_count() rather than PyArrow's default,
    # which may inherit OMP_NUM_THREADS and unintentionally serialize decoding.
    cpu_thread_count: int = 0
    # Opt-in: read each large corpus/queries parquet as MANY concurrent byte
    # ranges before parsing (the `aws s3 cp` strategy). Files written with a
    # large flush threshold hold ~ONE row group, so a single-column read
    # degenerates into one sequential stream that `io_thread_count` cannot
    # speed up; this restores full-bandwidth reads regardless of the file's
    # internal layout, on s3 and POSIX roots alike, at the cost of one file's
    # raw bytes of extra host RAM while that file is parsed. OFF by default.
    io_ranged_get: bool = False
    # Bounds the per-file score matrix (`queries × rows`) for dense/sparse
    # searches respectively — a big file (or large query set) can otherwise
    # OOM the GPU; set this to score in row-batches instead of the whole file
    # at once. None (default) = whole file (or, if every search of that
    # vector_type has an active filter, the UNION of their surviving rows —
    # see compute.py's `_union_keep`) in one matmul. Lives here (one value
    # per vector_type, run-wide) rather than per-search: every search of a
    # given vector_type ends up sharing one GPU pass over the corpus anyway
    # (see compute.py), so a per-search value would just be resolved down to
    # this same shared number regardless — this makes that explicit instead
    # of implicit. `SearchSpec.rows` does NOT change that: the shared score
    # matrix spans the vector_type's whole query-row union and each search
    # slices its own rows out of it AFTER scoring, so what this bounds is
    # still one matrix per vector_type, not one per search.
    # A value below some search's own `k` is never raised —
    # only warned about (that search just needs extra merge rounds to fill
    # its own top-K, at no extra memory cost to anyone).
    #
    # When every search of a vector_type has an active filter, this is now
    # the ONLY memory bound on that vector_type's shared pass: the union of
    # several large, mostly-disjoint filters can be nearly as big as the
    # whole file, transferred/scored in one shot if this is left at its
    # None default — set it explicitly if you have several such filters and
    # relied on each one's own (typically much smaller) surviving-row count
    # bounding memory implicitly.
    # `gt=0`: a batch size of 0 or negative isn't just useless, it's actively
    # wrong — `range(0, n_rows, step)` with a non-positive `step` is EMPTY, so
    # every file's batch loop would silently skip all rows, no exception, no
    # partial results, just an empty top-K for every query. Reject it here at
    # config-load time (the system boundary) rather than let it manifest as a
    # silent all-empty run downstream.
    dense_batch_size: int | None = Field(default=None, gt=0)
    sparse_batch_size: int | None = Field(default=None, gt=0)
    # The multivector (MaxSim) implementations (torch and triton_reduce) have
    # TWO memory axes: their intermediate is `(block_query_tokens ×
    # corpus_doc_tokens)`, so both corpus rows and queries are tiled to bound
    # the peak.
    #   multivector_batch_size  = docs per corpus-row slice (bounds doc-token
    #                             count per slice; None = whole file at once)
    #   multivector_query_block = queries per query-axis tile — a WHOLE number
    #                             of queries per block, never splitting one
    #                             query's tokens across blocks (None = all
    #                             queries at once)
    # `gt=0` for the same reason dense/sparse have it (a non-positive step
    # silently empties the batch loop — see dense_batch_size).
    multivector_batch_size: int | None = Field(default=None, gt=0)
    multivector_query_block: int | None = Field(default=None, gt=0)
    # MaxSim implementation. `torch` is the established matmul + segmented
    # reduction path. `triton_reduce` retains the cuBLAS FP32 matmul but
    # fuses the ragged max/sum reduction (requires CUDA + Triton, fails
    # clearly otherwise). `auto` selects `triton_reduce` on a compatible CUDA
    # run and otherwise falls back to torch. (A fully fused `triton` backend
    # was removed after measuring ~4x slower than triton_reduce — see
    # docs/brute-force/multivector-maxsim.md.)
    multivector_kernel: Literal["torch", "triton_reduce", "auto"] = "torch"
    # Target element count for each materialized `(block_query_tokens ×
    # slice_doc_tokens)` score matrix. It still derives either item-count knob
    # left unset, but is also enforced against the ACTUAL ragged query/document
    # offsets: explicit batch/query counts become upper bounds, not permission
    # to exceed this token-product budget. One over-budget document is processed
    # alone so every row still makes progress.
    multivector_token_budget: int | None = Field(default=None, gt=0)
    # CUDA-only ordered transfer prefetch for multivector slices. The next
    # packed ragged slice is copied through pinned host staging on a dedicated
    # CUDA stream while the current slice's GEMM/reduction/top-k work runs.
    # CPU and non-multivector paths remain synchronous.
    multivector_double_buffer: bool = False
    # Allow TF32 tensor-core matmuls on Ampere+ GPUs (CUDA only — a no-op on
    # CPU). OFF by default so ground truth stays bit-for-bit f32, matching
    # Qdrant's f32 scoring exactly. When on, the score matmul runs in TF32
    # (10-bit mantissa): measured on an A10G it makes the multivector matmul
    # ~1.75x faster (the matmul is ~66% of GPU score time, so ~1.4x on the
    # score path), with ~3e-4 median relative error — ranking-preserving in
    # testing (top-100 overlap 100/100 vs f32) and the live-Qdrant MaxSim
    # parity test still passes with it enabled. It is NOT bit-exact, so enable
    # it only for a run where you've confirmed that ~3e-4-scale score error
    # can't perturb the recall numbers you compute against this GT (e.g. no
    # pathological score ties at your k). Affects every vector_type's dense
    # matmul (dense scoring, multivector MaxSim); sparse SpMM is unaffected.
    allow_tf32: bool = False
    # Merge partials in row batches and stream results to disk to bound memory.
    # `None` auto-sizes for ~20M candidate slots; set explicitly to trade more
    # memory for larger batches and fewer parquet row groups. Merge warns when 
    # your value is above its own target.
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

    # Which of two EXACTLY-tied candidates wins.
    # Neither makes SCORES reproducible across batch sizes — re-tiling a matmul
    # changes its reduction order, which can change whether a tie exists at all.
    tiebreak: Literal["ordinal", "id"] = "ordinal"


# A single scalar payload value, as it would appear in a corpus column.
MatchValue = str | int | float | bool


class RangeCondition(BaseModel):
    """Numeric bounds, combinable like Qdrant's Range (e.g. `gte` + `lt` together).

    Frozen: makes this — and, transitively, `FilterCondition`/`Filter`, which
    contain it — hashable via pydantic's auto-generated `__hash__` (a tuple of
    field values), so a `Filter` can be used directly as a dict key wherever
    specs need grouping/deduping by identical filter (see compute.py). Also
    enforces the invariant that a `Filter` is never mutated after grouping,
    previously just assumed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> "RangeCondition":
        if all(v is None for v in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("range condition needs at least one of gt/gte/lt/lte")
        return self


class RangeFromQuery(BaseModel):
    """Per-query numeric bounds — same shape as `RangeCondition`, but each
    bound names a QUERIES column supplying THAT query's own value for the
    bound, instead of a literal number (e.g. `lt: max_budget` means "each
    query's own ceiling comes from its own `max_budget` column"). Evaluated
    via a broadcast comparison (`compute.py`), exactly as cheap as a literal
    `RangeCondition` — there's no cardinality/factorization cost the way
    per-query `match` lists have.

    Deliberately does not mix a literal bound and a per-query bound in one
    condition (no `gt: 0` alongside `lt: max_budget` here) — express that as
    two separate conditions in the same `must`/`should`/`must_not` list
    instead (a static `range: {gt: 0}` plus this `range_from_query: {lt:
    max_budget}`), which the existing AND/OR combination logic already
    handles for free."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gt: str | None = None
    gte: str | None = None
    lt: str | None = None
    lte: str | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> "RangeFromQuery":
        if all(v is None for v in (self.gt, self.gte, self.lt, self.lte)):
            raise ValueError("range_from_query needs at least one of gt/gte/lt/lte")
        return self


class FilterCondition(BaseModel):
    """One field predicate: keyword-style equality (`match`), numeric bounds
    (`range`), full-text (`match_text`), or a per-query variant of any of the
    three (`match_from_query`/`range_from_query`/`match_text_from_query`),
    which pulls its comparison value(s) from a column in the QUERIES file
    instead of a literal in this config — so two different queries in the
    same search can each be restricted to a different corpus subset (e.g.
    each scoped to its own tenant, budget, or search phrase). See
    `nova_bf.filters`/`compute.py` for how a per-query condition's mask ends
    up shaped `(n_queries, rows)` instead of `(rows,)`, and how that composes
    with everything else.

    `match` takes a scalar (equality) or a list (matches any of them —
    Qdrant's MatchAny); `match_from_query` does the same but per query, via a
    queries column holding either a scalar or a list per row. `match_text`
    requires every token of the string to appear as a token of the field —
    both sides run through `nova_bf.tokenize` (split on non-alphanumeric, lowercase),
    the same semantics as Qdrant's full-text `MatchText` against a `word`
    tokenizer index with `lowercase: true` (order-independent, AND of tokens);
    `match_text_from_query` is the same, with each query's own phrase read
    from a queries column — the one per-query variant with a real, different
    cost profile (see `filters._match_text_from_query_mask`'s docstring).
    Exactly one of the six must be set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The corpus column this condition reads and matches against — this is the
    # only place a filtered field is named; nothing else needs to declare it,
    # so `compute` reads exactly (and only) the columns the filter references.
    field: str
    # tuple, not list: lists aren't hashable, and this needs to be for Filter
    # (which contains this) to be usable as a dict key — see RangeCondition's
    # docstring. Pydantic coerces a YAML/list literal (`match: [1, 2, 3]`)
    # into a tuple automatically at validation time.
    match: MatchValue | tuple[MatchValue, ...] | None = None
    range: RangeCondition | None = None
    match_text: str | None = None
    # Per-query variants — see this class's docstring. Each names a QUERIES
    # column (not a corpus column); `field` above still names the CORPUS
    # column every one of these compares against.
    match_from_query: str | None = None
    range_from_query: RangeFromQuery | None = None
    match_text_from_query: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "FilterCondition":
        options = (
            self.match, self.range, self.match_text,
            self.match_from_query, self.range_from_query, self.match_text_from_query,
        )
        if sum(v is not None for v in options) != 1:
            raise ValueError(
                f"filter condition on `{self.field}` must set exactly one of "
                "`match`, `range`, `match_text`, `match_from_query`, "
                "`range_from_query`, or `match_text_from_query`"
            )
        return self

    @model_validator(mode="after")
    def _match_text_has_tokens(self) -> "FilterCondition":
        if self.match_text is not None and not tokenize(self.match_text):
            raise ValueError(
                f"filter condition on `{self.field}` has a `match_text` with no "
                "alphanumeric tokens — it would match nothing (match_text is "
                "tokenized like Qdrant's word tokenizer: split on "
                "non-alphanumeric characters, lowercased)"
            )
        return self

    def is_per_query(self) -> bool:
        """Does this ONE condition vary per query? The per-condition analog of
        `Filter.is_per_query()` — used to order a group's static conditions
        before its per-query ones during evaluation (`filters._static_first`),
        and to tell an exact static leaf from a per-query one in
        `compute._row_union_from_gpu_leaves`."""
        return (
            self.match_from_query is not None
            or self.range_from_query is not None
            or self.match_text_from_query is not None
        )


class Filter(BaseModel):
    """A corpus-side payload filter, shaped like Qdrant's own filter (must/should/must_not).

    Restricts which corpus points are eligible neighbors for every query in the
    run — it does not touch queries themselves, same as a Qdrant search filter.
    `must` = AND, `should` = OR-at-least-one, `must_not` = AND-NOT.

    Frozen and hashable (see `RangeCondition`'s docstring) — usable directly
    as a dict key wherever specs need grouping/deduping by identical filter
    (see compute.py's `spec_filter`/`_union_keep`/`keeps`), via
    pydantic's own `__eq__`/`__hash__`, with no separate canonicalization
    scheme needed: two `Filter`s are "the same" iff Python's own `==` says so,
    which for `int`/`float`/`nan`/`inf` `match` values already has exactly the
    right semantics (`5 == 5.0`, no precision loss for large ints, `nan`
    never equal to itself, `inf`/`-inf` equal to themselves by sign).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The one place the three condition-group names are listed — `fields()`
    # walks them via this, not its own hardcoded tuple, so a future group
    # added here can't be missed by `fields()`.
    _CONDITION_GROUPS = ("must", "should", "must_not")

    # tuple, not list: needed for this model to be hashable (see
    # RangeCondition's docstring). Pydantic coerces a YAML/list literal
    # (`must: [...]`) into a tuple automatically at validation time.
    must: tuple[FilterCondition, ...] = ()
    should: tuple[FilterCondition, ...] = ()
    must_not: tuple[FilterCondition, ...] = ()

    def fields(self) -> set[str]:
        return {
            c.field for group in self._CONDITION_GROUPS for c in getattr(self, group)
        }

    def all_conditions(self) -> tuple["FilterCondition", ...]:
        """Every condition in this filter, across all three groups — the one
        place a caller that needs to walk every leaf (regardless of which
        group it's in) does so via `_CONDITION_GROUPS`, not its own
        hardcoded `(*f.must, *f.should, *f.must_not)`, same reason
        `fields()`/`query_fields()` do."""
        return tuple(c for group in self._CONDITION_GROUPS for c in getattr(self, group))

    def query_fields(self) -> set[str]:
        """Every QUERIES column referenced by a per-query condition anywhere
        in this filter (any group, any of `match_from_query`/
        `range_from_query`/`match_text_from_query`) — the query-side analog
        of `fields()`'s "no separate list to keep in sync" principle:
        `compute.py` reads exactly (and only) the queries columns some
        per-query condition actually references."""
        cols: set[str] = set()
        for group in self._CONDITION_GROUPS:
            for c in getattr(self, group):
                if c.match_from_query is not None:
                    cols.add(c.match_from_query)
                if c.range_from_query is not None:
                    cols.update(
                        v for v in (
                            c.range_from_query.gt, c.range_from_query.gte,
                            c.range_from_query.lt, c.range_from_query.lte,
                        )
                        if v is not None
                    )
                if c.match_text_from_query is not None:
                    cols.add(c.match_text_from_query)
        return cols

    def is_per_query(self) -> bool:
        """Does any condition in this filter vary per query? A per-query
        filter has no single EXACT row-subset to offer a shared batch grid
        (different queries need different corpus rows), but `compute.py`'s
        `run_compute` can still union-compact it via a cheap, safe
        over-approximation rather than always falling back to the whole
        file — see `_is_per_query`/`_row_union_from_gpu_leaves`."""
        return bool(self.query_fields())


class RowSelector(BaseModel):
    """Which QUERY rows a search owns — `SearchSpec.rows`.

    Without one, a spec covers every row in the queries file, so a run that
    unions several query sets into one file has to neutralize each spec's
    foreign rows with a match-nothing sentinel value.

    A selector names the rows directly instead, and the sentinels become
    unnecessary:

        rows: {column: query_set, isin: [filtered_text]}

    `column` is read from the queries file alongside the per-query filter
    columns, so it does NOT need to be listed in `queries.payload_fields`
    (though it usually is, to carry through to the output). Values are compared
    as strings, so `isin: ["1"]` matches an integer 1 in the source column.

    NOT bit-exact against a full-file run when the run's selectors leave some
    rows unowned. Specs of one vector_type share one query matrix built over
    the UNION of their `rows`; if that union is a strict subset of the file the
    matrix is shorter, the scoring matmul's query dimension changes with it,
    and BLAS accumulates in a different order — scores move by ~1 float32 ULP
    (~5e-7 relative), enough to swap two documents scored within that margin.
    Subsets that between them cover every row (the two-halves case `rows` was
    added for) keep the matrix full height and stay bit-exact. See
    docs/brute-force/overview.md; same class of caveat as
    `ParamsConfig.allow_tf32`, several orders of magnitude smaller.
    """

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    isin: list[str] = Field(min_length=1)


class SearchSpec(BaseModel):
    """One independent top-K search to compute in a `nova-bf compute` run —
    its own vector_type, metric, k, and (optional) filter, scored and top-K'd
    independently of every other spec in `BruteForceConfig.searches` (NOT
    fused into one hybrid score). Multiple specs sharing a run still share
    corpus file IO/decode (see compute.py) — that sharing is the whole point
    of listing several here instead of running `nova-bf compute` once per
    spec. GPU batching (`params.dense_batch_size`/`sparse_batch_size`) is a
    run-level knob, not a per-search one — see `ParamsConfig`.

    `name` becomes part of the output filename (see `nova_bf.results`), so
    every spec in a run needs a distinct one. Optional: if omitted, `Brute
    ForceConfig` derives one from `vector_type`/`metric` (plus `_filtered`
    when a filter is set) and disambiguates collisions automatically — see
    `_assign_default_names`. A lone `SearchSpec` can't do this itself since
    disambiguation needs to see every spec in the run's `searches` list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    k: int = 1000
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    vector_type: Literal["dense", "sparse", "multivector"] = "dense"
    filter: Filter | None = None
    # Which QUERY rows this search owns (see RowSelector). None = every row,
    # the historical behavior. `filter` is the CORPUS-side predicate; this is
    # the query-side one, and the two are independent.
    rows: RowSelector | None = None

    @model_validator(mode="after")
    def _no_euclidean_for_non_dense(self) -> "SearchSpec":
        # euclidean only makes sense on a single pooled vector per item. Sparse
        # retrieval only ever uses dot/cosine; multivector MaxSim is a sum of
        # per-token dot (or cosine) maxima — an L2 distance between two ragged
        # token SETS has no MaxSim analog (and Qdrant's multivector comparator
        # is dot/cosine MaxSim only).
        if self.vector_type in ("sparse", "multivector") and self.metric == "euclidean":
            raise ValueError(
                f"metric='euclidean' is not supported with vector_type="
                f"'{self.vector_type}' — use 'dot' or 'cosine'"
                + (f" (search {self.name!r})" if self.name else "")
            )
        return self

    @model_validator(mode="after")
    def _rows_not_multivector(self) -> "SearchSpec":
        # The multivector path carries its own ragged (total_tokens, D) matrix
        # plus a length-n_q+1 token-offset array (`MultiVectorQuery`); taking a
        # query-row subset means rebuilding those offsets, which the dense and
        # sparse paths don't need. Rejected explicitly rather than silently
        # ignored — a subset that didn't apply would produce a correct-looking
        # result over the WRONG query set.
        if self.rows is not None and self.vector_type == "multivector":
            raise ValueError(
                "`rows` is not supported with vector_type='multivector'"
                + (f" (search {self.name!r})" if self.name else "")
            )
        return self

    @model_validator(mode="after")
    def _name_is_filename_safe(self) -> "SearchSpec":
        # None (not yet assigned — see BruteForceConfig._assign_default_names)
        # is fine here; an explicitly-set name is validated immediately.
        if self.name is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError(
                f"search name '{self.name}' must be non-empty and match "
                "[A-Za-z0-9_-]+ (it becomes part of the output filename)"
            )
        return self


def _assign_default_names(specs: list[SearchSpec]) -> None:
    """Fill in `.name` for every spec that didn't set one explicitly, deriving
    a readable default from `vector_type`/`metric` (plus `_filtered` when a
    filter is set) and disambiguating any collision — with each other, or
    with an explicit name elsewhere in `specs` — by appending an incrementing
    numeric suffix. Runs once over the whole list (not as a per-SearchSpec
    validator): a spec can't see its siblings, and disambiguation needs to."""
    used = {s.name for s in specs if s.name is not None}
    for s in specs:
        if s.name is not None:
            continue
        base = f"{s.vector_type}_{s.metric}"
        if s.filter is not None and s.filter.fields():
            base += "_filtered"
        name, n = base, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        s.name = name


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
        _assign_default_names(self.searches)
        names = [s.name for s in self.searches]
        if len(set(names)) != len(names):
            raise ValueError(f"`searches` names must be unique, got {names}")
        return self

    @model_validator(mode="after")
    def _validate_tiebreak(self) -> "BruteForceConfig":
        """`tiebreak: id` orders ties by the point id, so there has to be one.
        """
        if self.params.tiebreak == "id" and not self.corpus.id_column:
            raise ValueError(
                "params.tiebreak='id' orders ties by the point id, so it needs "
                "`corpus.id_column`. Without one the ids are derived from a hash "
                "of (file, row), whose order says nothing the default "
                "params.tiebreak='ordinal' does not already say — use that."
            )
        return self

    @model_validator(mode="after")
    def _validate_date_fields(self) -> "BruteForceConfig":
        """Keep the corpus and queries date declarations consistent so a
        datetime bound is never silently compared against a non-datetime column
        (which would mean a nonsensical unit mismatch — raw µs vs. plain
        number). A `range_from_query` whose corpus `field` is a declared date
        field must draw every bound from a declared QUERIES date field, and
        vice versa."""
        corpus_dates = set(normalize_date_fields(self.corpus.date_fields))
        query_dates = set(normalize_date_fields(self.queries.date_fields))
        for s in self.searches:
            if s.filter is None:
                continue
            for cond in s.filter.all_conditions():
                # A declared date field is parsed to epoch µs, so it only makes
                # sense with range/range_from_query. Using it with an
                # equality/text predicate would compare/tokenize the raw integer
                # µs and silently match nothing — reject it at config time.
                if cond.field in corpus_dates and (
                    cond.match is not None
                    or cond.match_text is not None
                    or cond.match_from_query is not None
                    or cond.match_text_from_query is not None
                ):
                    raise ValueError(
                        f"corpus date field '{cond.field}' is used with a "
                        "match/match_text predicate — a declared date field is "
                        "parsed to epoch µs and only supports `range`/"
                        "`range_from_query`"
                    )
                if cond.range_from_query is None:
                    continue
                r = cond.range_from_query
                bound_cols = [v for v in (r.gt, r.gte, r.lt, r.lte) if v is not None]
                field_is_date = cond.field in corpus_dates
                for col in bound_cols:
                    col_is_date = col in query_dates
                    if field_is_date and not col_is_date:
                        raise ValueError(
                            f"range_from_query on date field '{cond.field}' draws "
                            f"its bound from '{col}', which is not declared in "
                            "queries.date_fields — declare it so its values are "
                            "parsed to epoch µs and comparable to the corpus date"
                        )
                    if col_is_date and not field_is_date:
                        raise ValueError(
                            f"range_from_query bound '{col}' is a declared queries "
                            f"date field but corpus field '{cond.field}' is not a "
                            "date field (corpus.date_fields) — comparing a datetime "
                            "bound against a non-datetime column"
                        )
        return self


def _normalize_static_date_bounds(data: dict) -> dict:
    """In-place: rewrite every static `range` string bound on a declared corpus
    date field to its int64 epoch-µs value (as a number), so `RangeCondition`'s
    plain `float` bounds validate unchanged and every downstream range path
    stays numeric. A string bound on a NON-date field is rejected here with a
    clear message instead of surfacing as a bare pydantic "not a number" error.
    Runs on the raw dict BEFORE validation, so no frozen model needs rebuilding.
    """
    corpus_dates = normalize_date_fields((data.get("corpus") or {}).get("date_fields"))
    for spec in data.get("searches") or []:
        filt = spec.get("filter")
        if not isinstance(filt, dict):
            continue
        for group in ("must", "should", "must_not"):
            for cond in filt.get(group) or []:
                if not isinstance(cond, dict):
                    continue
                rng = cond.get("range")
                if not isinstance(rng, dict):
                    continue
                field = cond.get("field")
                str_bounds = {b for b in ("gt", "gte", "lt", "lte")
                              if isinstance(rng.get(b), str)}
                if str_bounds and field not in corpus_dates:
                    raise ValueError(
                        f"filter on '{field}' has a string `range` bound "
                        f"({sorted(str_bounds)}) but '{field}' is not declared in "
                        "corpus.date_fields — declare it (optionally with a format) "
                        "to use datetime bounds, or use a numeric bound"
                    )
                if field not in corpus_dates:
                    continue  # ordinary numeric range on a non-date field: untouched
                fmt = corpus_dates[field]
                for b in ("gt", "gte", "lt", "lte"):
                    v = rng.get(b)
                    if v is None:
                        continue
                    # A string bound is always parsed; a NUMERIC bound is rescaled
                    # only for epoch_s/epoch_ms fields (whose column is likewise
                    # rescaled) — for rfc3339/strptime/epoch_us it's already the µs
                    # value the column uses, so it's left as-is.
                    if isinstance(v, str) or is_epoch_format(fmt):
                        rng[b] = float(parse_scalar_epoch_us(v, fmt))
    return data


def load_config(path: str) -> BruteForceConfig:
    with open(path) as f:
        raw = expand_env(f.read())
    data = yaml.safe_load(raw)
    data = _normalize_static_date_bounds(data)
    return BruteForceConfig.model_validate(data)
