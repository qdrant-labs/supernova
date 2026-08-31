"""Shared result schema + output naming for compute and merge."""

from __future__ import annotations

import hashlib
import json
import logging

import pyarrow as pa

from nova_bf.config import BruteForceConfig, SearchSpec

# Reserved output columns; everything else in a result row is carried payload.
# `hit_tie` appears only on a SHARDED run's partials, and only when `merge`
# cannot apply the tie-break rule from `hit_ids` alone — see `build_result_table`.
RESERVED = ("query_id", "hit_ids", "hit_scores", "hit_tie")

# Schema-metadata key recording which tie-break rule produced a result. `merge`
# refuses partials whose rule disagrees with the config it was handed: editing
# `params.tiebreak` between compute and merge would otherwise silently reduce
# ties by a rule the partials were never built for.
TIEBREAK_KEY = b"nova_bf_tiebreak"

# Schema-metadata keys identifying WHICH RUN produced a partial.
RUN_KEY = b"nova_bf.run_fingerprint"
CONFIG_KEY = b"nova_bf.config_fingerprint"
NUM_JOBS_KEY = b"nova_bf.num_jobs"
JOB_RANK_KEY = b"nova_bf.job_rank"


def queries_stem(queries_path: str) -> str:
    base = queries_path.rstrip("/").split("/")[-1]
    return base[:-8] if base.endswith(".parquet") else base


def result_name(cfg: BruteForceConfig, spec: SearchSpec) -> str:
    return f"bf_{queries_stem(cfg.queries.path)}_{spec.name}_k{spec.k}.parquet"


def partial_dir(cfg: BruteForceConfig, spec: SearchSpec) -> str:
    return f"_bf_partial_{queries_stem(cfg.queries.path)}_{spec.name}_k{spec.k}"


# Arrow spells some float widths in ways that read badly in a metadata dump.
_DTYPE_NAMES = {"halffloat": "float16", "float": "float32", "double": "float64"}


def vector_dtype(schema: pa.Schema, column: str) -> str:
    """What a vector column's VALUES are stored as, e.g. `float16`.

    Reported because nova-bf upcasts everything to float32 before scoring, so
    the loaders cannot tell you this and the scored values do not reveal it —
    a corpus stored `list<halffloat>` and one stored `list<float>` produce
    byte-identical ground truth for the same vectors. It is the storage that
    differs, and the storage is what a consumer has to match.

    Returns `""` when the column is absent or has an unexpected shape, since
    this is provenance, not validation — a wrong guess here must not fail a
    run that would otherwise succeed.
    """
    try:
        t = schema.field(column).type
    except KeyError:
        return ""
    # dense/multivector: list<value>, possibly nested. sparse: struct with a
    # `values` list. Walk to whatever scalar sits at the bottom.
    for _ in range(3):
        if pa.types.is_struct(t):
            names = [f.name for f in t]
            if "values" not in names:
                return ""
            t = t.field(names.index("values")).type
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            t = t.value_type
        else:
            break
    name = str(t)
    return _DTYPE_NAMES.get(name, name)


def config_identity(cfg: BruteForceConfig, spec: SearchSpec) -> str:
    """
    A hash over everything in the CONFIG that decides this search's results.
    `merge` recomputes this from the config IT was handed and compares it with
    what the partials carry.

    Deliberately excludes anything that changes only speed or layout —
    `io_workers`, batch sizes, `merge_batch_size`, `output.path`.

    """
    fields = {
        "search": spec.name or "",
        "vector_type": spec.vector_type,
        "metric": spec.metric,
        "k": spec.k,
        "filter": (
            None if spec.filter is None
            else spec.filter.model_dump(mode="json", exclude_defaults=True)
        ),
        "rows": None if spec.rows is None else spec.rows.model_dump(mode="json"),
        "corpus_path": cfg.corpus.path,
        "corpus_include": cfg.corpus.include,
        "corpus_exclude": cfg.corpus.exclude,
        "corpus_id_column": cfg.corpus.id_column,
        "corpus_column": (
            cfg.corpus.sparse_column if spec.vector_type == "sparse"
            else cfg.corpus.multivector_column if spec.vector_type == "multivector"
            else cfg.corpus.dense_column
        ),
        "queries_path": cfg.queries.path,
        "queries_id_column": cfg.queries.id_column,
        "queries_column": (
            cfg.queries.sparse_column if spec.vector_type == "sparse"
            else cfg.queries.multivector_column if spec.vector_type == "multivector"
            else cfg.queries.dense_column
        ),
        "allow_tf32": cfg.params.allow_tf32,
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode()
    ).hexdigest()


def run_identity(
    config_sha: str,
    corpus_sha: str | None,
    num_jobs: int | None,
    partial_slice: bool,
    tiebreak: str,
) -> str:
    """A Content-derived hash identifying the RUN a partial belongs to.
    """
    payload = json.dumps(
        {
            "config": config_sha,
            "corpus": corpus_sha,
            "num_jobs": num_jobs,
            "partial_slice": partial_slice,
            "tiebreak": tiebreak,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def provenance(
    cfg: BruteForceConfig,
    spec: SearchSpec,
    dtypes: dict[str, str] | None = None,
    corpus_sha: str | None = None,
    num_jobs: int | None = None,
    job_rank: int | None = None,
    partial_slice: bool = False,
    run_sha: str | None = None,
) -> dict[bytes, bytes]:
    """How this ground truth was computed, for the parquet schema metadata.

    A result file is consumed long after the run — by `nova storm`, by a
    recall evaluation, by whoever inherits the artifact — and nothing else in
    it records how it was produced. Without this, questions like "was this
    scored in exact fp32 or TF32?" can only be answered by inspecting the
    stored values' bit patterns, which is a genuinely expensive way to look up
    a boolean.

    `allow_tf32` is the field that matters most: it is off by default so scores
    stay bit-exact f32, and a consumer comparing these scores to a live engine's
    needs a ~3e-4 looser tolerance when it was on. `corpus_dtype`/`queries_dtype`
    are the next most useful: nova-bf upcasts everything to float32 before
    scoring, so nothing in the output otherwise reveals that the corpus was
    stored `float16` or the queries were bf16-valued — facts that decide whether
    a consumer can reproduce these vectors at all. The rest identifies WHICH
    ground truth this is — corpus, queries, metric, k — so a stale file can be
    told from a current one.

    Values are strings because parquet metadata is `bytes -> bytes`; consumers
    should parse rather than assume types.
    """
    meta = {
        "nova_bf.search": spec.name or "",
        "nova_bf.vector_type": spec.vector_type,
        "nova_bf.metric": spec.metric,
        "nova_bf.k": str(spec.k),
        "nova_bf.filtered": str(spec.filter is not None).lower(),
        "nova_bf.rows_subset": str(spec.rows is not None).lower(),
        "nova_bf.corpus_path": cfg.corpus.path,
        "nova_bf.queries_path": cfg.queries.path,
        # Scoring precision. TF32 is ~3e-4 relative error; exact f32 otherwise.
        "nova_bf.allow_tf32": str(cfg.params.allow_tf32).lower(),
        # The column the corpus was scored from — its parquet dtype is a
        # property of those files, and nova-bf upcasts to f32 before scoring,
        # so the path plus this name is what pins the vector space down.
        "nova_bf.corpus_column": (
            cfg.corpus.sparse_column if spec.vector_type == "sparse"
            else cfg.corpus.multivector_column if spec.vector_type == "multivector"
            else cfg.corpus.dense_column
        )
        or "",
        # What the OUTPUT holds. Fixed by `build_result_table`, recorded anyway
        # so a consumer reads it rather than assuming it.
        "nova_bf.scores_dtype": "float32",
    }
    # Storage dtypes of the vectors actually scored. Absent rather than guessed
    # when the caller could not determine them — an empty value would read as a
    # claim, a missing key reads as "unknown".
    for key in ("corpus_dtype", "queries_dtype"):
        value = (dtypes or {}).get(key)
        if value:
            meta[f"nova_bf.{key}"] = value
    out = {k.encode(): v.encode() for k, v in meta.items()}
    # Which tie-break rule decided the exact ties in these rows. Stamped HERE,
    # rather than where the table is built, because there are two builders — a
    # partial/single-node result in `compute` and the reduced artifact in
    # `merge` — and only one of them used to do it, so a sharded run (the only
    # kind that produces the artifacts anyone ships) lost the record.
    out[TIEBREAK_KEY] = cfg.params.tiebreak.encode()
    # Which run this came from
    config_sha = config_identity(cfg, spec)
    out[CONFIG_KEY] = config_sha.encode()
    out[RUN_KEY] = (
        run_sha
        or run_identity(
            config_sha, corpus_sha, num_jobs, partial_slice, cfg.params.tiebreak
        )
    ).encode()
    # Sharded runs only: absent on a single-node result, which has no ranks.
    # Together these let `merge` check the rank set is exactly 0..num_jobs-1,
    # which is how a MISSING rank is caught — a partial count that is uniform
    # across searches (all that was checked before) is exactly what a rank that
    # died before writing any of its outputs leaves behind.
    if num_jobs is not None:
        out[NUM_JOBS_KEY] = str(num_jobs).encode()
    if job_rank is not None:
        out[JOB_RANK_KEY] = str(job_rank).encode()
    return out


def build_result_table(
    query_ids: list[str],
    payload: dict[str, list],
    hit_ids: list[list[str]],
    hit_scores: list[list[float]],
    metadata: dict[bytes, bytes] | None = None,
    hit_tie: list[list[int]] | None = None,
) -> pa.Table:
    data: dict = {"query_id": pa.array(query_ids, pa.string())}
    for col, vals in payload.items():
        data[col] = pa.array(vals)
    # `compute` hands these in as ready-built Arrow `ListArray`s — it decodes
    # a whole (n_q, k) block at once rather than a Python list per query — but
    # `merge` and the tests still pass lists of lists, so accept both rather
    # than forcing either caller to convert.
    def _list_col(v, ty):
        return v if isinstance(v, (pa.Array, pa.ChunkedArray)) else pa.array(v, pa.list_(ty))

    data["hit_ids"] = _list_col(hit_ids, pa.string())
    data["hit_scores"] = _list_col(hit_scores, pa.float32())
    if hit_tie is not None:
        data["hit_tie"] = _list_col(hit_tie, pa.int64())
    table = pa.table(data)
    # No tie-break stamp here: `provenance` carries it, so both this builder and
    # `merge`'s get it from one place.
    return table.replace_schema_metadata(metadata) if metadata else table


def warn_if_short(short: int, total: int, k: int, search_name: str, logger: logging.Logger) -> None:
    """Log if any query's FINAL top-K came out shorter than k. Not an error —
    hit_ids/hit_scores are already correctly truncated (see the `-inf` sentinel
    handling in compute.py) — just a signal that the corpus, or `filter` if one
    is configured, didn't have k matches for some queries, so this ground truth
    is smaller than requested rather than wrong.

    Takes pre-computed counts (not a hit_ids list) so the streaming `merge`, which
    never materializes a full hit_ids list, can tally `short` per batch and still
    share this one warning. `search_name` identifies which search this is about —
    a run can compute several (see SearchSpec), each with its own short-count.
    """
    if short:
        logger.warning(
            "search=%r: %d/%d quer%s returned fewer than k=%d hits — the corpus "
            "(after any `filter`) didn't have enough matches for them",
            search_name, short, total, "y" if short == 1 else "ies", k,
        )
