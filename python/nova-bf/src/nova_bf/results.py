"""Shared result schema + output naming for compute and merge."""

from __future__ import annotations

import logging

import pyarrow as pa

from nova_bf.config import BruteForceConfig, SearchSpec

# Reserved output columns; everything else in a result row is carried payload.
RESERVED = ("query_id", "hit_ids", "hit_scores")


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


def provenance(
    cfg: BruteForceConfig,
    spec: SearchSpec,
    dtypes: dict[str, str] | None = None,
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
    return {k.encode(): v.encode() for k, v in meta.items()}


def build_result_table(
    query_ids: list[str],
    payload: dict[str, list],
    hit_ids: list[list[str]],
    hit_scores: list[list[float]],
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    data: dict = {"query_id": pa.array(query_ids, pa.string())}
    for col, vals in payload.items():
        data[col] = pa.array(vals)
    data["hit_ids"] = pa.array(hit_ids, pa.list_(pa.string()))
    data["hit_scores"] = pa.array(hit_scores, pa.list_(pa.float32()))
    table = pa.table(data)
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
