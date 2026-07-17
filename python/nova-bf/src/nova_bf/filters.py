"""Evaluating a corpus-side `Filter` (see config.py) against one corpus file.

Runs once per file, over the whole file at once, via `pyarrow.compute` — O(rows),
independent of query count, UNLESS `filt` has a per-query condition
(`match_from_query`/`range_from_query`/`match_text_from_query`), in which case
`evaluate()`'s result is `(n_queries, rows)` instead of `(rows,)` — see its
docstring. A uniform (non-per-query) filter restricts which corpus points are
eligible neighbors for every query in the run, evaluated before scoring, not
per (query, row), same as a Qdrant search filter only ever touches the points
being searched; a per-query condition restricts each query independently, via
`compute.py`'s masked-fill path rather than row compaction (see there).

Text matching (`match_text`/`match_text_from_query`) uses Qdrant `word`-
tokenizer semantics — split on non-alphanumeric, lowercase each token, AND
of query tokens against each row's token set (`nova_bf.tokenize` /
`_token_row_masks`) — matching what a real Qdrant full-text index
(`tokenizer: word`, `lowercase: true`) computes, rather than the `\b`-regex
substring approximation this module previously used. Query strings and the
corpus column run through the SAME Arrow kernels, so the two sides agree on
tokens by construction (see `nova_bf.tokenize`'s module docstring).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nova_bf.config import Filter, FilterCondition
from nova_bf.tokenize import TOKEN_SPLIT_PATTERN, tokenize, tokenize_many

# Per-batch text-byte target for `_token_row_masks`: bounds every transient
# the scan materializes (the split token copy, its lowered copy, parent/code
# index arrays — a few multiples of the batch's text bytes) by BYTES, not
# rows, so huge-document corpora can't blow the bound; also keeps a batch's
# token count far below the 2^31 list-offset ceiling `split_pattern_regex`'s
# `list<large_string>` output still has even after the `large_string` cast.
_BATCH_TEXT_BYTES = 32 << 20


def _corpus_null_mask(table: pa.Table, field: str) -> np.ndarray:
    """`(rows,)`: which rows have a null value in `field`. Used by the
    per-query condition masks below to explicitly AND out corpus-side nulls
    — same "a null payload value never matches" convention the static path
    gets from `pc.fill_null(mask, False)`, made explicit here since a
    per-query condition builds its own numpy comparison directly rather than
    going through a pyarrow compute kernel that null-propagates for us."""
    return table[field].is_null().to_numpy(zero_copy_only=False)


def _match_from_query_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray],
) -> np.ndarray:
    """`(n_queries, rows)` — per-query equality (or MatchAny, if the queries
    column holds a list per row instead of a scalar). The scalar case is one
    broadcast comparison (`query_vals[:, None] == corpus_vals[None, :]`),
    reusing plain numpy `==` semantics — the same `MatchValue` rules
    (`5 == 5.0`, `nan != nan`, `True == 1`) `Filter`'s docstring already
    documents, for free. A null/missing per-query value already can't equal
    anything (an object-array `None` only equals another `None`, and a
    `nan` float never equals anything, including itself) — combined with
    `_corpus_null_mask` explicitly ANDed out below, a null on EITHER side
    never matches, symmetric with the static path's null handling."""
    corpus_vals = table[cond.field].to_numpy(zero_copy_only=False)
    query_vals = query_values[cond.match_from_query]
    not_null = ~_corpus_null_mask(table, cond.field)

    # Scan every value (not just the first) to decide scalar vs. MatchAny-list
    # encoding: a column can legitimately mix `None`/NaN (that query matches
    # nothing, same as a null scalar) with real lists, and checking only
    # index 0 would misclassify the whole column whenever THAT one value
    # happens to be null.
    if any(isinstance(v, (list, tuple, np.ndarray)) for v in query_vals):
        mask = _match_any_from_query_mask(corpus_vals, query_vals, not_null)
    else:
        mask = query_vals[:, None] == corpus_vals[None, :]
    return mask & not_null[None, :]


def _match_any_membership(vocab: np.ndarray, list_values) -> np.ndarray:
    """`(len(list_values), len(vocab))` boolean membership matrix: row `i`
    is `True` at column `j` iff `vocab[j]` is a member of `list_values[i]`
    (a `None`/NaN entry in `list_values` means that query matches nothing —
    every column stays `False`, same as an empty list, consistent with a
    null scalar query value never equaling anything). Shared by
    `_match_any_from_query_mask` (`vocab` = this batch's distinct CORPUS
    values) and `compute.py`'s GPU-native Front A path (`vocab` = the union
    of every query's OWN list values, built once at setup) — same
    build-a-position-dict-then-scatter algorithm either way, only the
    source of `vocab` differs."""
    pos = {v: i for i, v in enumerate(vocab)}
    membership = np.zeros((len(list_values), len(vocab)), dtype=bool)
    for i, values in enumerate(list_values):
        if values is None or (isinstance(values, (float, np.floating)) and values != values):
            continue
        idxs = [pos[v] for v in values if v in pos]
        if idxs:
            membership[i, idxs] = True
    return membership


def _match_any_from_query_mask(
    corpus_vals: np.ndarray, query_lists, not_null: np.ndarray,
) -> np.ndarray:
    """`(n_queries, rows)` — per-query MatchAny: query `q` keeps row `r` iff
    `corpus_vals[r]` is a member of `query_lists[q]`. Factorizes `corpus_vals`'
    DISTINCT values in this batch (cheap for the low-to-moderate-cardinality
    categorical fields this is meant for) rather than an O(n_queries * rows)
    membership test — cost tracks distinct-value count and per-query list
    sizes, not row count.

    `not_null` (from `_corpus_null_mask`) excludes null corpus rows from the
    distinct-value factorization BEFORE calling `np.unique` — `np.unique`'s
    sort can't compare `None` against a string/number, so a null mixed into
    an otherwise-typed object array raises `TypeError` if included. Null
    rows are left at `False` for every query (the caller's own `not_null`
    AND-out would catch this too, but getting it right here avoids a
    negative gather index accidentally wrapping to the LAST column)."""
    distinct = np.unique(corpus_vals[not_null]) if not_null.any() else np.empty(0, dtype=corpus_vals.dtype)
    pos = {v: i for i, v in enumerate(distinct)}
    corpus_idx = np.array([pos.get(v, -1) for v in corpus_vals], dtype=np.int64)

    membership = _match_any_membership(distinct, query_lists)

    result = np.zeros((len(query_lists), len(corpus_vals)), dtype=bool)
    valid_cols = corpus_idx >= 0
    result[:, valid_cols] = membership[:, corpus_idx[valid_cols]]
    return result


def _range_from_query_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray],
) -> np.ndarray:
    """`(n_queries, rows)` — per-query numeric bounds, broadcast per
    configured bound and ANDed together, same multi-bound logic
    `RangeCondition` already has. Just as cheap as scalar
    `match_from_query` — numeric comparisons broadcast natively, no
    cardinality/factorization concern the way list-valued `match_from_query`
    has. A null per-query bound value never matches for that query (`nan`
    compared to anything is `False`); corpus-side nulls are explicitly
    ANDed out the same way `_match_from_query_mask` does."""
    corpus_vals = table[cond.field].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    not_null = ~_corpus_null_mask(table, cond.field)
    r = cond.range_from_query
    mask = None
    for op, colname in (
        (np.greater, r.gt), (np.greater_equal, r.gte),
        (np.less, r.lt), (np.less_equal, r.lte),
    ):
        if colname is None:
            continue
        qvals = query_values[colname].astype(np.float64, copy=False)
        # op(corpus, bound) — matching the static path's op(col, bound)
        # (e.g. `lt: X` means "corpus value < X"), not the other way round.
        part = op(corpus_vals[None, :], qvals[:, None])
        mask = part if mask is None else (mask & part)
    return mask & not_null[None, :]


def _token_row_masks(
    col: pa.ChunkedArray, tokens: set[str], n_rows: int,
) -> dict[str, np.ndarray]:
    r"""`{token: (n_rows,) np.bool}` — for each query `token` (already
    lowercase, from `nova_bf.tokenize`), which rows of `col` contain it as
    one of their own tokens. THE text-matching primitive: both static
    `match_text` and per-query `match_text_from_query` are ANDs of these
    masks (`_phrase_mask`). Built in ONE tokenization pass over the column
    (`split_pattern_regex` then `utf8_lower` per row-batch — the same
    kernels, in the same split-then-lower order, the query side runs — then
    a vectorized `index_in`-against-the-query-vocabulary scatter). Cost is
    O(total text bytes + total corpus tokens), essentially independent of
    how many distinct query tokens are being asked for, where the old
    regex-per-word implementation paid a full column scan per distinct word.

    A null corpus row splits to a null token-list, which `list_flatten`/
    `list_parent_indices` simply skip — so null rows stay `False` in every
    mask ("a null payload value never matches") with no explicit fill step.

    The masks are rows of ONE `(n_tokens, n_rows)` array, and each batch
    scatters its own matches directly into that grid from its worker thread:
    batches own disjoint row-ranges (disjoint grid columns), so concurrent
    writes never touch the same byte, no per-batch result accumulates on the
    calling thread, and there's no sort/group step at all — just one boolean
    scatter per batch. The Arrow kernels release the GIL, so the thread pool
    is real parallelism.

    Batch size is derived, not fixed: at most `_BATCH_TEXT_BYTES` of text
    per batch (bounds every transient BY BYTES, huge documents included, and
    stays far under the 2^31 per-batch token-count list-offset ceiling), and
    at least ~2 batches per core when the file is big enough to split
    (parallelism on small files), floored so tiny batches don't drown in
    per-batch overhead. The accumulated grid is the same `n_tokens × n_rows`
    bytes the old per-word cache held, so steady-state memory is unchanged.

    `col` is cast to `large_string` up front: `combine_chunks()` on a batch
    concatenates chunk buffers, and a 32-bit `string` column's offsets can
    overflow there on huge-document corpora ("offset overflow while
    concatenating arrays"). `string`/`large_string` tokenize identically, so
    this is purely a capacity fix."""
    ordered = sorted(tokens)
    grid = np.zeros((len(ordered), n_rows), dtype=bool)
    masks = {t: grid[i] for i, t in enumerate(ordered)}
    if not ordered or n_rows == 0:
        return masks
    if pa.types.is_string(col.type):
        col = pc.cast(col, pa.large_string())
    value_set = pa.array(ordered, type=pa.large_string())

    cpus = os.cpu_count() or 1
    bytes_per_row = max(1, col.nbytes // n_rows)
    batch_rows = max(
        4_096,
        min(_BATCH_TEXT_BYTES // bytes_per_row, -(-n_rows // (2 * cpus))),
    )

    def scan(off: int) -> None:
        chunk = col.slice(off, batch_rows).combine_chunks()
        toks = pc.split_pattern_regex(chunk, pattern=TOKEN_SPLIT_PATTERN)
        lowered = pc.utf8_lower(pc.list_flatten(toks))
        parent = pc.list_parent_indices(toks)
        codes = pc.index_in(lowered, value_set=value_set)
        valid = pc.is_valid(codes)
        c = pc.filter(codes, valid).to_numpy(zero_copy_only=False)
        r = pc.filter(parent, valid).to_numpy(zero_copy_only=False)
        grid[c, r + off] = True  # disjoint column range per batch → thread-safe

    offsets = range(0, n_rows, batch_rows)
    workers = min(16, cpus, len(offsets))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(scan, offsets))
    else:
        for off in offsets:
            scan(off)
    return masks


def _phrase_mask(token_masks: dict[str, np.ndarray], toks) -> np.ndarray:
    """AND-fold of one phrase's token masks — the single place the rule
    "a row matches a phrase iff EVERY phrase token is one of its tokens"
    is spelled out; both the static and per-query paths call this.
    `toks` must be non-empty (callers resolve token-less phrases to
    all-False / reject them at config load)."""
    toks = list(toks)
    mask = token_masks[toks[0]]
    for t in toks[1:]:
        mask = mask & token_masks[t]
    return mask


def _text_token_masks(
    filt: Filter, table: pa.Table, query_values: dict[str, np.ndarray] | None,
) -> dict[str, dict[str, np.ndarray]]:
    """`{field: {token: (rows,) mask}}` for EVERY text condition anywhere in
    `filt`, built with one `_token_row_masks` pass per FIELD — a filter with
    several text conditions on the same column (e.g. a should-group of
    per-query slots all on `url`) tokenizes that column once, not once per
    condition. `evaluate()` builds this up front and threads it down through
    `_condition_mask`; direct `_condition_mask`/`_match_text_from_query_mask`
    callers (compute.py's leaf path, tests) can omit it and each condition
    builds its own masks."""
    field_tokens: dict[str, set[str]] = {}
    for cond in filt.all_conditions():
        if cond.match_text is not None:
            field_tokens.setdefault(cond.field, set()).update(tokenize(cond.match_text))
        elif cond.match_text_from_query is not None:
            tokens = field_tokens.setdefault(cond.field, set())
            for toks in tokenize_many(query_values[cond.match_text_from_query]):
                if toks:
                    tokens.update(toks)
    return {
        field: _token_row_masks(table[field], tokens, len(table))
        for field, tokens in field_tokens.items()
    }


def _match_text_static_mask(
    cond: FilterCondition, table: pa.Table,
    text_masks: dict[str, dict[str, np.ndarray]] | None,
) -> np.ndarray:
    """`(rows,)` — static `match_text`: every token of the phrase must be a
    token of the row (Qdrant MatchText vs. a `word`-tokenizer index; see
    `nova_bf.tokenize`). Config validation guarantees at least one token."""
    toks = set(tokenize(cond.match_text))
    token_masks = (text_masks or {}).get(cond.field)
    if token_masks is None:
        token_masks = _token_row_masks(table[cond.field], toks, len(table))
    return _phrase_mask(token_masks, toks)


def _match_text_from_query_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray],
    text_masks: dict[str, dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """`(n_queries, rows)` — each query's own free-text phrase, matched with
    the SAME tokenized semantics as static `match_text` (see
    `nova_bf.tokenize`), deduped per DISTINCT TOKEN SET: two phrases that
    tokenize identically (e.g. "High-Fat!" and "high fat") share one mask,
    and all phrases' distinct tokens are answered by one `_token_row_masks`
    tokenization pass over the column (shared per-FIELD across this filter's
    text conditions when `evaluate()` hands down `text_masks`) — cost is one
    scan of the text plus O(total corpus tokens), essentially independent of
    the query vocabulary size, where the old implementation paid a column
    scan per distinct word. See docs/brute-force/overview.md.

    A null/NaN phrase, or one with no alphanumeric tokens, never matches —
    the static `match_text` rejects token-less strings at config-load time
    (`FilterCondition._match_text_has_tokens`), but a per-query phrase comes
    from DATA, not a config literal, so it's resolved to all-`False` here."""
    phrases = query_values[cond.match_text_from_query]
    n_rows = len(table)
    result = np.zeros((len(phrases), n_rows), dtype=bool)

    by_tokens: dict[frozenset[str], list[int]] = {}
    for q, toks in enumerate(tokenize_many(phrases)):
        if toks:  # None (null/NaN phrase) and [] (no alphanumeric tokens) → all-False
            by_tokens.setdefault(frozenset(toks), []).append(q)
    if not by_tokens:
        return result

    token_masks = (text_masks or {}).get(cond.field)
    if token_masks is None:
        token_masks = _token_row_masks(
            table[cond.field], set().union(*by_tokens), n_rows,
        )
    for toks, qidxs in by_tokens.items():
        result[qidxs, :] = _phrase_mask(token_masks, toks)
    return result


def _condition_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray] | None = None,
    text_masks: dict[str, dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    if cond.match_from_query is not None:
        return _match_from_query_mask(cond, table, query_values)
    if cond.range_from_query is not None:
        return _range_from_query_mask(cond, table, query_values)
    if cond.match_text_from_query is not None:
        return _match_text_from_query_mask(cond, table, query_values, text_masks)

    col = table[cond.field]
    if cond.match_text is not None:
        # Already a numpy bool mask with corpus nulls resolved to False (see
        # `_token_row_masks`) — no arrow-null fill step to go through below.
        return _match_text_static_mask(cond, table, text_masks)
    if cond.match is not None:
        values = cond.match if isinstance(cond.match, tuple) else [cond.match]
        mask = pc.is_in(col, value_set=pa.array(values))
    else:
        r = cond.range
        mask = None
        for op, bound in (
            (pc.greater, r.gt),
            (pc.greater_equal, r.gte),
            (pc.less, r.lt),
            (pc.less_equal, r.lte),
        ):
            if bound is None:
                continue
            part = op(col, bound)
            mask = part if mask is None else pc.and_(mask, part)
    # A null payload value (field absent/None on that row) never matches —
    # same as Qdrant treating a missing field as non-matching.
    return pc.fill_null(mask, False).to_numpy(zero_copy_only=False)


def _static_first(conds) -> list[FilterCondition]:
    """A group's conditions reordered static-before-per-query (stable within
    each kind). AND/OR are commutative so the result is bit-identical either
    way — but combining every static `(rows,)` mask BEFORE the first
    per-query one keeps the accumulator 1-D as long as possible, instead of
    an early per-query leaf promoting it to `(n_queries, rows)` and every
    later static mask paying 2-D broadcast cost. Used by both `evaluate()`
    below and `compute._gpu_evaluate` (Front A), which mirror each other's
    combination logic."""
    return sorted(conds, key=lambda c: c.is_per_query())


def evaluate(
    filt: Filter, table: pa.Table, query_values: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """A boolean mask of which rows satisfy `filt` — shape `(len(table),)` if
    `filt` has no per-query condition anywhere (unchanged from before
    per-query filters existed: same cost, same shape), or `(n_queries,
    len(table))` the moment ANY condition, in ANY group, is per-query
    (`match_from_query`/`range_from_query`/`match_text_from_query`).

    Uses non-in-place `&`/`|`/`~` (not `&=`/`|=`) deliberately: this is what
    lets the `(rows,)` accumulator silently promote to `(n_queries, rows)`
    via numpy broadcasting the first time a per-query condition's mask
    appears, regardless of which group (`must`/`should`/`must_not`) it's in
    — an in-place op can't grow its own shape this way. Callers (`compute.py`)
    tell which case they got via `mask.ndim`."""
    n = len(table)
    # One tokenization pass per FIELD for every text condition in the filter
    # (see `_text_token_masks`) — {} when the filter has no text condition.
    text_masks = _text_token_masks(filt, table, query_values)
    keep = np.ones(n, dtype=bool)

    for cond in _static_first(filt.must):
        keep = keep & _condition_mask(cond, table, query_values, text_masks)

    if filt.should:
        any_match = np.zeros(n, dtype=bool)
        for cond in _static_first(filt.should):
            any_match = any_match | _condition_mask(cond, table, query_values, text_masks)
        keep = keep & any_match

    for cond in _static_first(filt.must_not):
        keep = keep & ~_condition_mask(cond, table, query_values, text_masks)

    return keep
