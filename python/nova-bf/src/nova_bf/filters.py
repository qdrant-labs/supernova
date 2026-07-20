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


def _text_prep(
    filt: Filter, table: pa.Table, query_values: dict[str, np.ndarray] | None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[FilterCondition, list[frozenset[str] | None]]]:
    """The one up-front tokenization step `evaluate()` does for a filter:

    - `text_masks`: `{field: {token: (rows,) mask}}` for EVERY text condition
      anywhere in `filt`, built with one `_token_row_masks` pass per FIELD —
      a filter with several text conditions on the same column (e.g. a
      should-group of per-query slots all on `url`) tokenizes that column
      once, not once per condition.
    - `cond_qsets`: for each `match_text_from_query` condition, each query's
      phrase as a `frozenset` of tokens (`None` for a null/NaN/token-less
      phrase — the "matches nothing" convention), tokenized once here and
      shared by the fused combine in `evaluate()`.

    Direct `_condition_mask`/`_match_text_from_query_mask` callers (tests,
    and any future direct caller — compute.py's leaf path is gated by
    `_gpu_eligible` and never reaches the text branches) skip this and each
    condition builds its own masks."""
    field_tokens: dict[str, set[str]] = {}
    cond_qsets: dict[FilterCondition, list[frozenset[str] | None]] = {}
    for cond in filt.all_conditions():
        if cond.match_text is not None:
            field_tokens.setdefault(cond.field, set()).update(tokenize(cond.match_text))
        elif cond.match_text_from_query is not None:
            tokens = field_tokens.setdefault(cond.field, set())
            qsets: list[frozenset[str] | None] = []
            for toks in tokenize_many(query_values[cond.match_text_from_query]):
                if toks:
                    tokens.update(toks)
                    qsets.append(frozenset(toks))
                else:
                    qsets.append(None)
            cond_qsets[cond] = qsets
    text_masks = {
        field: _token_row_masks(table[field], tokens, len(table))
        for field, tokens in field_tokens.items()
    }
    return text_masks, cond_qsets


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
    from DATA, not a config literal, so it's resolved to all-`False` here.

    NOT the production combine path: `evaluate()` routes per-query text
    conditions through its fused, query-major combine instead (see there),
    which never materializes this per-condition 2-D mask. This builder
    remains for direct `_condition_mask` callers and as the independent
    per-condition reference the A/B fuzz test pins the fused path against —
    a semantics change here MUST be mirrored in the fused path (the fuzz
    test is what catches a drift)."""
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
    via numpy broadcasting the first time a NON-TEXT per-query condition's
    mask appears, regardless of group. Per-query TEXT conditions
    (`match_text_from_query`) never materialize per-condition 2-D masks at
    all — they take the fused, query-major path below (see the inline
    rationale), which writes each query's finished row into the one output
    array directly. Either way the contract is the same: callers
    (`compute.py`) tell which case they got via `mask.ndim`."""
    n = len(table)
    # One tokenization pass per FIELD for every text condition in the filter,
    # plus each per-query phrase's token set (see `_text_prep`) — both empty
    # when the filter has no text condition.
    text_masks, cond_qsets = _text_prep(filt, table, query_values)

    # Split each group into its `match_text_from_query` members (combined by
    # the FUSED, query-major path below) and everything else (combined
    # condition-major, exactly as before — static masks are (rows,) and
    # cheap; non-text per-query masks are built 2-D by their own builders
    # either way).
    must_t = [c for c in filt.must if c.match_text_from_query is not None]
    should_t = [c for c in filt.should if c.match_text_from_query is not None]
    mnot_t = [c for c in filt.must_not if c.match_text_from_query is not None]

    keep = np.ones(n, dtype=bool)
    for cond in _static_first(filt.must):
        if cond.match_text_from_query is None:
            keep = keep & _condition_mask(cond, table, query_values, text_masks)

    rest_or = None
    if any(c.match_text_from_query is None for c in filt.should):
        rest_or = np.zeros(n, dtype=bool)
        for cond in _static_first(filt.should):
            if cond.match_text_from_query is None:
                rest_or = rest_or | _condition_mask(cond, table, query_values, text_masks)

    for cond in _static_first(filt.must_not):
        if cond.match_text_from_query is None:
            keep = keep & ~_condition_mask(cond, table, query_values, text_masks)

    if not (must_t or should_t or mnot_t):
        # No per-query text condition: the pre-fusion combine, unchanged.
        if rest_or is not None:
            keep = keep & rest_or
        return keep

    # --- fused, query-major combine for the per-query text conditions ---
    # Rationale: expanding every text condition to its own (n_queries, rows)
    # array and combining those was ~80% of filter time on the production
    # workload — pure memory traffic. Instead, group queries by their COMBO
    # of token sets across all text conditions (real query sets dedupe
    # heavily), compute each distinct combo's combined (rows,) result once
    # from the shared per-token masks, and write each query's final row
    # exactly ONCE. Bit-identical by construction: AND/OR are elementwise
    # and every (query, row) cell sees the same boolean formula, just
    # evaluated query-major instead of condition-major.
    # Every query column referenced by the filter must agree on n_queries —
    # mismatched lengths raised a loud broadcast ValueError on the old
    # condition-major path, and silently truncating here instead would be a
    # correctness trap for direct evaluate() callers (compute.py always
    # draws all columns from one queries table, so it can't hit this).
    lengths = {len(cond_qsets[c]) for c in (*must_t, *should_t, *mnot_t)}
    if len(lengths) > 1:
        raise ValueError(
            f"per-query text conditions reference query columns of differing "
            f"lengths: {sorted(lengths)}"
        )
    n_q = lengths.pop()
    if keep.ndim == 2 and keep.shape[0] != n_q:
        raise ValueError(f"query column length mismatch: {keep.shape[0]} vs {n_q}")
    if rest_or is not None and rest_or.ndim == 2 and rest_or.shape[0] != n_q:
        raise ValueError(f"query column length mismatch: {rest_or.shape[0]} vs {n_q}")

    combos: dict[tuple, list[int]] = {}
    for q in range(n_q):
        key = (
            tuple(cond_qsets[c][q] for c in must_t),
            tuple(cond_qsets[c][q] for c in should_t),
            tuple(cond_qsets[c][q] for c in mnot_t),
        )
        combos.setdefault(key, []).append(q)

    # (field, token-set) → (rows,) phrase mask, shared across combos. Entries
    # are refcounted by how many still-unprocessed combos need them and
    # evicted at zero, so peak cache size tracks LIVE masks, not every
    # distinct phrase the whole query set uses (which would grow linearly
    # with text-condition count on poorly-deduping query sets, where the old
    # condition-major path's peak was constant in condition count).
    def _combo_key_list(mkey, skey, nkey) -> list[tuple[str, frozenset[str]]]:
        """The cache keys processing this combo will touch — mirrors the
        combo loop exactly, including the dead-combo early-out."""
        if any(ts is None for ts in mkey):
            return []
        keys = [(c.field, ts) for c, ts in zip(must_t, mkey)]
        keys += [(c.field, ts) for c, ts in zip(mnot_t, nkey) if ts is not None]
        keys += [(c.field, ts) for c, ts in zip(should_t, skey) if ts is not None]
        return keys

    key_refs: dict[tuple[str, frozenset[str]], int] = {}
    for (mkey, skey, nkey) in combos:
        for k in _combo_key_list(mkey, skey, nkey):
            key_refs[k] = key_refs.get(k, 0) + 1
    phrase_cache: dict[tuple[str, frozenset[str]], np.ndarray] = {}

    def _pmask(cond: FilterCondition, ts: frozenset[str]) -> np.ndarray:
        got = phrase_cache.get((cond.field, ts))
        if got is None:
            got = _phrase_mask(text_masks[cond.field], ts)
            phrase_cache[(cond.field, ts)] = got
        return got

    keep_2d = keep.ndim == 2
    rest_or_2d = rest_or is not None and rest_or.ndim == 2
    out = np.empty((n_q, n), dtype=bool)
    for (mkey, skey, nkey), qidxs in combos.items():
        if any(ts is None for ts in mkey):
            # a null/token-less phrase in a `must` matches nothing for that
            # query, so the whole row is False regardless of anything else.
            out[qidxs] = False
            continue
        # 1-D parts shared by every query in this combo:
        parts: list[np.ndarray] = [_pmask(c, ts) for c, ts in zip(must_t, mkey)]
        for c, ts in zip(mnot_t, nkey):
            if ts is not None:  # None: matches nothing → ¬nothing keeps all
                parts.append(~_pmask(c, ts))
        or_2d = None
        if filt.should:
            s = None  # this combo's OR over the should group's text members
            for c, ts in zip(should_t, skey):
                if ts is None:
                    continue  # null phrase contributes False to the OR
                pm = _pmask(c, ts)
                s = pm if s is None else (s | pm)
            if rest_or is None:
                # should group is all-text: s (or nothing matched → False row)
                parts.append(s if s is not None else np.zeros(n, dtype=bool))
            elif not rest_or_2d:
                parts.append(rest_or if s is None else (rest_or | s))
            else:
                or_2d = rest_or[qidxs] if s is None else (rest_or[qidxs] | s)
        if not keep_2d:
            parts.append(keep)
        # parts can be empty (e.g. every must_not phrase null for this combo
        # while `keep` is 2-D) — the combo then constrains nothing 1-D.
        row = np.ones(n, dtype=bool)
        for p in parts:
            row = row & p
        if keep_2d or or_2d is not None:
            block = row[None, :]
            if keep_2d:
                block = block & keep[qidxs]
            if or_2d is not None:
                block = block & or_2d
            out[qidxs] = block
        else:
            out[qidxs] = row
        for k in _combo_key_list(mkey, skey, nkey):
            key_refs[k] -= 1
            if key_refs[k] == 0:
                phrase_cache.pop(k, None)
    return out
