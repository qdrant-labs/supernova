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
"""

from __future__ import annotations

import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nova_bf.config import Filter, FilterCondition


def _word_mask(word: str, col: pa.ChunkedArray) -> pa.ChunkedArray:
    """One word's substring-match mask — the single-word primitive both
    `_match_text_mask` (ANDs it across a whole phrase, no caching) and
    `_match_text_from_query_mask` (caches it across every DISTINCT phrase
    sharing that word — see there) build on, so the regex/escaping rule
    lives in exactly one place."""
    pattern = rf"\b{re.escape(word)}\b"
    return pc.match_substring_regex(col, pattern=pattern, ignore_case=True)


def _match_text_mask(text: str, col: pa.ChunkedArray) -> pa.ChunkedArray:
    """AND of per-word substring matches — Qdrant's MatchText semantics.

    This is a whitespace+word-boundary-regex approximation of Qdrant's real
    tokenizer, not a byte-for-byte replica: a hyphenated query word like
    `high-fat` is matched as one literal token rather than split into
    `high`/`fat`, and a word ending directly in trailing punctuation (`C++`)
    can fail to get a `\\b` boundary on that side. Good enough for
    keyword-style corpus filtering.
    """
    mask = None
    for word in text.split():
        part = _word_mask(word, col)
        mask = part if mask is None else pc.and_(mask, part)
    return mask


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


def _literal_then_regex_word_mask(word: str, col: pa.ChunkedArray) -> np.ndarray:
    """`(rows,)` numpy bool, nulls already resolved to `False` — same result
    as `pc.fill_null(pc.match_substring_regex(col, pattern=rf"\\b{word}\\b",
    ignore_case=True), False).to_numpy(...)`, computed via a cheaper two-pass
    narrowing instead of one regex pass over the whole column: a plain
    literal substring match (no regex compile, no `\\b` bookkeeping) first
    narrows to rows that could possibly match, since a `\\b`-bounded match
    always IMPLIES plain substring presence — a row the literal pass
    excludes can never pass the stricter regex either, so this is exact,
    not a heuristic (holds even on inputs where Arrow's regex `\\b` itself
    behaves surprisingly, e.g. accented letters — verified directly: regex-match
    is always a subset of literal-match, so the two-pass result is always
    identical to a single regex pass, whatever that regex does). The pricier
    regex only re-verifies that (usually much smaller) subset. Skips the
    regex pass entirely when nothing survives the literal prefilter. Used
    only by `_match_text_from_query_mask`'s per-word cache below, where a
    numpy array (nulls pre-filled) is what gets cached and ANDed across a
    phrase's words anyway — measured ~2-3x faster than the previous
    single-regex-pass implementation on real corpus text (see
    docs/brute-force/overview.md)."""
    literal = pc.fill_null(pc.match_substring(col, word, ignore_case=True), False).to_numpy(zero_copy_only=False)
    idx = np.nonzero(literal)[0]
    if len(idx) == 0:
        return literal
    pattern = rf"\b{re.escape(word)}\b"
    verified = pc.fill_null(
        pc.match_substring_regex(col.take(pa.array(idx)), pattern=pattern, ignore_case=True), False,
    ).to_numpy(zero_copy_only=False)
    result = np.zeros(len(col), dtype=bool)
    result[idx] = verified
    return result


_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")


def _simple_word_masks(
    col: pa.ChunkedArray, words: set[str], n_rows: int, batch_rows: int = 200_000,
) -> dict[str, np.ndarray]:
    r"""`{word: (n_rows,) np.bool}` for plain-ASCII `\w+` `words` (passed
    already lowercased), built in ONE tokenization pass per row-batch instead
    of one full-column scan per distinct word (optimization B).

    Tokenizes `lower(text)` by splitting on `\W+` — the SAME RE2 word class the
    `\b` verify in `_literal_then_regex_word_mask` uses — so for an ASCII `\w+`
    word, "row matches `\bword\b` case-insensitively" is *exactly* "word is one
    of that row's `\W`-delimited tokens". This equivalence holds ONLY for ASCII
    `\w+` words: a word containing punctuation or non-ASCII letters can match a
    `\b`-bounded span that is not a whole token (e.g. `high-fat`), so the caller
    keeps those on the regex path — the fuzz test asserts bit-identical output.
    Case-folding is done once here via `utf8_lower` (optimization D) rather than
    once per word.

    Row-batched so the transient flattened-token array stays bounded on a
    multi-GB reshard `text` column. The accumulated per-word masks are the same
    `(n_rows,)` arrays the per-word cache already held, so peak memory matches
    the old scan-per-word path."""
    masks = {w: np.zeros(n_rows, dtype=bool) for w in words}
    ordered = list(words)
    total = len(col)
    off = 0
    while off < total:
        chunk = col.slice(off, batch_rows).combine_chunks()
        toks = pc.split_pattern_regex(pc.utf8_lower(chunk), pattern=r"\W+")
        flat = pc.list_flatten(toks)
        parent = pc.list_parent_indices(toks)
        codes = pc.index_in(flat, value_set=pa.array(ordered, type=flat.type))
        valid = pc.is_valid(codes)
        c = pc.filter(codes, valid).to_numpy(zero_copy_only=False)
        r = pc.filter(parent, valid).to_numpy(zero_copy_only=False)
        if len(c):
            order = np.argsort(c, kind="stable")
            cs = c[order]
            rs = r[order].astype(np.int64) + off
            lo = np.searchsorted(cs, np.arange(len(ordered)), side="left")
            hi = np.searchsorted(cs, np.arange(len(ordered)), side="right")
            for wi in range(len(ordered)):
                if hi[wi] > lo[wi]:
                    masks[ordered[wi]][rs[lo[wi]:hi[wi]]] = True
        off += batch_rows
    return masks


def _match_text_from_query_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray],
) -> np.ndarray:
    """`(n_queries, rows)` — each query's own free-text phrase, matched with
    the SAME word-boundary-AND semantics as `_match_text_mask`, but deduped
    per DISTINCT WORD rather than per distinct phrase: two phrases sharing a
    word (e.g. "wireless mouse" and "wireless keyboard") reuse that word's
    regex pass instead of redoing it. Cost approaches `O(distinct words ×
    rows)` rather than `O(distinct phrases × rows)` — a real win whenever
    real query phrases share vocabulary, the common case. Cache key is the
    lowercased word: `ignore_case=True` already makes "Wireless"/"wireless"
    match identical rows, so casing shouldn't fragment the cache. Each
    word's mask is itself computed via `_literal_then_regex_word_mask`'s
    literal-prefilter, on top of the word-level cache — the two optimizations
    are independent and compose. See docs/brute-force/overview.md."""
    col = table[cond.field]
    # `_literal_then_regex_word_mask`'s regex-verify pass does `col.take(idx)`,
    # which concatenates the gathered chunks into one array. On a corpus whose
    # text column exceeds 2 GB in a single file (e.g. a FineWeb reshard's
    # `text`), a 32-bit `string` column's offsets overflow there
    # ("offset overflow while concatenating arrays"). Cast to `large_string`
    # (per-chunk, so the cast itself never concatenates past 2 GB) so the gather
    # uses 64-bit offsets. `string`/`large_string` are match-identical, so this
    # is purely a capacity fix, not a semantic change.
    if pa.types.is_string(col.type):
        col = pc.cast(col, pa.large_string())
    phrases = query_values[cond.match_text_from_query]
    n_rows = len(table)
    result = np.zeros((len(phrases), n_rows), dtype=bool)

    by_phrase: dict[str, list[int]] = {}
    for q, phrase in enumerate(phrases):
        if phrase is None or phrase != phrase:  # None, or nan (nan != nan)
            continue
        if not phrase.strip():
            # A blank/whitespace-only phrase never matches, same as null —
            # the static `match_text` rejects this at config-load time (see
            # `FilterCondition._match_text_not_blank`), but a per-query
            # phrase comes from DATA, not a config literal, so it can't be
            # rejected upfront the same way. Without this check, a phrase
            # would contribute zero words below and never populate `mask`.
            continue
        by_phrase.setdefault(phrase, []).append(q)

    # Null propagation note: filling each word's mask to False here (rather
    # than ANDing arrow masks with null propagation and filling once at the
    # end, as the old phrase-level version did) is equivalent — a null
    # corpus value makes EVERY word's regex mask null at that row, so
    # filling each to False before ANDing still yields False there, same as
    # ANDing nulls through and filling once at the end.
    word_cache: dict[str, np.ndarray] = {}
    # Optimization B: build the masks for all plain-ASCII \w+ words in ONE
    # tokenization pass (`_simple_word_masks`) instead of one full-column scan
    # per distinct word. Words with punctuation or non-ASCII letters aren't
    # safe to answer by tokenization (see there), so they fall through to the
    # exact per-word regex path below — bit-identical to the old behavior.
    simple = {w.lower() for phrase in by_phrase for w in phrase.split() if _ASCII_WORD.fullmatch(w)}
    if simple:
        word_cache.update(_simple_word_masks(col, simple, n_rows))
    for phrase, qidxs in by_phrase.items():
        mask = None
        for word in phrase.split():
            key = word.lower()
            cached = word_cache.get(key)
            if cached is None:
                cached = _literal_then_regex_word_mask(word, col)
                word_cache[key] = cached
            mask = cached if mask is None else (mask & cached)
        result[qidxs, :] = mask
    return result


def _condition_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if cond.match_from_query is not None:
        return _match_from_query_mask(cond, table, query_values)
    if cond.range_from_query is not None:
        return _range_from_query_mask(cond, table, query_values)
    if cond.match_text_from_query is not None:
        return _match_text_from_query_mask(cond, table, query_values)

    col = table[cond.field]
    if cond.match is not None:
        values = cond.match if isinstance(cond.match, tuple) else [cond.match]
        mask = pc.is_in(col, value_set=pa.array(values))
    elif cond.match_text is not None:
        mask = _match_text_mask(cond.match_text, col)
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
    keep = np.ones(n, dtype=bool)

    for cond in _static_first(filt.must):
        keep = keep & _condition_mask(cond, table, query_values)

    if filt.should:
        any_match = np.zeros(n, dtype=bool)
        for cond in _static_first(filt.should):
            any_match = any_match | _condition_mask(cond, table, query_values)
        keep = keep & any_match

    for cond in _static_first(filt.must_not):
        keep = keep & ~_condition_mask(cond, table, query_values)

    return keep
