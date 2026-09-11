"""Evaluate a corpus-side `Filter` against one corpus file.

Uniform filters produce a `(rows,)` boolean mask and are evaluated once per
file before scoring. Per-query conditions produce a row-packed
`PackedRowMask` over `(queries, rows)` and are applied per query during
scoring rather than by compacting corpus rows.

Text filters use Qdrant-style `word` tokenization: split on non-alphanumeric
characters, lowercase tokens, and require all query tokens to occur in the
row's token set. Query and corpus text use the same tokenization path.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nova_bf import nativetok
from nova_bf.config import Filter, FilterCondition
from nova_bf.tokenize import TOKEN_SPLIT_PATTERN, tokenize, tokenize_many

class PackedRowMask:
    """A `(queries, rows)` boolean mask packed along the row axis.

    `packed` has shape `(n_queries, ceil(n_rows / 8))` using NumPy's
    big-endian bit order. Row packing reduces mask residency 8x while keeping
    contiguous row ranges cheap to slice.

    `n_rows` records the true width because the final byte may contain padding.
    """

    __slots__ = ("packed", "n_rows")

    def __init__(self, packed: np.ndarray, n_rows: int):
        if packed.ndim != 2 or packed.dtype != np.uint8:
            raise ValueError(
                f"PackedRowMask wants a 2-D uint8 array, got "
                f"{packed.ndim}-D {packed.dtype}"
            )
        if n_rows < 0 or packed.shape[1] != (n_rows + 7) // 8:
            # Require exactly the byte width implied by `n_rows`.
            raise ValueError(
                f"n_rows={n_rows} needs exactly {(max(0, n_rows) + 7) // 8} "
                f"packed bytes, got {packed.shape[1]}"
            )
        self.packed = packed
        self.n_rows = n_rows

    @property
    def n_queries(self) -> int:
        return self.packed.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the unpacked mask."""
        return (self.packed.shape[0], self.n_rows)

    @property
    def ndim(self) -> int:
        """Always 2 — `compute.py` still tells the per-query case from the
        uniform `(rows,)` bool array by `ndim`."""
        return 2

    def __getitem__(self, qrows) -> "PackedRowMask":
        """Narrow the query axis. `qrows` must KEEP that axis 2-D.

        A bare integer drops it, and the constructor would then complain that
        the array is 1-D — pointing at the packing rather than at the index
        that caused it. Refuse it here, with the fix, since `pm[q]` is the
        natural thing to reach for. Not `bool`: `packed[True]` ADDS an axis
        rather than dropping one, so that message would be untrue; the
        constructor's ndim check catches it.
        """
        if isinstance(qrows, (int, np.integer)) and not isinstance(qrows, bool):
            raise TypeError(
                f"PackedRowMask must stay 2-D: indexing with the integer "
                f"{qrows} would drop the query axis. Use "
                f"`[{qrows}:{qrows + 1}]` to keep query {qrows} as a 1-tall "
                f"mask, or `.unpack()[{qrows}]` for its bool row."
            )
        return PackedRowMask(self.packed[qrows], self.n_rows)

    def any(self) -> bool:
        """Whether any REAL row bit is set.

        Masks the last byte's padding instead of reading the raw bytes. Those
        padding bits sit outside `n_rows`, so a mask carrying them would
        report True here while `unpack()` — which trims with `count=n_rows` —
        reports False: two methods of one object contradicting each other.

        Every mask this module builds keeps the padding at 0 (`_tail_mask`
        records why, and the combine's closing `&` against `_packed_ones`
        enforces it whatever the parts held), so the raw read agreed in
        practice. Masking makes that a property of the class rather than a
        promise about its callers — the same reason
        `compute._packed_slice_any` is exact about its head and tail bytes.
        """
        if not self.packed.shape[1]:
            return False
        if bool(self.packed[:, :-1].any()):
            return True
        return bool((self.packed[:, -1] & _tail_mask(self.n_rows)).any())

    def unpack(self) -> np.ndarray:
        """Expand to the represented `(n_queries, n_rows)` boolean mask."""
        return np.unpackbits(
            self.packed, axis=1, count=self.n_rows
        ).astype(bool)



def _tail_mask(n_rows: int) -> int:
    """Mask of valid bits in the final row-packed byte.

    `np.packbits` is big-endian, so partial-byte rows occupy the high bits.
    Packed operations preserve zero padding except `~`, which must re-mask.
    """
    r = n_rows & 7
    return 0xFF if r == 0 else (0xFF << (8 - r)) & 0xFF


def _packed_not(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Invert a packed row mask while keeping tail padding bits zero."""
    out = ~packed

    # `~` also flips padding bits; clear them in the final byte.
    if n_rows & 7 and out.shape[-1]:
        out[..., -1] &= _tail_mask(n_rows)
    return out


def _packed_ones(n_rows: int) -> np.ndarray:
    """An all-True packed row (padding bits still 0)."""
    out = np.full((n_rows + 7) // 8, 0xFF, dtype=np.uint8)
    if n_rows & 7 and len(out):
        out[-1] = _tail_mask(n_rows)
    return out


class TokenGrid:
    """Row-packed token membership over a corpus file.

    Maps each token to a `(ceil(n_rows / 8),)` uint8 mask in NumPy `packbits`
    order. Packing reduces row-mask storage and combine traffic by 8x and lets
    downstream phrase masks remain packed throughout combination.

    `grid[token]` returns the packed row mask; `grid.mask(token)` expands it to
    `(n_rows,)` bool for reference/test paths.
    """

    __slots__ = ("packed", "n_rows", "_index")

    def __init__(self, packed: np.ndarray, n_rows: int, ordered: list[str]):
        self.packed = packed
        self.n_rows = n_rows
        self._index = {t: i for i, t in enumerate(ordered)}

    def __contains__(self, token) -> bool:
        return token in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def keys(self):
        return self._index.keys()

    def __getitem__(self, token) -> np.ndarray:
        return self.packed[self._index[token]]

    def mask(self, token) -> np.ndarray:
        """The `(n_rows,)` bool row. Allocates; not for the hot path."""
        return np.unpackbits(self[token], count=self.n_rows).astype(bool)


def pack_rows(mask: np.ndarray) -> PackedRowMask:
    """Bit-pack an `(n_queries, n_rows)` bool mask along the row axis."""
    return PackedRowMask(np.packbits(mask, axis=1), mask.shape[1])


# Per-batch text-byte target for `_token_row_masks`: bounds every transient
# the scan materializes (the split token copy, its lowered copy, parent/code
# index arrays — a few multiples of the batch's text bytes) by BYTES, not
# rows, so huge-document corpora can't blow the bound; also keeps a batch's
# token count far below the 2^31 list-offset ceiling `split_pattern_regex`'s
# `list<large_string>` output still has even after the `large_string` cast.
_BATCH_TEXT_BYTES = 32 << 20


def _pool_width(pool) -> int:
    """Return the thread-count estimate used to size scan batches.

    Uses the shared pool's width when present; otherwise returns the usable CPU
    count, which may exceed the private pool's actual concurrency.
    """
    width = getattr(pool, "_max_workers", None)
    if width:
        return width
    # Respect affinity/cgroup CPU limits. Import lazily to avoid an import cycle.
    from nova_bf.compute import _usable_cpu_count

    return _usable_cpu_count()


# Per-task cap on the temporary bool sub-grid built before row packing.
# Bounds `n_tokens * batch_rows` bytes; concurrent tasks scale this transient
# with pool width while avoiding the full-height bool grid.
_SUBGRID_BYTES = 16 << 20


def _scan_batch_rows(col_nbytes: int, n_rows: int, width: int,
                     n_tokens: int = 1) -> int:
    """Choose a byte-aligned tokenization batch size.

    Size is limited by estimated text bytes, the exact temporary bool-grid
    size, and a target of roughly two batches per worker. A 4096-row preferred
    minimum is overridden by tighter memory limits.

    Batches are multiples of 8 so concurrent tasks own disjoint bytes in the
    packed output. If a memory cap implies fewer than 8 rows, 8 is the smallest
    representable batch and necessarily exceeds that cap.
    """
    bytes_per_row = max(1, col_nbytes // max(1, n_rows))

    # Text uses a file-wide average; the bool-grid bound is exact.
    cap = min(
        _BATCH_TEXT_BYTES // bytes_per_row,
        _SUBGRID_BYTES // max(1, n_tokens),
    )

    # Prefer enough batches to keep the pool fed without creating tiny tasks.
    rows = max(4_096, min(cap, -(-n_rows // (2 * max(1, width)))))

    # Memory limits override the preferred minimum; keep boundaries byte-aligned.
    return max(8, min(rows, cap) & ~7)


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
    col: pa.ChunkedArray, tokens: set[str], n_rows: int, pool=None,
) -> TokenGrid:
    """Build row-packed corpus membership masks for query tokens.
    
    Tokenizes the corpus once using the same split-then-lower path as query
    text, then scatters matching occurrences into a packed
    `(n_tokens, ceil(n_rows / 8))` grid. Static and per-query text filters
    combine these masks with `_phrase_mask`.

    Null corpus rows produce no tokens and therefore match nothing.

    Batches write directly into disjoint byte ranges of the shared packed
    grid. Batch sizing balances estimated text size, temporary bool-grid
    memory, and scan parallelism.

    String input is widened to `large_string` before chunk combination to
    avoid 32-bit string-offset overflow on large inputs.

    `pool` may provide the process-wide scan pool; otherwise a private pool
    is used.
    """
    ordered = sorted(tokens)
    n_tok = len(ordered)
    grid = np.zeros((n_tok, (n_rows + 7) // 8), dtype=np.uint8)
    out = TokenGrid(grid, n_rows, ordered)
    if not ordered or n_rows == 0:
        return out
    if pa.types.is_string(col.type):
        col = pc.cast(col, pa.large_string())
    # If available, the native scanner consumes Arrow's buffers directly in
    # one Rust call per batch.  It derives its Unicode tables from this Arrow
    # build and self-checks against the pipeline below; a missing extension or
    # any failed check leaves the established Arrow path untouched.
    native = nativetok.prepare(ordered)
    value_set = pa.array(ordered, type=pa.large_string()) if native is None else None

    batch_rows = _scan_batch_rows(col.nbytes, n_rows, _pool_width(pool), n_tok)

    # Concurrent batches must own disjoint packed bytes.
    batch_rows = max(8, batch_rows & ~7)

    def scan(off: int) -> None:
        chunk = col.slice(off, batch_rows).combine_chunks()
        n_here = len(chunk)
        
        # Build this batch's bool grid, then pack it directly into its
        # byte-aligned region of the shared output.
        sub_grid = np.zeros((n_tok, n_here), dtype=bool)
        if native is not None:
            nativetok.scan_into(chunk, native, sub_grid)
        else:
            toks = pc.split_pattern_regex(chunk, pattern=TOKEN_SPLIT_PATTERN)
            lowered = pc.utf8_lower(pc.list_flatten(toks))
            parent = pc.list_parent_indices(toks)
            codes = pc.index_in(lowered, value_set=value_set)
            valid = pc.is_valid(codes)
            c = pc.filter(codes, valid).to_numpy(zero_copy_only=False)
            r = pc.filter(parent, valid).to_numpy(zero_copy_only=False)
            sub_grid[c, r] = True
        grid[:, off >> 3 : (off + n_here + 7) >> 3] = np.packbits(sub_grid, axis=1)

    offsets = range(0, n_rows, batch_rows)
    if pool is not None and len(offsets) > 1:
        # Submit individual batches to the shared pool so work from concurrent
        # readers can share its fixed global concurrency.
        futures = []
        try:
            for off in offsets:
                futures.append(pool.submit(scan, off))
        except BaseException as exc:
            for f in futures:
                try:
                    f.result()
                except BaseException:
                    pass
            raise exc
        # Drain already-submitted tasks before abandoning their shared grid.
        error = None
        for f in futures:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                error = error or exc
        if error is not None:
            raise error
        return out
    from nova_bf.compute import _usable_cpu_count

    workers = min(16, _usable_cpu_count(), len(offsets))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as own:
            list(own.map(scan, offsets))
    else:
        for off in offsets:
            scan(off)
    return out


def _phrase_mask(token_masks: "TokenGrid", toks) -> np.ndarray:
    """Return the packed rows containing every token in a non-empty phrase."""
    toks = list(toks)
    mask = token_masks[toks[0]]
    for t in toks[1:]:
        mask = mask & token_masks[t]
    return mask


@dataclass(frozen=True)
class TextQueryPrep:
    """File-independent preprocessing for per-query text filters.

    `field_tokens` collects all requested tokens per corpus field.
    `cond_qsets` stores each `match_text_from_query` condition's token set per
    query, with `None` for phrases that match nothing.

    Built once by `prepare_text_queries()` and reused across corpus files.
    The prep stores no provenance, so callers must pair it with the same filter
    and query values it was built from; `evaluate()` checks condition coverage
    and query counts but cannot detect same-length stale query values.
    """

    field_tokens: dict[str, set[str]]
    cond_qsets: dict[FilterCondition, list[frozenset[str] | None]]


def prepare_text_queries(
    filt: Filter, query_values: dict[str, np.ndarray] | None,
) -> TextQueryPrep:
    """Tokenize this filter's static phrases and every per-query phrase ONCE
    for a whole run. See `TextQueryPrep`."""
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
    return TextQueryPrep(field_tokens, cond_qsets)


def _text_prep(
    filt: Filter, table: pa.Table, query_values: dict[str, np.ndarray] | None,
    pool=None, prep: "TextQueryPrep | None" = None,
) -> tuple[dict[str, TokenGrid], dict[FilterCondition, list[frozenset[str] | None]]]:
    """Build corpus token masks and per-query phrase token sets for a filter.

    Corpus text is tokenized once per field using the union of tokens requested
    by that field's text conditions. `prep` may supply the file-independent
    query-side preprocessing; otherwise it is built here.
    """
    if prep is None:
        prep = prepare_text_queries(filt, query_values)
    else:
         # Reject preps missing any per-query text condition.
        missing = [
            c for c in filt.all_conditions()
            if c.match_text_from_query is not None and c not in prep.cond_qsets
        ]
        if missing:
            raise ValueError(
                f"TextQueryPrep does not cover {len(missing)} of this filter's "
                f"match_text_from_query condition(s); it was built for a "
                f"different filter"
            )
        for c in filt.all_conditions():
            if c.match_text is None:
                continue
            need = set(tokenize(c.match_text))
            have = prep.field_tokens.get(c.field)
            if have is None or not need <= have:
                raise ValueError(
                    f"TextQueryPrep does not cover static match_text "
                    f"{c.match_text!r} on field {c.field!r}; it was built "
                    f"for a different filter"
                )

         # When query values are available, also verify the query count.
        for c in filt.all_conditions():
            col = c.match_text_from_query
            if col is None:
                continue
            vals = None if query_values is None else query_values.get(col)
            if vals is None:
                continue
            want = len(vals)
            got = len(prep.cond_qsets[c])
            if got != want:
                raise ValueError(
                    f"TextQueryPrep was built for {got} queries but "
                    f"query_values[{col!r}] has {want}; it is stale"
                )
    field_tokens, cond_qsets = prep.field_tokens, prep.cond_qsets
    text_masks = {
        field: _token_row_masks(table[field], tokens, len(table), pool)
        for field, tokens in field_tokens.items()
    }
    return text_masks, cond_qsets


def _match_text_static_mask(
    cond: FilterCondition, table: pa.Table,
    text_masks: dict[str, TokenGrid] | None,
) -> np.ndarray:
    """`(rows,)` — static `match_text`: every token of the phrase must be a
    token of the row (Qdrant MatchText vs. a `word`-tokenizer index; see
    `nova_bf.tokenize`). Config validation guarantees at least one token."""
    toks = set(tokenize(cond.match_text))
    token_masks = (text_masks or {}).get(cond.field)
    if token_masks is None:
        token_masks = _token_row_masks(table[cond.field], toks, len(table))
    # Packed in the grid (see `TokenGrid`); this builder's contract is a
    # `(rows,)` bool, so unpack here. Off the production path — the fused
    # combine in `evaluate()` keeps everything packed.
    return np.unpackbits(
        _phrase_mask(token_masks, toks), count=len(table)
    ).astype(bool)


def _match_text_from_query_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray],
    text_masks: dict[str, TokenGrid] | None = None,
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
        result[qidxs, :] = np.unpackbits(
            _phrase_mask(token_masks, toks), count=n_rows
        ).astype(bool)
    return result


def _condition_mask(
    cond: FilterCondition, table: pa.Table, query_values: dict[str, np.ndarray] | None = None,
    text_masks: dict[str, TokenGrid] | None = None,
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
    pool=None, prep: "TextQueryPrep | None" = None,
) -> "np.ndarray | PackedRowMask":
    """Evaluate `filt` against one corpus file.

    Uniform filters return a `(rows,)` boolean mask. If any condition is
    per-query, returns a row-packed `PackedRowMask` representing
    `(n_queries, rows)`.

    Text conditions share one corpus tokenization pass per field.
    `match_text_from_query` conditions use a packed, query-major combine so
    per-condition `(n_queries, rows)` masks are never materialized.

    `pool` may supply the shared text-scan pool, and `prep` may supply
    file-independent query-text preprocessing.

    The pre-fusion accumulators use non-in-place boolean operations so a 1-D
    mask can broadcast to 2-D when a non-text per-query condition appears.
    """
    n = len(table)

    # Tokenize each text field once and reuse its token masks across conditions.
    text_masks, cond_qsets = _text_prep(filt, table, query_values, pool, prep)

    # Handle per-query text conditions separately in the packed query-major path.
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
        return pack_rows(keep) if keep.ndim == 2 else keep

    # Group queries with identical text-condition token sets so each distinct
    # boolean combination is evaluated once.
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

    # Cache packed phrase masks while they still have remaining uses.
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

    # Build the per-query result directly in row-packed form. Text phrase masks
    # remain packed throughout; non-text accumulators are packed once before the
    # combo loop.
    nb = (n + 7) // 8
    keep_p = None if keep_2d else np.packbits(keep)
    keep_pk = np.packbits(keep, axis=1) if keep_2d else None
    rest_or_p = (np.packbits(rest_or)
                 if rest_or is not None and not rest_or_2d else None)
    rest_or_pk = np.packbits(rest_or, axis=1) if rest_or_2d else None
    out = np.empty((n_q, nb), dtype=np.uint8)
    for (mkey, skey, nkey), qidxs in combos.items():
        if any(ts is None for ts in mkey):
            # A null/token-less required phrase makes this query combo impossible.
            out[qidxs] = 0
            continue
        # 1-D PACKED parts shared by every query in this combo:
        parts: list[np.ndarray] = [_pmask(c, ts) for c, ts in zip(must_t, mkey)]
        for c, ts in zip(mnot_t, nkey):
            if ts is not None:  # None: matches nothing → ¬nothing keeps all
                parts.append(_packed_not(_pmask(c, ts), n))
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
                parts.append(s if s is not None else np.zeros(nb, dtype=np.uint8))
            elif not rest_or_2d:
                parts.append(rest_or_p if s is None else (rest_or_p | s))
            else:
                or_2d = (rest_or_pk[qidxs] if s is None
                         else (rest_or_pk[qidxs] | s))
        if not keep_2d:
            parts.append(keep_p)
        # Start from packed all-True and apply the combo's shared constraints.
        row = _packed_ones(n)
        for p in parts:
            row &= p
        if keep_2d or or_2d is not None:
             # Remaining non-text constraints vary by query within the combo.
            block = row[None, :]
            if keep_2d:
                block = block & keep_pk[qidxs]
            if or_2d is not None:
                block = block & or_2d
            out[qidxs] = block
        else:
            # The production shape: one packed row, broadcast to the combo.
            out[qidxs] = row
        for k in _combo_key_list(mkey, skey, nkey):
            key_refs[k] -= 1
            if key_refs[k] == 0:
                phrase_cache.pop(k, None)
    return PackedRowMask(out, n)
