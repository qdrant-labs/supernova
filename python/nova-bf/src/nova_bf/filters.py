"""Evaluating a corpus-side `Filter` (see config.py) against one corpus file.

Runs once per file, over the whole file at once, via `pyarrow.compute` — O(rows),
independent of query count. The filter restricts which corpus points are
eligible neighbors for every query in the run; it is evaluated before scoring,
not per (query, row), same as a Qdrant search filter only ever touches the
points being searched.
"""

from __future__ import annotations

import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nova_bf.config import Filter, FilterCondition


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
        pattern = rf"\b{re.escape(word)}\b"
        part = pc.match_substring_regex(col, pattern=pattern, ignore_case=True)
        mask = part if mask is None else pc.and_(mask, part)
    return mask


def _condition_mask(cond: FilterCondition, table: pa.Table) -> np.ndarray:
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


def evaluate(filt: Filter, table: pa.Table) -> np.ndarray:
    """A `(len(table),)` boolean mask: which rows satisfy `filt`."""
    n = len(table)
    keep = np.ones(n, dtype=bool)

    for cond in filt.must:
        keep &= _condition_mask(cond, table)

    if filt.should:
        any_match = np.zeros(n, dtype=bool)
        for cond in filt.should:
            any_match |= _condition_mask(cond, table)
        keep &= any_match

    for cond in filt.must_not:
        keep &= ~_condition_mask(cond, table)

    return keep
