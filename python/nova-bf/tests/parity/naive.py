"""A plain-Python reference implementation of everything nova-bf computes.

Deliberately slow and deliberately dumb: nested loops over Python floats, one
document at a time, no numpy vectorization, no batching, no shared helper
imported from `nova_bf`. That is the whole point — it can only agree with
nova-bf by both of them implementing the same *meaning*, never by both calling
the same code. Anything clever here (a vectorized rewrite, reusing
`nova_bf.filters.evaluate`, reusing `nova_bf.tokenize`) would quietly turn
this oracle into a mirror.

The one thing it does NOT try to reproduce is nova-bf's tie-break rule.
Reproducing it would mean agreeing on which scores are exactly equal, and two
independent float reductions do not agree on that — so `topk` ranks by
(score desc, corpus row asc) and the comparison helpers in `compare.py` treat
the top-K boundary with a tolerance instead. Ties as such are covered by the
dedicated tie-break suite (`tests/test_tiebreak*.py`,
`tests/test_qdrant_tiebreak_parity.py`), not here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------- documents


@dataclass
class Doc:
    """One corpus point, carrying every modality plus its payload.

    `row` is the point's global corpus row number in file-then-row order —
    nova-bf's own ordinal, and the id used throughout the harness so a hit can
    be traced back to the row that produced it.
    """

    row: int
    payload: dict = field(default_factory=dict)
    dense: list[float] | None = None
    # token id -> weight, ALREADY coalesced (a repeated raw index is one
    # dimension whose value is the sum of its occurrences — see
    # `sparse_from_pairs`).
    sparse: dict[int, float] | None = None
    # list of token vectors; [] and None both mean "has no multivector", which
    # makes the point a non-candidate for a multivector search.
    multivector: list[list[float]] | None = None


def sparse_from_pairs(indices, values) -> dict[int, float]:
    """`(indices, values)` as stored on disk -> `{token: weight}`.

    A token id repeated within one row is ONE dimension whose value is the sum
    of its occurrences (a hash collision is not two separate dimensions), so
    duplicates are summed rather than last-write-wins. nova-bf coalesces the
    same way (`compute._coalesce_by_row_col`); getting this wrong here would
    make the harness accept a corpus where they disagree.
    """
    out: dict[int, float] = {}
    for i, v in zip(indices, values):
        out[int(i)] = out.get(int(i), 0.0) + float(v)
    return out


# ------------------------------------------------------------------ scoring


def _l2(vec) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def dense_score(q: list[float], d: list[float], metric: str) -> float:
    """nova-bf's convention: LARGER is nearer, for every metric. Euclidean is
    therefore the NEGATED L2 distance (not squared, not the raw distance) —
    the form `compute._scores` returns and the form Qdrant's `EUCLID` score
    has to be negated into before comparison."""
    if metric == "dot":
        return sum(float(a) * float(b) for a, b in zip(q, d))
    if metric == "cosine":
        nq, nd = _l2(q), _l2(d)
        if nq == 0.0 or nd == 0.0:
            return 0.0
        return sum(float(a) * float(b) for a, b in zip(q, d)) / (nq * nd)
    if metric == "euclidean":
        return -math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(q, d)))
    raise ValueError(f"unknown dense metric {metric!r}")


def sparse_overlaps(q: dict[int, float], d: dict[int, float]) -> bool:
    """Do these two sparse vectors share at least one stored token?

    This — not the score — is what decides whether a sparse point is a
    CANDIDATE at all. A sparse search is an inverted-index intersection: a
    document that shares no token with the query is never visited, so it is
    absent from the results rather than present with a score of 0. Qdrant
    works that way and nova-bf reproduces it (`compute._zero_gate_file_ok`
    and the structural indicator path it falls back to).

    The distinction is invisible on strictly-positive data — where no overlap
    and a zero score coincide — and very visible on signed embeddings, where
    a real overlap can dot to a NEGATIVE score and must still outrank the
    non-candidates. Note it is STRUCTURAL: a token stored with the value 0.0
    is still an overlap.
    """
    return any(t in d for t in q)


def sparse_score(q: dict[int, float], d: dict[int, float], metric: str) -> float:
    """Dot over the shared support; cosine divides by both rows' TRUE norms.

    "True" matters: nova-bf truncates each corpus row to the union of the
    queries' token ids before scoring, which cannot change a dot product (the
    dropped dimensions are zero in every query anyway) but WOULD shrink the
    norm if the norm were taken after truncation, inflating cosine. So the
    norm here is over the full row, matching `compute._sparse_file_norms`.
    """
    dot = sum(w * d[t] for t, w in q.items() if t in d)
    if metric == "dot":
        return dot
    if metric == "cosine":
        nq = math.sqrt(sum(w * w for w in q.values()))
        nd = math.sqrt(sum(w * w for w in d.values()))
        if nq == 0.0 or nd == 0.0:
            return 0.0
        return dot / (nq * nd)
    raise ValueError(f"unknown sparse metric {metric!r}")


def maxsim(qtok: list[list[float]], dtok: list[list[float]], metric: str) -> float:
    """Late-interaction MaxSim: for each QUERY token take its best similarity
    against any document token, then sum those maxima over query tokens.

    `cosine` is MaxSim over per-token-normalized vectors (Qdrant's own
    definition of a cosine multivector), not a cosine of anything pooled.
    """
    if metric == "cosine":
        qtok = [_normalize(t) for t in qtok]
        dtok = [_normalize(t) for t in dtok]
    elif metric != "dot":
        raise ValueError(f"unknown multivector metric {metric!r}")
    total = 0.0
    for qt in qtok:
        best = None
        for dt in dtok:
            s = sum(float(a) * float(b) for a, b in zip(qt, dt))
            if best is None or s > best:
                best = s
        total += 0.0 if best is None else best
    return total


def _normalize(vec: list[float]) -> list[float]:
    n = _l2(vec)
    return list(vec) if n == 0.0 else [float(x) / n for x in vec]


# ------------------------------------------------------------------ filters


_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")


def tokens(text) -> list[str]:
    """Qdrant `word`-tokenizer semantics: split the ORIGINAL text on maximal
    runs of non-alphanumerics, then lowercase each token.

    Written against ASCII on purpose. nova-bf splits with RE2's `\\p{L}\\p{N}`
    and lowercases with Arrow, which differ from Python on a known handful of
    non-ASCII codepoints (see `nova_bf.tokenize`'s docstring); the harness
    generates ASCII-only text so this reference is exact rather than
    approximately exact, and that divergence stays where it is documented
    instead of being silently absorbed into a passing test.
    """
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _TOKEN_SPLIT.split(text) if t]


def to_epoch_us(value, fmt: str = "rfc3339") -> int | None:
    """A declared date field's value as int64 epoch microseconds — Qdrant's own
    internal datetime representation, so a nova-bf `range` over a date field
    and a Qdrant `DatetimeRange` are value-for-value comparable."""
    if value is None:
        return None
    if fmt == "epoch_s":
        return int(round(float(value) * 1_000_000))
    if fmt == "epoch_ms":
        return int(round(float(value) * 1_000))
    if fmt == "epoch_us":
        return int(value)
    if fmt == "rfc3339":
        text = value.replace("Z", "+00:00") if isinstance(value, str) else value
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def _as_number(value, field_name: str, date_fields: dict[str, str]):
    """Range comparisons are numeric. A field declared as a date is parsed to
    epoch µs first; anything else is used as-is."""
    if field_name in date_fields:
        return to_epoch_us(value, date_fields[field_name])
    return value


def condition_matches(cond, payload: dict, query_row: dict,
                      date_fields: dict[str, str],
                      query_date_fields: dict[str, str] | None = None) -> bool:
    """Does ONE `FilterCondition` hold for one point?

    `date_fields` declares which CORPUS columns are datetimes, `query_date_fields`
    which QUERIES columns are — the two are separate declarations in the config
    and both get normalized to epoch microseconds at load. A per-query bound
    drawn from a declared query date column therefore has to be parsed here
    too; comparing a µs corpus value against a raw RFC-3339 string is not a
    near-miss, it is a type error, and nova-bf never does it.

    A null payload value never matches anything — the field being absent is
    not a wildcard. That is Qdrant's rule and nova-bf's (`pc.fill_null(mask,
    False)`), and it is the rule an over-eager reference would get wrong by
    letting `None` compare its way through a range.
    """
    query_date_fields = query_date_fields or {}
    value = payload.get(cond.field)

    # --- per-query variants: same predicate, comparison value read from the
    # query's own row instead of a literal in the config.
    if cond.match_from_query is not None:
        wanted = query_row[cond.match_from_query]
        if value is None or wanted is None:
            return False
        if isinstance(wanted, (list, tuple)):
            return any(value == w for w in wanted)
        return value == wanted
    if cond.match_text_from_query is not None:
        phrase = query_row[cond.match_text_from_query]
        if value is None or not isinstance(phrase, str):
            return False
        want = tokens(phrase)
        have = set(tokens(value))
        # An empty phrase constrains nothing on the rows it is applied to,
        # matching an AND-fold over zero tokens.
        return all(t in have for t in want)
    if cond.range_from_query is not None:
        if value is None:
            return False
        num = _as_number(value, cond.field, date_fields)
        r = cond.range_from_query
        for attr, op in (("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")):
            col = getattr(r, attr)
            if col is None:
                continue
            bound = query_row[col]
            if col in query_date_fields:
                bound = to_epoch_us(bound, query_date_fields[col])
            if bound is None or not _cmp(num, op, bound):
                return False
        return True

    # --- static variants
    if cond.match is not None:
        if value is None:
            return False
        wanted = cond.match if isinstance(cond.match, tuple) else (cond.match,)
        return any(value == w for w in wanted)
    if cond.match_text is not None:
        if value is None:
            return False
        have = set(tokens(value))
        return all(t in have for t in tokens(cond.match_text))
    if cond.range is not None:
        if value is None:
            return False
        num = _as_number(value, cond.field, date_fields)
        r = cond.range
        for attr, op in (("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")):
            bound = getattr(r, attr)
            if bound is None:
                continue
            if not _cmp(num, op, bound):
                return False
        return True
    raise ValueError(f"condition on {cond.field!r} sets none of the six predicates")


def _cmp(value, op: str, bound) -> bool:
    if op == ">":
        return value > bound
    if op == ">=":
        return value >= bound
    if op == "<":
        return value < bound
    return value <= bound


def filter_matches(filt, payload: dict, query_row: dict, date_fields: dict[str, str],
                   query_date_fields: dict[str, str] | None = None) -> bool:
    """`must` = AND, `should` = at-least-one, `must_not` = AND-NOT — Qdrant's
    own composition. An EMPTY `should` is no constraint at all (not an OR over
    nothing, which would reject everything); an empty filter keeps every row.
    """
    if filt is None:
        return True
    for cond in filt.must:
        if not condition_matches(cond, payload, query_row, date_fields, query_date_fields):
            return False
    for cond in filt.must_not:
        if condition_matches(cond, payload, query_row, date_fields, query_date_fields):
            return False
    if filt.should:
        if not any(
            condition_matches(c, payload, query_row, date_fields, query_date_fields)
            for c in filt.should
        ):
            return False
    return True


# -------------------------------------------------------------------- top-K


def score_one(query, doc: Doc, vector_type: str, metric: str) -> float | None:
    """This query's score against this point, or None when the point is not a
    candidate for this search at all.

    Non-candidacy is not the same as scoring zero, and the difference shows up
    the moment real scores can be negative:

      * dense — every point is a candidate;
      * multivector — a point with a null or empty token set has nothing to
        MaxSim against, so it is absent rather than scoring 0;
      * sparse — a point sharing no token with the query is never visited by
        the inverted index, so it is absent too (`sparse_overlaps`).
    """
    if vector_type == "dense":
        return None if doc.dense is None else dense_score(query, doc.dense, metric)
    if vector_type == "sparse":
        if doc.sparse is None or not sparse_overlaps(query, doc.sparse):
            return None
        return sparse_score(query, doc.sparse, metric)
    if vector_type == "multivector":
        if not doc.multivector:
            return None
        return maxsim(query, doc.multivector, metric)
    raise ValueError(f"unknown vector_type {vector_type!r}")


def topk(
    query,
    docs: list[Doc],
    *,
    vector_type: str,
    metric: str,
    k: int,
    filt=None,
    query_row: dict | None = None,
    date_fields: dict[str, str] | None = None,
    query_date_fields: dict[str, str] | None = None,
) -> list[tuple[int, float]]:
    """`[(row, score), …]` — the eligible points ranked by score descending,
    truncated to `k`. Ordinal (corpus row) breaks ties, which is nova-bf's
    `tiebreak: ordinal` rule; see this module's docstring for why that
    agreement is not what the parity tests lean on."""
    query_row = query_row or {}
    date_fields = date_fields or {}
    scored = []
    for doc in docs:
        if not filter_matches(filt, doc.payload, query_row, date_fields,
                              query_date_fields):
            continue
        s = score_one(query, doc, vector_type, metric)
        if s is None:
            continue
        scored.append((doc.row, s))
    scored.sort(key=lambda rs: (-rs[1], rs[0]))
    return scored[:k]


# ------------------------------------------------------------------- oracle


class Oracle:
    """`topk` for many (metric, filter) combinations over one corpus, without
    recomputing what does not change.

    A corpus-side filter is a predicate on the POINT, evaluated before scoring
    and independent of the score (that is what makes it a pre-filter rather
    than a re-rank). So the score of a (query, point) pair is the same in every
    case that shares a vector_type and metric, and the eligibility of a point
    is the same in every case that shares a filter. This caches those two
    axes separately and combines them per case.

    It calls exactly the same `score_one` and `filter_matches` as the direct
    path — this is memoization, not a second implementation, and
    `test_oracle_matches_the_direct_path` pins that by checking a sample of
    cases against `topk` itself.
    """

    def __init__(self, docs: list[Doc], queries: list[dict], date_fields: dict[str, str],
                 query_date_fields: dict[str, str] | None = None):
        self.docs = docs
        self.queries = queries
        self.date_fields = date_fields
        self.query_date_fields = query_date_fields or {}
        self._scores: dict[tuple, list[list[float | None]]] = {}
        self._masks: dict[tuple, list[list[bool]]] = {}

    def _score_matrix(self, vector_type: str, metric: str, query_key: str):
        key = (vector_type, metric)
        if key not in self._scores:
            self._scores[key] = [
                [score_one(q[query_key], d, vector_type, metric) for d in self.docs]
                for q in self.queries
            ]
        return self._scores[key]

    def _mask(self, filt):
        # A `Filter` is frozen and hashable by design (see config.py), so it is
        # its own cache key — two spellings of the same filter share a mask.
        key = filt
        if key not in self._masks:
            self._masks[key] = [
                [filter_matches(filt, d.payload, q["payload"], self.date_fields,
                                self.query_date_fields)
                 for d in self.docs]
                for q in self.queries
            ]
        return self._masks[key]

    def topk(self, *, vector_type: str, metric: str, k: int, filt=None,
             queries=None, query_key: str | None = None):
        """`{query_index: [(row, score), …]}` for every query (or just
        `queries`)."""
        query_key = query_key or vector_type
        scores = self._score_matrix(vector_type, metric, query_key)
        mask = self._mask(filt)
        out = {}
        for qi in (range(len(self.queries)) if queries is None else queries):
            ranked = [
                (doc.row, scores[qi][j])
                for j, doc in enumerate(self.docs)
                if mask[qi][j] and scores[qi][j] is not None
            ]
            ranked.sort(key=lambda rs: (-rs[1], rs[0]))
            out[qi] = ranked[:k]
        return out
