"""The matrix under test: every filter shape × every modality/metric.

`FILTERS` enumerates the filter language — each of the six condition kinds,
each of the three groups, both the static and the per-query form of the four
that have one, plus the shapes where they compose. `MODALITIES` enumerates the
seven scored configurations (dense × 3 metrics, sparse × 2, multivector × 2).
`CASES` is the cross-product: 7 × the filter table, every cell checked against
BOTH oracles.

The cross-product is the point. A filter is evaluated once per corpus file and
shared by every search that names it, and all searches of one vector_type
share a single batch grid — so "does `match_text` work" and "does `match_text`
work for the multivector search sharing a pass with two sparse ones" are
different questions, and only the second is the one production asks.

Filters are written as plain dicts, exactly as they appear in a YAML config
(dates as RFC-3339 strings, not epoch microseconds) — `runner.build_config`
puts them through the same normalization `load_config` does.
"""

from __future__ import annotations

# name -> filter dict (or None). Names are also part of the nova-bf output
# filename, so they stay [A-Za-z0-9_-].
FILTERS: dict[str, dict | None] = {
    # --- no filter: the control. Every parity failure elsewhere has to be
    # attributable to the filter, which requires knowing the unfiltered search
    # is right.
    "nofilter": None,

    # --- static, one condition per kind
    "match": {"must": [{"field": "language", "match": "eng"}]},
    "matchany": {"must": [{"field": "category", "match": ["news", "paper"]}]},
    "rangeint": {"must": [{"field": "views", "range": {"gte": 2000, "lt": 8000}}]},
    "rangefloat": {"must": [{"field": "rating", "range": {"gt": 2.5}}]},
    "matchtext": {"must": [{"field": "title", "match_text": "vector search"}]},
    # A declared date field: the bound is an RFC-3339 string here and epoch µs
    # by the time nova-bf sees it, against Qdrant's DatetimeRange.
    "datetime": {"must": [{"field": "published_at",
                           "range": {"gte": "2014-01-01T00:00:00Z",
                                     "lt": "2015-01-01T00:00:00Z"}}]},
    # `tier` is null on a third of the corpus — this pins "a null payload
    # value never matches" on a field a filter actually reads.
    "nullable": {"must": [{"field": "tier", "match": "gold"}]},

    # --- the other two groups
    "mustnot": {"must_not": [{"field": "language", "match": "eng"}]},
    "should": {"should": [{"field": "category", "match": "news"},
                          {"field": "category", "match": "blog"}]},
    # must AND should AND must_not at once, over three different fields and
    # three different condition kinds.
    "compound": {
        "must": [{"field": "views", "range": {"gte": 1000}}],
        "should": [{"field": "language", "match": "eng"},
                   {"field": "language", "match": "fra"}],
        "must_not": [{"field": "tier", "match": "gold"}],
    },

    # --- per-query: the mask is (n_queries, rows), and every query wants a
    # different subset (see corpus._build_queries), so a per-query filter that
    # collapsed to one shared mask would fail rather than pass by accident.
    "pqmatch": {"must": [{"field": "language", "match_from_query": "q_language"}]},
    "pqmatchany": {"must": [{"field": "language",
                             "match_from_query": "q_languages"}]},
    "pqrange": {"must": [{"field": "views",
                          "range_from_query": {"gte": "q_min_views",
                                               "lt": "q_max_views"}}]},
    "pqtext": {"must": [{"field": "title", "match_text_from_query": "q_phrase"}]},
    "pqdatetime": {"must": [{"field": "published_at",
                             "range_from_query": {"gte": "q_after"}}]},
    # Static and per-query conditions in the same filter, across two groups —
    # the case where the mask starts 1-D and is promoted to 2-D mid-evaluation.
    "pqmixed": {
        "must": [{"field": "category", "match": ["news", "blog", "paper"]},
                 {"field": "language", "match_from_query": "q_language"}],
        "must_not": [{"field": "views", "range": {"lt": 500}}],
    },
}

# (vector_type, metric) — every scored configuration nova-bf supports.
# Euclidean is dense-only by construction (`SearchSpec._no_euclidean_for_non_dense`).
MODALITIES = (
    ("dense", "dot"),
    ("dense", "cosine"),
    ("dense", "euclidean"),
    ("sparse", "dot"),
    ("sparse", "cosine"),
    ("multivector", "dot"),
    ("multivector", "cosine"),
)

K = 25  # < the smallest filtered result set, so every case is a real ranking


class Case:
    """One cell of the matrix: a nova-bf search, and the two oracle queries
    that must agree with it."""

    __slots__ = ("name", "vector_type", "metric", "filter_name", "filter_dict", "k")

    def __init__(self, vector_type, metric, filter_name, filter_dict, k=K):
        self.name = f"{vector_type[:2]}{metric[:3]}_{filter_name}"
        self.vector_type = vector_type
        self.metric = metric
        self.filter_name = filter_name
        self.filter_dict = filter_dict
        self.k = k

    def spec(self) -> dict:
        from .runner import spec

        return spec(self.name, vector_type=self.vector_type, metric=self.metric,
                    k=self.k, filter=self.filter_dict)

    @property
    def id(self) -> str:
        return f"{self.vector_type}-{self.metric}-{self.filter_name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Case {self.id}>"


CASES = [
    Case(vt, metric, fname, fdict)
    for vt, metric in MODALITIES
    for fname, fdict in FILTERS.items()
]

CASES_BY_NAME = {c.name: c for c in CASES}
