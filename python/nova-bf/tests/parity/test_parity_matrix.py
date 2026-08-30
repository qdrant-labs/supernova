"""The main event: every filter × every modality, against both oracles.

Each cell of `cases.CASES` is asserted three ways:

  * `test_matches_naive` — nova-bf agrees with the plain-Python reference.
    This is the SEMANTIC check: it fails when nova-bf has the meaning of a
    metric, a modality or a filter condition wrong, and it runs everywhere,
    with no server.
  * `test_matches_qdrant` — nova-bf agrees with a live Qdrant exact search
    under the translated filter. This is the FIDELITY check: nova-bf's whole
    job is grading Qdrant recall, so ground truth Qdrant's own exact search
    disagrees with is wrong by definition.
  * `test_eligible_rows_are_exactly_the_filter_predicate` — an independent
    guard that the filter did something. Without it a filter that silently
    matched every row would sail through both parity checks (both engines
    would return the same unfiltered ranking and agree perfectly).

The third one deserves the emphasis: parity between two engines is only
evidence if the thing being compared is actually constrained. Two engines can
agree beautifully on the wrong answer.
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare, naive, qdrant_ref

ALL_CASES = cases_mod.CASES
IDS = [c.id for c in ALL_CASES]


@pytest.mark.parametrize("case", ALL_CASES, ids=IDS)
def test_matches_naive(case, ds, oracle, matrix_run, device):
    """nova-bf vs. the plain-Python reference, on every query."""
    got = matrix_run[case.name]
    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_model(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] {case.id} q{qi}: nova-bf vs naive")


@pytest.mark.qdrant
@pytest.mark.parametrize("case", ALL_CASES, ids=IDS)
def test_matches_qdrant(case, ds, matrix_run, client, collection, device):
    """nova-bf vs. a live Qdrant exact search under the same filter."""
    got = matrix_run[case.name]
    want = qdrant_ref.topk(client, collection, ds, vector_type=case.vector_type,
                           metric=case.metric, k=case.k,
                           filt=_filter_model(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] {case.id} q{qi}: nova-bf vs qdrant")


@pytest.mark.parametrize("case", ALL_CASES, ids=IDS)
def test_eligible_rows_are_exactly_the_filter_predicate(case, ds, matrix_run, device):
    """Every returned row satisfies the filter, and the result is only short of
    `k` when the filter genuinely left fewer eligible candidates.

    Both halves matter. The first catches a filter that lets extra rows
    through; the second catches one that drops rows it should have kept, which
    the first would happily accept (a filter matching nothing passes "every
    returned row matches" vacuously).
    """
    filt = _filter_model(case, ds)
    for qi, query in enumerate(ds.queries):
        eligible = [
            doc for doc in ds.docs
            if naive.filter_matches(filt, doc.payload, query["payload"],
                                    ds.date_fields, ds.query_date_fields)
            and naive.score_one(query[case.vector_type], doc,
                                case.vector_type, case.metric) is not None
        ]
        hits = matrix_run[case.name][qi]
        for row, _ in hits:
            doc = ds.doc(row)
            assert naive.filter_matches(
                filt, doc.payload, query["payload"], ds.date_fields,
                ds.query_date_fields), (
                f"[{device}] {case.id} q{qi}: returned row {row} "
                f"(payload {doc.payload}) does not satisfy the filter")
        assert len(hits) == min(case.k, len(eligible)), (
            f"[{device}] {case.id} q{qi}: returned {len(hits)} hits but "
            f"{len(eligible)} rows are eligible and k={case.k}")


def test_the_filters_actually_discriminate(ds):
    """A guard on the FIXTURE, not on nova-bf: every filter in the table must
    leave a non-trivial subset — enough rows to fill a top-K, but not the whole
    corpus. A filter that matched everything (or nothing) would make its whole
    row of the matrix vacuous, and that failure is silent in every other test
    here."""
    from .cases import FILTERS, K

    for name, fdict in FILTERS.items():
        if fdict is None:
            continue
        filt = _filter_from_dict(ds, fdict)
        counts = [
            sum(naive.filter_matches(filt, d.payload, q["payload"], ds.date_fields,
                                     ds.query_date_fields)
                for d in ds.docs)
            for q in ds.queries
        ]
        assert min(counts) > K, (
            f"filter {name!r} leaves only {min(counts)} rows for some query — "
            f"too few to rank a top-{K} against")
        assert max(counts) < len(ds.docs), (
            f"filter {name!r} matches every row for some query — it constrains "
            "nothing, so every parity check on it is vacuous")


def test_oracle_matches_the_direct_path(ds, oracle):
    """The oracle caches score matrices and filter masks separately (see
    `naive.Oracle`); this pins that the shortcut equals the direct
    `naive.topk` over the raw docs, so the memoization cannot drift into being
    a second, subtly different implementation."""
    for case in ALL_CASES[::17]:   # a spread across modalities and filters
        filt = _filter_model(case, ds)
        cached = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                             k=case.k, filt=filt)
        for qi, query in enumerate(ds.queries):
            direct = naive.topk(
                query[case.vector_type], ds.docs, vector_type=case.vector_type,
                metric=case.metric, k=case.k, filt=filt,
                query_row=query["payload"], date_fields=ds.date_fields,
                query_date_fields=ds.query_date_fields)
            compare.assert_identical(
                cached[qi], direct, label=f"{case.id} q{qi}: oracle vs direct")


# --------------------------------------------------------------------------


def _filter_model(case, ds):
    """The case's filter as the validated `Filter` model nova-bf itself will
    hold — including the date-bound normalization, so the oracles compare
    against the same instants nova-bf does rather than against the raw YAML
    strings."""
    return _filter_from_dict(ds, case.filter_dict)


def _filter_from_dict(ds, fdict):
    if fdict is None:
        return None
    from .runner import build_config, spec

    cfg = build_config(
        ds, [spec("probe", filter=fdict)], out_tag="unused")
    return cfg.searches[0].filter
