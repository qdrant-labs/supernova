"""`match_from_query` type check: exact match only between integers, strings
or booleans (lists of them on the query side); anything else is a config error
at run start, with a message naming the condition and the fix."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import check_match_types

INT = [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint32()]
STR = [pa.string(), pa.large_string(), pa.dictionary(pa.int32(), pa.string())]
BOOL = [pa.bool_()]
BAD = [pa.float32(), pa.float64(), pa.timestamp("us"), pa.date32(), pa.binary()]


def _f(field="c", q="q"):
    return Filter(must=[FilterCondition(field=field, match_from_query=q)])


def _check(ct, qt, **kw):
    check_match_types([("s", _f())], pa.schema([("c", ct)]), pa.schema([("q", qt)]), **kw)


@pytest.mark.parametrize("group", [INT, STR, BOOL])
def test_same_kind_allowed_including_lists(group):
    for ct in group:
        for qt in group:
            _check(ct, qt)
            _check(ct, pa.list_(qt))
            _check(ct, pa.large_list(qt))


@pytest.mark.parametrize("ct,qt", [
    (pa.int64(), pa.float64()), (pa.float64(), pa.int64()), (pa.float32(), pa.float32()),
    (pa.int64(), pa.bool_()), (pa.bool_(), pa.int8()), (pa.string(), pa.int64()),
    (pa.int64(), pa.string()), (pa.string(), pa.bool_()), (pa.int64(), pa.list_(pa.float64())),
    (pa.timestamp("us"), pa.timestamp("us")), (pa.binary(), pa.binary()),
    (pa.uint64(), pa.int64()), (pa.int32(), pa.uint64()),       # float64 compare
    (pa.uint64(), pa.uint64()), (pa.uint64(), pa.list_(pa.uint64())),   # uint64: cast to int64
    (pa.list_(pa.string()), pa.string()),                       # list corpus field
])
def test_mixed_or_unsupported_rejected(ct, qt):
    with pytest.raises(ValueError) as e:
        _check(ct, qt)
    msg = str(e.value)
    assert "`c`" in msg and "`q`" in msg and "range_from_query" in msg and "cast" in msg


def test_null_columns_dates_missing_and_other_conditions_skipped():
    _check(pa.null(), pa.float64())                 # all-null corpus column never matches
    _check(pa.int64(), pa.null())
    _check(pa.timestamp("us"), pa.string(), skip_fields={"c"}, skip_query_cols={"q"})  # both dates
    # columns that don't exist are reported elsewhere
    check_match_types([("s", _f("zz", "q"))], pa.schema([("c", pa.int64())]),
                      pa.schema([("q", pa.float64())]))
    # only match_from_query is checked
    f = Filter(must=[FilterCondition(field="c", range_from_query={"gte": "q"})])
    check_match_types([("s", f)], pa.schema([("c", pa.int64())]), pa.schema([("q", pa.float64())]))
    check_match_types([("s", None)], pa.schema([]), pa.schema([]))


@pytest.mark.parametrize("corpus_date", [True, False])
def test_date_on_one_side_only_rejected(corpus_date):
    kw = {"skip_fields": {"c"}} if corpus_date else {"skip_query_cols": {"q"}}
    with pytest.raises(ValueError, match="only the .* side is a declared date field"):
        _check(pa.string(), pa.string(), **kw)


def test_every_offender_named():
    f = Filter(must=[FilterCondition(field="a", match_from_query="qa")],
               should=[FilterCondition(field="b", match_from_query="qb")],
               must_not=[FilterCondition(field="c", match_from_query="qc")])
    cs = pa.schema([("a", pa.float64()), ("b", pa.int64()), ("c", pa.string())])
    qs = pa.schema([("qa", pa.float64()), ("qb", pa.int64()), ("qc", pa.int64())])
    with pytest.raises(ValueError) as e:
        check_match_types([("mysearch", f)], cs, qs)
    msg = str(e.value)
    assert "`a`" in msg and "`c`" in msg and "`b`" not in msg and "search `mysearch`" in msg


def test_run_compute_fails_fast(tmp_path):
    """The check runs in run_compute before any corpus data is read."""
    from nova_bf.compute import run_compute
    from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                QueriesConfig, SearchSpec)
    cdir = tmp_path / "c"; cdir.mkdir()
    pq.write_table(pa.table({"dense_embedding": pa.array([[1.0, 0.0]] * 3, pa.list_(pa.float32())),
                             "id": ["a", "b", "c"], "n": pa.array([1, 2, 3], pa.int64())}),
                   str(cdir / "f.parquet"))
    pq.write_table(pa.table({"dense_embedding": pa.array([[1.0, 0.0]], pa.list_(pa.float32())),
                             "qid": ["q0"], "qn": pa.array([2.0], pa.float64())}),
                   str(tmp_path / "q.parquet"))
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(tmp_path / "out")),
        searches=[SearchSpec(name="s", vector_type="dense", metric="dot", k=2,
                             filter=_f("n", "qn"))])
    with pytest.raises(ValueError, match="match_from_query compares"):
        run_compute(cfg)


# --- static `match` values vs the corpus column --------------------------------

from nova_bf.filters import check_static_match_types  # noqa: E402


def _static(t, value, **kw):
    f = Filter(must=[FilterCondition(field="c", match=value)])
    check_static_match_types([("s", f)], pa.schema([("c", t)]), **kw)


@pytest.mark.parametrize("t,value", [
    (pa.int64(), 5), (pa.int64(), 5.0), (pa.int64(), 2.5), (pa.uint64(), 2**63 + 5),
    (pa.int8(), (1, 2.0)), (pa.float64(), 5), (pa.float32(), 2.5),
    (pa.string(), "5"), (pa.large_string(), ("a", "b")), (pa.bool_(), True),
    (pa.dictionary(pa.int32(), pa.string()), "a"), (pa.dictionary(pa.int32(), pa.int64()), 7),
    (pa.null(), True), (pa.binary(), "x"), (pa.timestamp("us"), 5),
])
def test_static_match_fitting_values_accepted(t, value):
    _static(t, value)


@pytest.mark.parametrize("t,value", [
    (pa.int64(), True), (pa.int64(), "5"), (pa.uint64(), False), (pa.int32(), (1, True)),
    (pa.float64(), True), (pa.float64(), "2.5"), (pa.string(), 5), (pa.string(), True),
    (pa.string(), ("a", 5.0)), (pa.bool_(), 1), (pa.bool_(), "true"), (pa.bool_(), 0.0),
    (pa.dictionary(pa.int32(), pa.string()), 5), (pa.dictionary(pa.int32(), pa.int64()), "7"),
])
def test_static_match_wrong_kind_rejected(t, value):
    with pytest.raises(ValueError, match="static match values don't fit") as e:
        _static(t, value)
    assert "`s`" in str(e.value) and "`c`" in str(e.value)


def test_static_match_date_fields_missing_columns_and_other_conditions_skipped():
    _static(pa.timestamp("us"), "2020-01-01", skip_fields={"c"})
    check_static_match_types([("s", Filter(must=[FilterCondition(field="zz", match=True)]))],
                             pa.schema([("c", pa.int64())]))
    check_static_match_types([("s", Filter(must=[FilterCondition(field="c", range={"gt": 1})]))],
                             pa.schema([("c", pa.string())]))
    check_static_match_types([("s", None)], pa.schema([]))


def test_static_match_every_offender_named_and_called_from_check_match_types():
    f = Filter(must=[FilterCondition(field="a", match=True)],
               should=[FilterCondition(field="b", match=5)])
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    with pytest.raises(ValueError) as e:
        check_match_types([("s", f)], schema, pa.schema([]))
    msg = str(e.value)
    assert "`a`" in msg and "`b`" in msg and "True" in msg and "5" in msg
