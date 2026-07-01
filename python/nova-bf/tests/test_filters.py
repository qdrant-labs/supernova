"""Unit tests for `nova_bf.filters.evaluate` — no torch required."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    QueriesConfig,
    RangeCondition,
)
from nova_bf.filters import evaluate


def _table(**cols):
    return pa.table(cols)


def test_match_scalar_equality():
    t = _table(language=["eng", "fra", "eng", "deu"])
    f = Filter(must=[FilterCondition(field="language", match="eng")])
    assert evaluate(f, t).tolist() == [True, False, True, False]


def test_match_any_of_list():
    t = _table(language=["eng", "fra", "spa", "deu"])
    f = Filter(must=[FilterCondition(field="language", match=["eng", "spa"])])
    assert evaluate(f, t).tolist() == [True, False, True, False]


def test_range_single_bound():
    t = _table(cost=[1, 5, 10, 20])
    f = Filter(must=[FilterCondition(field="cost", range=RangeCondition(lt=10))])
    assert evaluate(f, t).tolist() == [True, True, False, False]


def test_range_combined_bounds_within_one_condition():
    t = _table(cost=[1, 5, 10, 20])
    f = Filter(must=[FilterCondition(field="cost", range=RangeCondition(gte=5, lt=20))])
    assert evaluate(f, t).tolist() == [False, True, True, False]


def test_must_is_and():
    t = _table(language=["eng", "eng", "fra", "fra"], cost=[1, 20, 1, 20])
    f = Filter(must=[
        FilterCondition(field="language", match="eng"),
        FilterCondition(field="cost", range=RangeCondition(lt=10)),
    ])
    assert evaluate(f, t).tolist() == [True, False, False, False]


def test_should_is_any_of():
    t = _table(language=["eng", "fra", "deu"])
    f = Filter(should=[
        FilterCondition(field="language", match="eng"),
        FilterCondition(field="language", match="fra"),
    ])
    assert evaluate(f, t).tolist() == [True, True, False]


def test_must_not_excludes():
    t = _table(language=["eng", "fra", "deu"])
    f = Filter(must_not=[FilterCondition(field="language", match="fra")])
    assert evaluate(f, t).tolist() == [True, False, True]


def test_null_payload_value_never_matches():
    t = _table(cost=[1, None, 10])
    f = Filter(must=[FilterCondition(field="cost", range=RangeCondition(lt=100))])
    assert evaluate(f, t).tolist() == [True, False, True]


def test_condition_requires_exactly_one_of_match_or_range():
    with pytest.raises(ValueError):
        FilterCondition(field="cost")
    with pytest.raises(ValueError):
        FilterCondition(field="cost", match=5, range=RangeCondition(lt=10))


def test_range_condition_requires_a_bound():
    with pytest.raises(ValueError):
        RangeCondition()


def test_config_filter_field_is_the_only_declaration_needed():
    # No separate "which columns to read" list — a condition's `field` is itself
    # the declaration, and `.fields()` (what compute.py reads) reflects it directly.
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path="/tmp/corpus"),
        queries=QueriesConfig(path="/tmp/q.parquet"),
        output=OutputConfig(path="/tmp/out"),
        filter=Filter(must=[FilterCondition(field="language", match="eng")]),
    )
    assert cfg.filter.fields() == {"language"}
