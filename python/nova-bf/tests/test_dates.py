"""Unit tests for nova_bf.dates — the datetime -> epoch-µs parsing primitives."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.dates import (
    convert_table_date_columns,
    normalize_date_fields,
    parse_scalar_epoch_us,
    to_epoch_us_array,
)


def _ref_us(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


# --- normalize_date_fields ---------------------------------------------------

def test_normalize_none_empty():
    assert normalize_date_fields(None) == {}
    assert normalize_date_fields([]) == {}


def test_normalize_list_defaults_to_rfc3339():
    assert normalize_date_fields(["a", "b"]) == {"a": "rfc3339", "b": "rfc3339"}


def test_normalize_dict_keeps_format_and_fills_blank():
    assert normalize_date_fields({"a": "%Y%m%d", "b": None, "c": ""}) == {
        "a": "%Y%m%d", "b": "rfc3339", "c": "rfc3339",
    }


# --- parse_scalar_epoch_us ---------------------------------------------------

def test_scalar_rfc3339_z():
    assert parse_scalar_epoch_us("2013-05-18T05:48:54Z") == 1368856134000000
    assert parse_scalar_epoch_us("2013-05-18T05:48:54Z") == _ref_us("2013-05-18T05:48:54Z")


def test_scalar_rfc3339_offset_equals_utc():
    # 05:48:54+02:00 is the same instant as 03:48:54Z
    assert parse_scalar_epoch_us("2013-05-18T07:48:54+02:00") == _ref_us("2013-05-18T05:48:54Z")


def test_scalar_naive_treated_as_utc():
    assert parse_scalar_epoch_us("2013-05-18T05:48:54") == _ref_us("2013-05-18T05:48:54Z")


def test_scalar_strptime_pattern():
    assert parse_scalar_epoch_us("20130518", "%Y%m%d") == _ref_us("2013-05-18T00:00:00Z")


def test_scalar_epoch_scales():
    assert parse_scalar_epoch_us(1000, "epoch_s") == 1_000_000_000
    assert parse_scalar_epoch_us(1000, "epoch_ms") == 1_000_000
    assert parse_scalar_epoch_us(1000, "epoch_us") == 1000


def test_scalar_bad_value_raises():
    with pytest.raises(ValueError):
        parse_scalar_epoch_us("not-a-date")


def test_scalar_epoch_us_large_value_stays_exact():
    # 2^53 + 1 is not representable in float64; epoch_us must not route through it
    big = 9_007_199_254_740_993
    assert parse_scalar_epoch_us(big, "epoch_us") == big
    assert parse_scalar_epoch_us(str(big), "epoch_us") == big


# --- to_epoch_us_array -------------------------------------------------------

def test_array_rfc3339_with_null():
    col = pa.array(["2013-05-18T05:48:54Z", None, "2020-12-31T23:59:59Z"])
    out = to_epoch_us_array(col, "rfc3339")
    assert out.type == pa.int64()
    assert out.to_pylist() == [
        _ref_us("2013-05-18T05:48:54Z"), None, _ref_us("2020-12-31T23:59:59Z"),
    ]


def test_array_null_becomes_nan_in_numpy():
    col = pa.array(["2013-05-18T05:48:54Z", None])
    arr = to_epoch_us_array(col, "rfc3339").to_numpy(zero_copy_only=False).astype(np.float64)
    assert arr[0] == float(_ref_us("2013-05-18T05:48:54Z"))
    assert np.isnan(arr[1])  # null -> NaN -> compares False, matching the numeric path


def test_array_scalar_and_array_agree():
    s = "2013-05-18T05:48:54Z"
    (val,) = to_epoch_us_array(pa.array([s]), "rfc3339").to_pylist()
    assert val == parse_scalar_epoch_us(s)


def test_array_native_timestamp_input():
    ts = pa.array([datetime(2013, 5, 18, 5, 48, 54, tzinfo=timezone.utc)],
                  type=pa.timestamp("us", tz="UTC"))
    assert to_epoch_us_array(ts, "rfc3339").to_pylist() == [_ref_us("2013-05-18T05:48:54Z")]


def test_array_strptime_pattern():
    col = pa.array(["20130518", "20201231"])
    assert to_epoch_us_array(col, "%Y%m%d").to_pylist() == [
        _ref_us("2013-05-18T00:00:00Z"), _ref_us("2020-12-31T00:00:00Z"),
    ]


def test_array_epoch_seconds_rescaled():
    col = pa.array([1, 2], type=pa.int64())
    assert to_epoch_us_array(col, "epoch_s").to_pylist() == [1_000_000, 2_000_000]


def test_array_epoch_us_large_value_stays_exact():
    big = 9_007_199_254_740_993  # 2^53 + 1
    out = to_epoch_us_array(pa.array([big], type=pa.int64()), "epoch_us")
    assert out.type == pa.int64()
    assert out.to_pylist() == [big]  # not corrupted by a float64 round-trip


def test_array_unparseable_raises():
    with pytest.raises(Exception):
        to_epoch_us_array(pa.array(["nope"]), "rfc3339")


# --- convert_table_date_columns ---------------------------------------------

def test_convert_table_replaces_only_declared():
    t = pa.table({
        "date": pa.array(["2013-05-18T05:48:54Z", None]),
        "text": pa.array(["a", "b"]),
    })
    out = convert_table_date_columns(t, {"date": "rfc3339"})
    assert out.schema.field("date").type == pa.int64()
    assert out.schema.field("text").type == pa.string()  # untouched
    assert out.column("date").to_pylist() == [_ref_us("2013-05-18T05:48:54Z"), None]


def test_convert_table_absent_field_is_noop():
    t = pa.table({"text": pa.array(["a"])})
    out = convert_table_date_columns(t, {"missing": "rfc3339"})
    assert out.column("text").to_pylist() == ["a"]


def test_convert_table_empty_mapping_returns_same():
    t = pa.table({"x": pa.array([1])})
    assert convert_table_date_columns(t, {}) is t
