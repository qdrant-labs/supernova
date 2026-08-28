"""Unit tests for `nova_sweep.report.build_row` — how nova-storm's `--json`
summary is merged into a report row. Storm's recall buckets are nested
`{n, mean}` objects, so they must be flattened to dotted scalar columns (not
written as struct columns) to keep `sweep_results.parquet` a flat table.
"""

from __future__ import annotations

from nova_sweep.report import build_row

# A representative nova-storm `--json` summary: scalar fields plus the nested
# recall buckets and the `empty_ground_truth` count.
STORM_SUMMARY = {
    "requests": 100,
    "errors": 0,
    "qps": 1234.5,
    "p50_ms": 1.1,
    "full_recall": {"n": 60, "mean": 0.95},
    "short_recall": {"n": 20, "mean": 0.80},
    "total_recall": {"n": 80, "mean": 0.9125},
    "empty_ground_truth": 5,
    "filter_overreturn": 12,
}


def _row(summary):
    return build_row(
        data_layout={"_name": "default"},
        data_layout_name="default",
        collection_name="c",
        index_variant={"_name": "default"},
        search={"_name": "default"},
        summary=summary,
        reindex_seconds=0.0,
        search_seconds=0.0,
        ok=True,
        error=None,
    )


def test_scalar_summary_fields_pass_through_unchanged():
    row = _row(STORM_SUMMARY)
    assert row["requests"] == 100
    assert row["qps"] == 1234.5
    assert row["empty_ground_truth"] == 5
    assert row["filter_overreturn"] == 12


def test_recall_buckets_flatten_to_dotted_scalar_columns():
    row = _row(STORM_SUMMARY)
    # Nested {n, mean} become dotted keys, and the raw nested dict is gone --
    # so pyarrow writes plain double/int64 columns, not a struct column.
    assert row["full_recall.mean"] == 0.95
    assert row["full_recall.n"] == 60
    assert row["short_recall.mean"] == 0.80
    assert row["total_recall.n"] == 80
    assert "full_recall" not in row
    assert not any(isinstance(v, dict) for v in row.values())


def test_absent_recall_is_simply_absent_not_an_error():
    # No ground truth configured -> storm emits null recall buckets. `None`
    # values pass through as null columns; no `.mean`/`.n` keys are invented.
    summary = {"requests": 10, "full_recall": None, "total_recall": None, "empty_ground_truth": 0}
    row = _row(summary)
    assert row["requests"] == 10
    assert row["full_recall"] is None
    assert "full_recall.mean" not in row


def test_missing_summary_records_no_storm_columns():
    row = _row(None)
    assert row["ok"] is True
    assert "total_recall.mean" not in row
    assert "requests" not in row
