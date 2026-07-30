from __future__ import annotations

from nova_sweep.report import build_row


def test_build_row_keeps_new_storm_operation_fields() -> None:
    row = build_row(
        data_layout={"name": "base"},
        data_layout_name="base",
        collection_name="c",
        index_variant={"_name": "iv"},
        search={"_name": "sv"},
        summary={
            "requests": 100,
            "errors": 3,
            "query_requests": 70,
            "upsert_requests": 20,
            "delete_requests": 10,
            "query_errors": 1,
            "upsert_errors": 1,
            "delete_errors": 1,
            "total_recall": {"n": 80, "mean": 0.88},
        },
        reindex_seconds=1.5,
        search_seconds=2.5,
        ok=True,
        error=None,
    )
    assert row["requests"] == 100
    assert row["query_requests"] == 70
    assert row["upsert_requests"] == 20
    assert row["delete_requests"] == 10
    assert row["query_errors"] == 1
    assert row["upsert_errors"] == 1
    assert row["delete_errors"] == 1
    assert row["total_recall.n"] == 80
    assert row["total_recall.mean"] == 0.88
