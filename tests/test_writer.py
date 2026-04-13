import os
import tempfile

import pytest
import pyarrow.parquet as pq

from vectorforge.models import EmbeddedRecord, SparseEmbedding
from vectorforge.storage.writer import write_batch


def test_write_batch_dense_only():
    records = [
        EmbeddedRecord(
            row_id=0,
            source_row_id=0,
            chunk_id=0,
            chunk_index=0,
            text="hello world",
            dense_embedding=[0.1, 0.2, 0.3],
            columns={"title": "greeting", "url": "http://example.com"},
        ),
        EmbeddedRecord(
            row_id=1,
            source_row_id=0,
            chunk_id=0,
            chunk_index=1,
            text="another record",
            dense_embedding=[0.4, 0.5, 0.6],
            columns={"title": "second", "url": "http://example.com/2"},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_batch(records, tmpdir, batch_id=0)

        assert os.path.exists(path)
        assert path.endswith("batch_00000000.parquet")

        table = pq.read_table(path)
        assert table.num_rows == 2
        assert table.column("text").to_pylist() == ["hello world", "another record"]
        assert table.column("source_row_id").to_pylist() == [0, 0]
        assert table.column("chunk_index").to_pylist() == [0, 1]
        assert table.column("row_id").to_pylist() == [0, 1]

        # Dynamic columns from source data
        assert table.column("title").to_pylist() == ["greeting", "second"]
        assert table.column("url").to_pylist() == ["http://example.com", "http://example.com/2"]

        embeddings = table.column("dense_embedding").to_pylist()
        assert len(embeddings[0]) == 3


def test_write_batch_sparse_only():
    records = [
        EmbeddedRecord(
            row_id=0,
            source_row_id=0,
            chunk_id=0,
            chunk_index=0,
            text="hello",
            sparse_embedding=SparseEmbedding(indices=[0, 5, 10], values=[1.0, 0.5, 0.3]),
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_batch(records, tmpdir, batch_id=0, dense_column=None, sparse_column="sparse_embedding")

        table = pq.read_table(path)
        assert table.num_rows == 1
        assert "sparse_embedding" in table.column_names
        sparse = table.column("sparse_embedding").to_pylist()[0]
        assert sparse["indices"] == [0, 5, 10]
        assert sparse["values"] == pytest.approx([1.0, 0.5, 0.3])


def test_write_batch_both():
    records = [
        EmbeddedRecord(
            row_id=0,
            source_row_id=0,
            chunk_id=0,
            chunk_index=0,
            text="hello",
            dense_embedding=[0.1, 0.2],
            sparse_embedding=SparseEmbedding(indices=[0, 1], values=[1.0, 0.5]),
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_batch(
            records, tmpdir, batch_id=0,
            dense_column="dense_embedding",
            sparse_column="sparse_embedding",
        )

        table = pq.read_table(path)
        assert table.num_rows == 1
        assert "dense_embedding" in table.column_names
        assert "sparse_embedding" in table.column_names
        assert table.column("dense_embedding").to_pylist()[0] == [
            0.10000000149011612,
            0.20000000298023224,
        ]
        sparse = table.column("sparse_embedding").to_pylist()[0]
        assert sparse["indices"] == [0, 1]
