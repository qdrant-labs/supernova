import os
import tempfile

import pyarrow.parquet as pq

from vectorforge.models import ChunkResult, EmbeddedRecord
from vectorforge.storage.writer import write_chunk


def test_write_chunk_creates_parquet():
    result = ChunkResult(
        chunk_id=0,
        records=[
            EmbeddedRecord(
                row_id=0,
                chunk_id=0,
                text="hello world",
                source="test",
                embedding=[0.1, 0.2, 0.3],
                model="test-model",
                payload={"key": "value"},
            ),
            EmbeddedRecord(
                row_id=1,
                chunk_id=0,
                text="another record",
                source="test",
                embedding=[0.4, 0.5, 0.6],
                model="test-model",
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_chunk(result, tmpdir)

        assert os.path.exists(path)
        assert path.endswith("chunk_00000000.parquet")

        table = pq.read_table(path)
        assert table.num_rows == 2
        assert table.column("text").to_pylist() == ["hello world", "another record"]
        assert table.column("model").to_pylist() == ["test-model", "test-model"]
        assert table.column("row_id").to_pylist() == [0, 1]

        embeddings = table.column("embedding").to_pylist()
        assert len(embeddings[0]) == 3
