import os

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.models import EmbeddedRecord

SCHEMA = pa.schema([
    pa.field("row_id", pa.int64()),
    pa.field("source_row_id", pa.int64()),
    pa.field("chunk_id", pa.int32()),
    pa.field("chunk_index", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
    pa.field("embedding", pa.list_(pa.float32())),
    pa.field("model", pa.string()),
])


def write_batch(records: list[EmbeddedRecord], output_dir: str, batch_id: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"batch_{batch_id:08d}.parquet"
    path = os.path.join(output_dir, filename)

    table = pa.table({
        "row_id":        [r.row_id for r in records],
        "source_row_id": [r.source_row_id for r in records],
        "chunk_id":      [r.chunk_id for r in records],
        "chunk_index":   [r.chunk_index for r in records],
        "text":          [r.text for r in records],
        "source":        [r.source for r in records],
        "embedding":     [r.embedding for r in records],
        "model":         [r.model for r in records],
    }, schema=SCHEMA)

    pq.write_table(table, path, compression="snappy")
    return path