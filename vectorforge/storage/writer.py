import os

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.models import EmbeddedRecord

# Fixed columns that vectorforge always adds, with explicit types
FIXED_SCHEMA = [
    pa.field("row_id", pa.int64()),
    pa.field("source_row_id", pa.int64()),
    pa.field("chunk_id", pa.int32()),
    pa.field("chunk_index", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32())),
]

FIXED_COLUMN_NAMES = {f.name for f in FIXED_SCHEMA}


def write_batch(records: list[EmbeddedRecord], output_dir: str, batch_id: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"batch_{batch_id:08d}.parquet"
    path = os.path.join(output_dir, filename)

    # Fixed columns with explicit types
    data = {
        "row_id":        [r.row_id for r in records],
        "source_row_id": [r.source_row_id for r in records],
        "chunk_id":      [r.chunk_id for r in records],
        "chunk_index":   [r.chunk_index for r in records],
        "text":          [r.text for r in records],
        "embedding":     [r.embedding for r in records],
    }
    table = pa.table(data, schema=pa.schema(FIXED_SCHEMA))

    # Dynamic columns — let PyArrow infer types from the data
    if records and records[0].columns:
        for col_name in records[0].columns:
            if col_name not in FIXED_COLUMN_NAMES:
                values = [r.columns.get(col_name) for r in records]
                table = table.append_column(col_name, pa.array(values))

    pq.write_table(table, path, compression="snappy")
    return path
