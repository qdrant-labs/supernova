import os

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.models import EmbeddedRecord

# Fixed columns that vectorforge always adds
FIXED_COLUMNS = ["row_id", "source_row_id", "chunk_id", "chunk_index", "text", "embedding"]


def _infer_arrow_type(values: list) -> pa.DataType:
    """Infer a PyArrow type from a list of Python values."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            return pa.bool_()
        if isinstance(v, int):
            return pa.int64()
        if isinstance(v, float):
            return pa.float64()
        if isinstance(v, list):
            return pa.list_(pa.float32())
        return pa.string()
    return pa.string()


def write_batch(records: list[EmbeddedRecord], output_dir: str, batch_id: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"batch_{batch_id:08d}.parquet"
    path = os.path.join(output_dir, filename)

    # Fixed columns
    data = {
        "row_id":        [r.row_id for r in records],
        "source_row_id": [r.source_row_id for r in records],
        "chunk_id":      [r.chunk_id for r in records],
        "chunk_index":   [r.chunk_index for r in records],
        "text":          [r.text for r in records],
        "embedding":     [r.embedding for r in records],
    }

    # Dynamic columns from source data
    if records and records[0].columns:
        for col_name in records[0].columns:
            if col_name not in FIXED_COLUMNS:
                data[col_name] = [r.columns.get(col_name) for r in records]

    # Build schema: fixed types for known columns, inferred for the rest
    fields = [
        pa.field("row_id", pa.int64()),
        pa.field("source_row_id", pa.int64()),
        pa.field("chunk_id", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
    ]
    for col_name in data:
        if col_name not in FIXED_COLUMNS:
            fields.append(pa.field(col_name, _infer_arrow_type(data[col_name])))

    schema = pa.schema(fields)
    table = pa.table(data, schema=schema)
    pq.write_table(table, path, compression="snappy")
    return path