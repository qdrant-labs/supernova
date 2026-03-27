import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.models import ChunkResult

SCHEMA = pa.schema([
    pa.field("row_id", pa.int64()),
    pa.field("chunk_id", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
    pa.field("embedding", pa.list_(pa.float32())),
    pa.field("model", pa.string()),
    pa.field("payload", pa.string()),
])


def write_chunk(result: ChunkResult, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"chunk_{result.chunk_id:08d}.parquet"
    path = os.path.join(output_dir, filename)

    rows = result.records
    table = pa.table({
        "row_id":    [r.row_id for r in rows],
        "chunk_id":  [r.chunk_id for r in rows],
        "text":      [r.text for r in rows],
        "source":    [r.source for r in rows],
        "embedding": [r.embedding for r in rows],
        "model":     [r.model for r in rows],
        "payload":   [json.dumps(r.payload) for r in rows],
    }, schema=SCHEMA)

    pq.write_table(table, path, compression="snappy")
    return path
