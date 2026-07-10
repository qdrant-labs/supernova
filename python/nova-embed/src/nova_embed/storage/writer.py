import os

import pyarrow as pa
import pyarrow.parquet as pq

from nova_embed.models import EmbeddedRecord, OutputKind

SPARSE_EMBEDDING_TYPE = pa.struct(
    [
        pa.field("indices", pa.list_(pa.uint32())),
        pa.field("values", pa.list_(pa.float32())),
    ]
)

DENSE_EMBEDDING_TYPE = pa.list_(pa.float32())
MULTIVECTOR_EMBEDDING_TYPE = pa.list_(pa.list_(pa.float32()))


def _dense_value(v):
    return v


def _sparse_value(v):
    return None if v is None else {"indices": v.indices, "values": v.values}


def _multivector_value(v):
    # Embedders may keep .vectors as a 2D ndarray for efficient downstream math;
    # pyarrow needs a list-of-1D-arrays here, so flatten per row.
    if v is None:
        return None
    vectors = v.vectors
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


_KIND_TO_ARROW = {
    OutputKind.DENSE: (DENSE_EMBEDDING_TYPE, _dense_value),
    OutputKind.SPARSE: (SPARSE_EMBEDDING_TYPE, _sparse_value),
    OutputKind.MULTIVECTOR: (MULTIVECTOR_EMBEDDING_TYPE, _multivector_value),
}


def write_batch(
    records: list[EmbeddedRecord],
    output_dir: str,
    batch_id: int,
    output_specs: list,  # list[OutputSpec] — (name, column, kind) is all we use
    filename_prefix: str = "",
    row_group_size: int | None = None,
) -> str:
    """Write one parquet batch: every source row column (chunk-rewritten where a
    chunker was active), then one typed embedding column per output spec.

    Embedding columns are ALWAYS written with their declared arrow type, even if
    every value in this batch is null (on_empty_input="null") — the schema must
    be identical across batches or downstream readers choke on the mismatch.
    """
    filename = f"{filename_prefix}batch_{batch_id:08d}.parquet"
    path = os.path.join(output_dir, filename)
    # filename_prefix may include '/' (shard_by_rank); ensure the subdir exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []

    # Source row columns — let PyArrow infer types from the data. Embedding
    # column names were checked unique against each other at config time; a
    # collision with a SOURCE column can only be seen here, so check it now.
    row_columns = list(records[0].row.keys()) if records else []
    spec_columns = {spec.column for spec in output_specs}
    collisions = sorted(spec_columns.intersection(row_columns))
    if collisions:
        raise ValueError(
            f"output column(s) {collisions} collide with source columns. "
            f"Set a different output_column on the embedder entry."
        )

    for col in row_columns:
        arr = pa.array([r.row.get(col) for r in records])
        arrays.append(arr)
        fields.append(pa.field(col, arr.type))

    for spec in output_specs:
        arrow_type, convert = _KIND_TO_ARROW[OutputKind(spec.kind)]
        values = [convert(r.embeddings.get(spec.name)) for r in records]
        arrays.append(pa.array(values, type=arrow_type))
        fields.append(pa.field(spec.column, arrow_type))

    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    pq.write_table(table, path, compression="snappy", row_group_size=row_group_size)
    return path
