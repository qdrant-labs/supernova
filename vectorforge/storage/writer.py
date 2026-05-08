import os

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.models import EmbeddedRecord

SPARSE_EMBEDDING_TYPE = pa.struct(
    [
        pa.field("indices", pa.list_(pa.uint32())),
        pa.field("values", pa.list_(pa.float32())),
    ]
)

MULTIVECTOR_EMBEDDING_TYPE = pa.list_(pa.list_(pa.float32()))


def write_batch(
    records: list[EmbeddedRecord],
    output_dir: str,
    batch_id: int,
    dense_column: str | None = "dense_embedding",
    sparse_column: str | None = None,
    multivector_column: str | None = None,
    rendered_text_column: str = "text",
    filename_prefix: str = "",
    row_group_size: int | None = None,
) -> str:
    filename = f"{filename_prefix}batch_{batch_id:08d}.parquet"
    path = os.path.join(output_dir, filename)
    # filename_prefix may include '/' (shard_by_rank); ensure the subdir exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        rendered_text_column: [r.text for r in records],
    }

    schema_fields = [pa.field(rendered_text_column, pa.string())]

    # Dense embedding column
    if dense_column and records and records[0].dense_embedding is not None:
        data[dense_column] = [r.dense_embedding for r in records]
        schema_fields.append(pa.field(dense_column, pa.list_(pa.float32())))

    # Sparse embedding column
    if sparse_column and records and records[0].sparse_embedding is not None:
        data[sparse_column] = [
            {"indices": r.sparse_embedding.indices, "values": r.sparse_embedding.values}
            for r in records
        ]
        schema_fields.append(pa.field(sparse_column, SPARSE_EMBEDDING_TYPE))

    # Multivector embedding column (N vectors of D floats per row, N varies).
    # Embedders may keep .vectors as a 2D ndarray for efficient downstream math;
    # pyarrow needs a list-of-1D-arrays here, so flatten per row.
    if multivector_column and records and records[0].multivector_embedding is not None:

        def _to_list_of_rows(v):
            # accepts 2D ndarray or list[list[float]]; pyarrow wants per-vector objects
            if hasattr(v, "tolist"):
                return v.tolist()
            return v

        data[multivector_column] = [
            _to_list_of_rows(r.multivector_embedding.vectors) for r in records
        ]
        schema_fields.append(pa.field(multivector_column, MULTIVECTOR_EMBEDDING_TYPE))

    table = pa.table(data, schema=pa.schema(schema_fields))

    # Dynamic columns -- let PyArrow infer types from the data.
    # Skip columns that collide with anything we've already written:
    # the ID columns, any embedding columns, and the rendered-text column.
    skip_names = {dense_column, sparse_column, multivector_column, rendered_text_column}
    if records and records[0].columns:
        for col_name in records[0].columns:
            if col_name not in skip_names:
                values = [r.columns.get(col_name) for r in records]
                table = table.append_column(col_name, pa.array(values))

    pq.write_table(table, path, compression="snappy", row_group_size=row_group_size)
    return path
