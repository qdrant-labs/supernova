import hashlib
import os
import uuid

import numpy as np
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


_KIND_TO_ARROW = {
    OutputKind.DENSE: (DENSE_EMBEDDING_TYPE, _dense_value),
    OutputKind.SPARSE: (SPARSE_EMBEDDING_TYPE, _sparse_value),
}


def _build_multivector_array(records, name: str) -> pa.Array:
    """Build a ``list<list<float32>>`` column from per-record ``(n_tokens, dim)``
    float32 ndarrays WITHOUT going through Python lists.

    The obvious ``ndarray.tolist()`` (which we used to do) reifies every float as
    a ~28-byte Python float object — an ~8x blow-up over the raw float32 buffer —
    and the whole ``flush_threshold`` batch is materialised at once, so a
    multivector (colbert) flush OOM-kills the worker (each record is ~1.3 MB raw
    but ~11 MB as Python floats; a 2k-record flush is ~2.7 GB raw vs ~22 GB).

    Instead we concatenate the raw float32 buffers (near zero-copy into Arrow)
    and hand-build the two nested offset layers: inner list = one 1024-vector per
    token, outer list = one record. ``None`` (empty-input policy = null) becomes a
    null outer entry. Result is byte-identical to the old path (verified in
    tests) at ~1x the raw size.
    """
    arrays: list[np.ndarray | None] = []
    for r in records:
        v = r.embeddings.get(name)
        arrays.append(None if v is None else np.asarray(v.vectors, dtype=np.float32))

    non_null = [a for a in arrays if a is not None]
    dim = int(non_null[0].shape[1]) if non_null else 1
    total_tokens = int(sum(a.shape[0] for a in non_null))
    flat = (
        np.concatenate([a.reshape(-1) for a in non_null])
        if non_null
        else np.empty(0, dtype=np.float32)
    )

    # inner: list<float32>, one entry per token, each exactly `dim` floats
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, total_tokens * dim + 1, dim, dtype=np.int32)),
        pa.array(flat, type=pa.float32()),
    )

    # outer: list<list<float32>>, one entry per record; null where the record's
    # multivector is None (offsets stay flat across a null so it spans nothing)
    outer_offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    has_null = False
    cum = 0
    for i, a in enumerate(arrays):
        if a is None:
            has_null = True
        else:
            cum += a.shape[0]
        outer_offsets[i + 1] = cum
    mask = (
        pa.array([a is None for a in arrays], type=pa.bool_()) if has_null else None
    )
    return pa.ListArray.from_arrays(pa.array(outer_offsets), inner, mask=mask)


def _content_uuid(path: str) -> uuid.UUID:
    """Deterministic UUID for a file, derived from its bytes (blake2b-128).

    Chunked read: batch files can be hundreds of MB and never need to be in
    memory at once. blake2b hashes at GB/s — noise next to embedding time.
    """
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return uuid.UUID(bytes=h.digest())


def write_batch(
    records: list[EmbeddedRecord],
    output_dir: str,
    batch_id: int,
    output_specs: list,  # list[OutputSpec] — (name, column, kind) is all we use
    filename_prefix: str = "",
    row_group_size: int | None = None,
    content_addressed: bool = False,
    shard_buckets: int | None = None,
) -> str:
    """Write one parquet batch: every source row column (chunk-rewritten where a
    chunker was active), then one typed embedding column per output spec.

    Embedding columns are ALWAYS written with their declared arrow type, even if
    every value in this batch is null (on_empty_input="null") — the schema must
    be identical across batches or downstream readers choke on the mismatch.

    Output layout (see PipelineConfig): with ``content_addressed`` the file is
    named by its content hash (rank/batch counters and prefix dropped — the
    name is globally unique by construction); with ``shard_buckets`` it lands
    in a hash-derived bucket subdir. Both are applied by renaming AFTER the
    parquet is fully written, so the hash covers the final bytes.
    """
    filename = f"{filename_prefix}batch_{batch_id:08d}.parquet"
    path = os.path.join(output_dir, filename)
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
        kind = OutputKind(spec.kind)
        if kind == OutputKind.MULTIVECTOR:
            # numpy-native build — a per-value .tolist() here OOMs the flush (see
            # _build_multivector_array). Type is MULTIVECTOR_EMBEDDING_TYPE.
            arrays.append(_build_multivector_array(records, spec.name))
            fields.append(pa.field(spec.column, MULTIVECTOR_EMBEDDING_TYPE))
        else:
            arrow_type, convert = _KIND_TO_ARROW[kind]
            values = [convert(r.embeddings.get(spec.name)) for r in records]
            arrays.append(pa.array(values, type=arrow_type))
            fields.append(pa.field(spec.column, arrow_type))

    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    pq.write_table(table, path, compression="snappy", row_group_size=row_group_size)

    if content_addressed or shard_buckets:
        file_uuid = _content_uuid(path)
        final_name = f"{file_uuid}.parquet" if content_addressed else filename
        subdir = ""
        if shard_buckets:
            # bucket from the same hash: uniform spread, no rank coordination
            width = max(3, len(str(shard_buckets - 1)))
            bucket = int.from_bytes(file_uuid.bytes[:8], "big") % shard_buckets
            subdir = f"{bucket:0{width}d}"
        final_path = os.path.join(output_dir, subdir, final_name)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.replace(path, final_path)
        return final_path

    return path
