"""Native, GIL-free S3 / local range downloader that writes into caller buffers,
plus a dense-vector parquet fast path (`Client.dense_plan` / `read_dense`)."""

from ._rust_fetch import Client, CorruptParquet, DensePlan, UnsupportedLayout, __version__, chunk_spans

__all__ = ["Client", "CorruptParquet", "DensePlan", "UnsupportedLayout", "__version__", "chunk_spans", "read_dense"]


def read_dense(client, url, column, size=None, chunks_in_flight=8, out=None):
    """Decode top-level list column `column` of parquet `url` into a row-major
    `(n_rows, dim)` numpy array of its STORED dtype (float32 or float16, copied
    bit-for-bit), fetching only that column's bytes.

    `out`: optional preallocated C-contiguous array of the right shape and
    dtype to decode into (e.g. a pinned buffer); a new one is allocated if
    None. Raises UnsupportedLayout when the column is not a layout the native
    decoder reproduces exactly -- the caller should fall back to pyarrow.
    """
    import numpy as np

    plan = client.dense_plan(url, column, size)
    if plan.n_rows == 0:
        return np.zeros((0, 0), dtype=np.float32)  # as pyarrow-based dense_to_2d
    shape, dtype = (plan.n_rows, plan.dim), np.dtype(plan.dtype)
    if out is None:
        out = np.empty(shape, dtype=dtype)
    elif out.shape != shape or out.dtype != dtype or not out.flags.c_contiguous:
        raise ValueError(f"out must be C-contiguous {dtype} {shape}, got {out.dtype} {out.shape}")
    if plan.nbytes:
        client.read_dense_into(plan, out.reshape(-1).view(np.uint8), chunks_in_flight)
    return out
