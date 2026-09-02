"""CUDA kernel for exact multivector (late-interaction) scoring.

This module is imported lazily by :mod:`nova_bf.compute`, so merge-only and
CPU installations do not need Triton.  The kernel folds a cuBLAS-materialized
token-similarity matrix into per-(query, document) MaxSim scores, using the
same prefix-sum offset arrays as the PyTorch reference path (the
``triton_reduce`` backend).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(
    # None of these change the generated code — keeping them out of the
    # specialization key stops Triton from JIT-recompiling per slice shape.
    # Adaptive ragged slicing (`_ragged_batch_ranges`) deliberately varies the
    # per-slice document count, so `n_documents` as a compile-time constant
    # meant one ~0.3-1s compile per distinct slice size, per query-block size.
    do_not_specialize=[
        "query_start",
        "query_token_base",
        "similarity_row_stride",
        "n_queries",
        "n_documents",
        "out_row_stride",
    ],
)
def _fused_ragged_reduce_kernel(
    similarity_ptr,
    query_offsets_ptr,
    document_offsets_ptr,
    output_ptr,
    query_start,
    query_token_base,
    similarity_row_stride,
    n_queries,
    n_documents,
    out_row_stride,
    BLOCK_QUERY: tl.constexpr,
    BLOCK_DOCUMENT: tl.constexpr,
):
    """Fold a materialized token GEMM directly into query/document scores.

    The input is a row-major ``query_tokens x document_tokens`` matrix from
    cuBLAS. Each program owns one query/document pair, reads its contiguous
    ragged rectangle, computes max over document tokens and sum over query
    tokens, and writes one scalar. No atomics or token-by-document intermediate
    are required.
    """
    pair_index = tl.program_id(0)
    local_query = pair_index // n_documents
    document_index = pair_index % n_documents
    query_index = query_start + local_query

    query_begin = tl.load(query_offsets_ptr + query_index) - query_token_base
    query_end = tl.load(query_offsets_ptr + query_index + 1) - query_token_base
    document_begin = tl.load(document_offsets_ptr + document_index)
    document_end = tl.load(document_offsets_ptr + document_index + 1)

    query_lane = tl.arange(0, BLOCK_QUERY)
    document_lane = tl.arange(0, BLOCK_DOCUMENT)
    score = tl.zeros((), dtype=tl.float32)

    query_tile = query_begin
    while query_tile < query_end:
        query_row = query_tile + query_lane
        query_valid = query_row < query_end
        row_max = tl.full((BLOCK_QUERY,), float("-inf"), tl.float32)

        document_tile = document_begin
        while document_tile < document_end:
            document_column = document_tile + document_lane
            document_valid = document_column < document_end
            values = tl.load(
                similarity_ptr
                + query_row[:, None] * similarity_row_stride
                + document_column[None, :],
                mask=query_valid[:, None] & document_valid[None, :],
                other=float("-inf"),
            )
            row_max = tl.maximum(row_max, tl.max(values, axis=1))
            document_tile += BLOCK_DOCUMENT

        score += tl.sum(tl.where(query_valid, row_max, 0.0), axis=0)
        query_tile += BLOCK_QUERY

    score = tl.where(
        (query_end > query_begin) & (document_end > document_begin),
        score,
        float("-inf"),
    )
    # `out_row_stride` (not `n_documents`) so the caller can hand a row-slice
    # view of a larger output matrix and skip a separate device-to-device copy.
    tl.store(output_ptr + local_query * out_row_stride + document_index, score)


# Triton forms these offsets in int32, so oversized shapes can wrap and read 
# the wrong query's tokens without error. Decline those shapes instead.
_INT32_MAX = (1 << 31) - 1


def offsets_fit_int32(
    n_query_tokens: int,
    n_doc_tokens: int,
    n_queries: int,
    n_documents: int,
    out_row_stride: int,
    block_query: int = 8,
    block_document: int = 128,
) -> bool:
    """Return whether all kernel pointer offsets fit in int32. 
    Includes tile padding because masked lanes still form addresses. 
    Pure arithmetic so oversized cases can be tested without a GPU. 
    """
    if n_query_tokens <= 0 or n_doc_tokens <= 0:
        return True
    
    tile = (n_query_tokens + block_query) * n_doc_tokens + n_doc_tokens + block_document
    out_max = max(0, n_queries - 1) * max(0, out_row_stride) + max(0, n_documents - 1)
    return tile <= _INT32_MAX and out_max <= _INT32_MAX


def fused_ragged_maxsim_reduce(
    similarity,
    query_offsets,
    document_offsets,
    *,
    query_start: int,
    query_token_base: int,
    n_queries: int,
    out=None,
    block_query: int = 8,
    block_document: int = 128,
    num_warps: int = 4,
):
    """Reduce a cuBLAS token-similarity matrix into ragged MaxSim scores.

    `out` (optional) is written in place and must be a CUDA float32
    `(n_queries, n_documents)` tensor whose last dimension is contiguous —
    a row-slice view of a larger matrix qualifies, which is exactly the
    caller's use (`out[qs:qe]`), sparing a separate device-to-device copy.
    """
    import torch

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (similarity, query_offsets, document_offsets)
    ):
        raise TypeError("fused ragged reduction inputs must be torch tensors")
    if not all(
        tensor.is_cuda for tensor in (similarity, query_offsets, document_offsets)
    ):
        raise ValueError("fused ragged reduction requires CUDA tensors")
    if similarity.dtype != torch.float32 or similarity.ndim != 2:
        raise TypeError("fused ragged reduction requires a 2D float32 similarity tensor")
    if query_offsets.dtype != torch.int64 or document_offsets.dtype != torch.int64:
        raise TypeError("fused ragged reduction requires int64 offset tensors")
    if (
        query_offsets.device != similarity.device
        or document_offsets.device != similarity.device
    ):
        raise ValueError("fused ragged reduction inputs must be on one CUDA device")
    if not similarity.is_contiguous():
        similarity = similarity.contiguous()

    n_documents = document_offsets.numel() - 1
    if out is None:
        out = torch.empty(
            (n_queries, n_documents), dtype=torch.float32, device=similarity.device
        )
    else:
        if out.shape != (n_queries, n_documents):
            raise ValueError(
                f"fused ragged reduction out shape {tuple(out.shape)} != "
                f"({n_queries}, {n_documents})"
            )
        if out.dtype != torch.float32 or not out.is_cuda:
            raise TypeError("fused ragged reduction out must be a CUDA float32 tensor")
        if n_documents > 0 and out.stride(1) != 1:
            raise ValueError("fused ragged reduction out must have a contiguous last dim")
    if n_queries == 0 or n_documents == 0:
        return out

    if not offsets_fit_int32(
        similarity.shape[0], similarity.shape[1], n_queries, n_documents,
        out.stride(0), block_query, block_document,
    ):
        # Refuse rather than launch
        raise ValueError(
            f"fused ragged reduction: a {tuple(similarity.shape)} similarity "
            f"tile makes the kernel's int32 pointer arithmetic overflow "
            f"(> {_INT32_MAX} elements). Lower params.multivector_token_budget "
            "(it sizes this matrix directly), or set params.multivector_kernel="
            "'torch', whose reference path has no such limit."
        )

    _fused_ragged_reduce_kernel[(n_queries * n_documents,)](
        similarity,
        query_offsets,
        document_offsets,
        out,
        query_start,
        query_token_base,
        similarity.stride(0),
        n_queries,
        n_documents,
        out.stride(0),
        BLOCK_QUERY=block_query,
        BLOCK_DOCUMENT=block_document,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
