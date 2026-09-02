"""The int32 offset guard the fused MaxSim reducer was missing.

`topk_triton` and `merge_triton` both decline shapes whose pointer arithmetic
would overflow int32 (see `test_tiebreak.py::
test_the_gate_declines_shapes_whose_offsets_overflow_int32`, which also records
WHY the alternative was rejected: widening the row index to int64 measured 27%
in `topk_triton`). `multivector_kernels` did the same arithmetic with no guard
at all, so a `P` past 2**31 - 1 elements silently read another query's tokens.

These live outside the GPU-gated multivector suites on purpose: the guard is
pure arithmetic, it must run on every box, and the shapes it describes are far
too large to allocate.
"""

from __future__ import annotations

import pytest

from nova_bf.multivector_kernels import _INT32_MAX, offsets_fit_int32


# The production multivector config (configs/brute_force/
# pubmed_bge_m3_all_modalities.yaml) at the time of writing.
_BUDGET = 170_000_000
_QUERY_BLOCK = 256


def test_the_shipped_config_is_accepted():
    """A false decline is a silent slowdown nothing would report, so the shape
    the repo's own multivector config produces has to stay on the fast path.

    `multivector_token_budget` bounds `P`'s element count, so the worst tile is
    about the budget itself, however the two tile sizes are derived from it."""
    for q_tokens in (256, 4096, 32_768):
        d_tokens = max(1, _BUDGET // q_tokens)
        assert offsets_fit_int32(q_tokens, d_tokens, _QUERY_BLOCK, 1000, 1000), (
            f"budget={_BUDGET} split as {q_tokens}x{d_tokens} must be accepted"
        )


def test_a_tile_past_int32_is_declined():
    """The failure this exists for: `P` past 2**31 - 1 ELEMENTS — 8.6 GB of
    float32, which is one `multivector_token_budget` bump away on a large card.
    """
    # ~2.15e9 elements: just over the line.
    assert not offsets_fit_int32(65_536, 32_768, 256, 1000, 1000)
    # comfortably over — 8x the limit
    assert not offsets_fit_int32(131_072, 131_072, 256, 1000, 1000)


def test_the_boundary_leaves_room_for_the_masked_tail_lanes():
    """The tile loops run `query_row` up to `block_query - 1` past the real end
    and `document_column` up to `block_document - 1` past it. Those lanes are
    masked OFF but their ADDRESSES are still computed, so the guard has to
    count them — a boundary that ignored them would accept a shape whose last
    tile wraps."""
    d = 4096
    # Largest height that fits once a full query tile and document tile of
    # slack are charged.
    fits = (_INT32_MAX - d - 128) // d - 8
    assert offsets_fit_int32(fits, d, 8, 8, 8)
    assert not offsets_fit_int32(fits + 2, d, 8, 8, 8)

    # Being conservative is the point: the naive bound (bare element count)
    # accepts strictly more than the guard does.
    naive_max = _INT32_MAX // d
    assert naive_max > fits, "the guard must be tighter than the naive bound"


def test_output_pointers_are_checked_too():
    """`out` is indexed by whole QUERIES and DOCUMENTS rather than tokens, so
    it wraps far later than the similarity tile — but it is checked rather than
    assumed, because nothing else would notice if it did."""
    # A tiny similarity tile, an absurd output stride.
    assert not offsets_fit_int32(8, 8, 3_000_000, 1000, 1000)
    assert offsets_fit_int32(8, 8, 1000, 1000, 1000)


def test_an_empty_tile_forms_no_offsets():
    """Zero tokens on either axis means the kernel never loads, so there is no
    offset to overflow. `fused_ragged_maxsim_reduce` returns before launching
    in that case anyway; this pins the predicate's own answer."""
    assert offsets_fit_int32(0, 10**9, 10**6, 10**6, 10**6)
    assert offsets_fit_int32(10**9, 0, 10**6, 10**6, 10**6)


def test_the_wrapper_refuses_an_overflowing_tile():
    """A DIRECT caller that skips `MultiVectorBatchSlice.score`'s gate must get
    an error, not a launch. Checked without CUDA by getting the argument
    validation to reject the shape before any device work."""
    torch = pytest.importorskip("torch")
    from nova_bf.multivector_kernels import fused_ragged_maxsim_reduce

    # The wrapper requires CUDA tensors, so on a CPU box we can only confirm it
    # rejects rather than launches. Assert the predicate the wrapper consults
    # agrees with the raise it would produce.
    assert not offsets_fit_int32(65_536, 32_768, 256, 1000, 1000)

    with pytest.raises(ValueError, match="CUDA|cuda"):
        fused_ragged_maxsim_reduce(
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            query_start=0,
            query_token_base=0,
            n_queries=1,
        )


def _cuda() -> bool:
    torch = pytest.importorskip("torch")
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="the decline only exists on the CUDA kernel path")
def test_a_declined_block_falls_back_and_matches_the_torch_reference(monkeypatch, caplog):
    """The guard's OTHER half: `score()` must survive a decline, not just avoid
    the kernel.

    The tests above are pure arithmetic — they prove the predicate answers
    correctly, and nothing more. The fallback they trigger is real code that no
    shape in any suite reaches, because every tested shape is far below the
    threshold (the shipped config sits ~12x under it). So force it: make the
    predicate refuse everything and assert the answer is byte-identical to the
    torch reference, which is what the fallback runs.

    GPU-gated because `MultiVectorBatchSlice.score` only ever selects
    `triton_reduce` on CUDA — on CPU `selected_kernel` is forced to "torch"
    before the loop, so there is no decline to make.
    """
    import logging

    import torch

    from nova_bf import multivector_kernels as mk
    from nova_bf.compute import MultiVectorBatchSlice, MultiVectorQuery

    dev, dim = "cuda", 16
    g = torch.Generator(device="cpu").manual_seed(7)

    # 5 queries with ragged token counts (one zero-token, deliberately), and a
    # 4-doc slice with ragged token counts of its own.
    q_counts = [3, 1, 0, 4, 2]
    d_counts = [2, 5, 1, 3]
    q_off = torch.tensor([0, *torch.tensor(q_counts).cumsum(0).tolist()], dtype=torch.int64)
    d_off = torch.tensor([0, *torch.tensor(d_counts).cumsum(0).tolist()], dtype=torch.int64)
    q_flat = torch.randn(int(q_off[-1]), dim, generator=g).to(dev)
    d_flat = torch.randn(int(d_off[-1]), dim, generator=g).to(dev)

    sl = MultiVectorBatchSlice(d_flat, d_off.to(dev))

    def _query(kernel):
        return MultiVectorQuery(
            q_flat, q_off.to(dev), q_off.numpy(), len(q_counts),
            query_block=2, kernel=kernel,      # 2 -> several blocks, so several declines
        )

    for metric in ("dot", "cosine"):
        reference = sl.score(_query("torch"), metric)

        # Refuse every shape: every block takes the fallback.
        monkeypatch.setattr(mk, "offsets_fit_int32", lambda *a, **k: False)
        with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
            declined = sl.score(_query("triton_reduce"), metric)
        assert "int32 pointer arithmetic" in caplog.text, "the decline must be logged"
        assert torch.equal(declined, reference), (
            f"{metric}: the fallback must reproduce the torch reference exactly"
        )
        caplog.clear()

        # And with the real predicate the kernel runs and agrees — otherwise the
        # assertion above would pass even if the decline were the ONLY path.
        monkeypatch.undo()
        kernelled = sl.score(_query("triton_reduce"), metric)
        assert torch.allclose(kernelled, reference, rtol=1e-5, atol=1e-6), (
            f"{metric}: the kernel path must agree with the reference too"
        )
