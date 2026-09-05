"""The chunked sparse-scoring path (`params.sparse_chunk`), end to end — see
docs/brute-force/sparse-chunked-scoring-2026-09-02.md.

`_sparse_scores` (compute.py) picks a formulation for one corpus slice: when
the dense `(vocab, rows)` corpus operand fits `_SPARSE_SWAP_MAX_DENSE_BYTES`
it densifies and runs one dense GEMM ("fits"); otherwise `params.sparse_chunk`
(default True) decides between splitting the corpus rows into several smaller
dense GEMMs (staying on the branch the doc measured as both faster and far
more reproducible — on CUDA; this file exercises it on CPU, where the branch
is reachable but the nondeterminism it targets is not) or falling back to a
sparse-CSR transpose matmul (`sparse_chunk: false`).

`tests/test_sparse_formulation.py` covers the unit-level contract directly
against `_sparse_scores`, including the tail-slice regression the doc names
('N9'). What this file adds is the same thing `test_parity_kernels.py` adds
for the Triton kernels: proof the branch is correct IN SITU, end to end
through `run_compute`, composed with filters and both oracles — not just
internally consistent with itself.

`_SPARSE_SWAP_MAX_DENSE_BYTES` is read ONCE AT IMPORT from
`NOVA_BF_SPARSE_SWAP_MAX` (see compute.py), so — unlike `sparse_chunk`, which
is a real config field read per run — the "does not fit" case itself has to be
forced by patching the module attribute directly, the same way the unit tests
do.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

import nova_bf.compute as compute_mod

from . import compare, qdrant_ref
from .cases import FILTERS, K
from .runner import run, spec
from .test_parity_matrix import _filter_from_dict

PROBES = [
    ("c_sparse_dot", "sparse", "dot", None),
    ("c_sparse_cos", "sparse", "cosine", FILTERS["compound"]),
]
IDS = [p[0] for p in PROBES]

# corpus.py's sparse vocab is 40 tokens (4-byte float32 = 160 bytes/row), so a
# tiny batch already fits the real 512 MiB default many times over. Both
# numbers below are chosen relative to EACH OTHER, not to that default: the
# budget must starve a `sparse_batch_size`-row batch (forcing "does not fit")
# while still being comfortably wider than one row (forcing >1 chunk once
# `sparse_chunk` actually splits it) — `test_the_budget_actually_forced_the_
# branch` below is what proves this held.
TINY_BUDGET = 40 * 4 * 5  # ~5 rows/chunk


def _specs():
    return [spec(name, vector_type=vt, metric=m, k=K, filter=f)
            for name, vt, m, f in PROBES]


@contextmanager
def _forced_budget(nbytes):
    with patch.object(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", nbytes):
        yield


def _run(ds, *, chunk: bool):
    with _forced_budget(TINY_BUDGET):
        return run(ds, _specs(), out_tag=f"sparsechunk_{chunk}", device="cpu",
                   params={"sparse_batch_size": 17, "sparse_chunk": chunk})


@pytest.fixture(scope="session")
def chunk_runs(ds):
    """Both arms, on CPU: the branch itself is device-agnostic (plain
    `torch.matmul` over CSR sub-tensors), so unlike the Triton kernels this
    needs no GPU to exercise."""
    return {chunk: _run(ds, chunk=chunk) for chunk in (False, True)}


def test_the_budget_actually_forced_the_branch(ds):
    """Keeps the rest of this file honest. `_SPARSE_BRANCHES` is the run's own
    record of which formulation it took: a nonzero `scored_chunked` count with
    a zero `scored_swapped` count is what proves the forced budget actually
    starved the "fits whole" fast path, rather than every comparison below
    passing because both arms quietly took it."""
    compute_mod._reset_sparse_branches()
    with _forced_budget(TINY_BUDGET):
        run(ds, _specs(), out_tag="sparsechunk_probe", device="cpu",
            params={"sparse_batch_size": 17, "sparse_chunk": True})
    assert compute_mod._SPARSE_BRANCHES["scored_chunked"] > 0, \
        "forced budget did not starve the whole-slice fast path"
    assert compute_mod._SPARSE_BRANCHES["scored_swapped"] == 0
    assert compute_mod._SPARSE_BRANCHES["scored_fallback"] == 0


@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_chunked_path_agrees_with_the_naive_oracle(entry, ds, oracle, chunk_runs):
    """The end-to-end correctness claim: composed with filters and sharing,
    not just internally consistent with `_sparse_scores` in isolation."""
    name, vt, metric, fdict = entry
    got = chunk_runs[True][name]
    want = oracle.topk(vector_type=vt, metric=metric, k=K,
                       filt=_filter_from_dict(ds, fdict))
    for qi in sorted(got):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"chunked {name} q{qi} vs naive")


@pytest.mark.qdrant
@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_chunked_path_agrees_with_qdrant(entry, ds, client, collection, chunk_runs):
    """Fidelity to the engine the ground truth actually grades, not just to
    nova-bf's own reference implementation."""
    name, vt, metric, fdict = entry
    got = chunk_runs[True][name]
    want = qdrant_ref.topk(client, collection, ds, vector_type=vt, metric=metric,
                           k=K, filt=_filter_from_dict(ds, fdict))
    for qi in sorted(got):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"chunked {name} q{qi} vs qdrant")


@pytest.mark.parametrize("entry", PROBES, ids=IDS)
def test_chunked_path_agrees_with_the_transpose_fallback_it_replaces(entry, chunk_runs):
    """`sparse_chunk: true` is an alternative route to the SAME "does not fit"
    case `sparse_chunk: false` already handles, not a change of which
    documents are eligible — the two must agree to score tolerance. Not
    asserted bit-identical: they are different formulations (a dense GEMM per
    chunk vs. a sparse-CSR transpose matmul), same as `_sparse_scores`'s
    "swapped" vs. "fallback" forms in test_sparse_formulation.py."""
    name, vt, metric, _f = entry
    chunked = chunk_runs[True][name]
    fallback = chunk_runs[False][name]
    for qi in sorted(fallback):
        compare.assert_scores_agree(
            chunked[qi], fallback[qi], metric=metric,
            label=f"{name} q{qi}: sparse_chunk=true vs sparse_chunk=false")
