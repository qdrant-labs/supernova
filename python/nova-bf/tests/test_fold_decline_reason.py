"""`merge_triton._why_declined` — the one line you get when the fold kernel
is skipped.

Deliberately NOT gated on CUDA, unlike `test_tiebreak_fold_kernel.py`: this
message exists precisely for runs where the kernel is unavailable, so its
coverage has to run on machines without one. `_why_declined` is best-effort
and never raises, so plain CPU tensors exercise every branch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import nova_bf.merge_triton as mt


def _decline_inputs(n_q=4, k=3, w=2):
    """Otherwise-valid CPU tensors. `_why_declined` is best-effort and runs
    after the fallback is already decided, so it never needs real CUDA."""
    import torch

    return dict(
        state_key=torch.zeros(n_q, k, dtype=torch.int64),
        state_enc=torch.zeros(n_q, k, dtype=torch.int64),
        part_key=torch.zeros(n_q, w, dtype=torch.int64),
        part_enc=torch.zeros(w, dtype=torch.int64),
        k=k,
    )


@pytest.mark.parametrize("mutate,expect", [
    (lambda t: {"live": t["live"], "thr": None}, "pair"),
    (lambda t: {"live": None, "thr": t["thr"]}, "pair"),
    (lambda t: {"live": t["live"].to(torch.int64), "thr": t["thr"]}, "not torch.uint8"),
    (lambda t: {"live": t["live"][:-1].contiguous(), "thr": t["thr"]}, "one per query row"),
    (lambda t: {"live": t["live"].unsqueeze(0), "thr": t["thr"]}, "not 1-D"),
    (lambda t: {"live": t["live"], "thr": t["thr"].to(torch.int32)}, "not torch.int64"),
    (lambda t: {"live": t["live"], "thr": t["thr"].repeat(2)[::2]}, "not contiguous"),
])
def test_decline_reason_names_the_pruning_gate(monkeypatch, mutate, expect):
    """`available()` grew gates on `live`/`thr`, and the reason string has to
    grow with them. Before this, a decline caused by one of those branches fell
    through to the catch-all "shape or device mismatch" — and since the line is
    logged once per process by design, that one misleading message is all you
    get to diagnose a real slowdown."""
    import torch

    monkeypatch.delenv("NOVA_BF_NO_FOLD_KERNEL", raising=False)
    # `_why_declined` short-circuits when the kernel was never built; this test
    # is about the reason text, not about whether Triton is present.
    monkeypatch.setattr(mt, "_fold", object())

    args = _decline_inputs()
    n_q = args["state_key"].shape[0]
    base = {"live": torch.ones(n_q, dtype=torch.uint8),
            "thr": torch.zeros(n_q, dtype=torch.int64)}
    kw = mutate(base)

    reason = mt._why_declined(**args, **kw)
    assert expect in reason, f"expected {expect!r}, got {reason!r}"
    assert reason != "shape or device mismatch", "fell through to the catch-all"


def test_decline_reason_still_reaches_the_older_cases(monkeypatch):
    """Negative control: with valid pruning inputs the new block must not
    swallow the decline — it should fall through to the pre-existing checks."""
    import torch

    monkeypatch.delenv("NOVA_BF_NO_FOLD_KERNEL", raising=False)
    monkeypatch.setattr(mt, "_fold", object())
    args = _decline_inputs()
    n_q = args["state_key"].shape[0]
    reason = mt._why_declined(
        **args,
        live=torch.ones(n_q, dtype=torch.uint8),
        thr=torch.zeros(n_q, dtype=torch.int64),
    )
    assert "not on CUDA" in reason, reason


def test_decline_reason_unaffected_when_pruning_is_off(monkeypatch):
    """With `thr=None` (pruning disabled) the new block is skipped entirely."""
    import torch

    monkeypatch.delenv("NOVA_BF_NO_FOLD_KERNEL", raising=False)
    monkeypatch.setattr(mt, "_fold", object())
    reason = mt._why_declined(**_decline_inputs())
    assert "not on CUDA" in reason, reason
