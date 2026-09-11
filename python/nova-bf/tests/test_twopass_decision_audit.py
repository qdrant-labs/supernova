"""The decision audit: grade every fp16 decision, not just the safe ones.

`audit_live_rows` — the continuous, always-on check — only ever sees rows the
bound KEPT. Those rows are exact-scored either way, so "N rows audited, zero
violations" is drawn entirely from the decisions where correctness did not
depend on the bound. The rows that were PRUNED, the only place a wrong bound
can lose a result, were never looked at.

`audit_decisions` runs both passes over the whole slice and grades all four
outcomes against exact scores:

    correct_prune   denied, and the slice really held nothing.  Correct.
    FALSE PRUNE     denied, but a candidate was there.          A LOST RESULT.
    correct_live    kept for rerun, and it was needed.          Correct.
    wasted_live     kept for rerun, nothing was there.          Safe, pure cost.

Only `FALSE PRUNE` is a bug. `wasted_live` is what `eps`'s slack costs, and it
is the half of the matrix nothing else in the suite can see.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from nova_bf import twopass


@pytest.fixture(autouse=True)
def _clean():
    twopass.reset()
    yield
    twopass.reset()


def _slice(tops, thr, live):
    """A corpus of two columns whose scores under `dot` are read straight off Q.

    C is the 2x2 identity, so `Q @ C.T == Q` and a row's exact top is
    `max(q0, q1)`. Nothing here exercises the bound — the audit's job is to
    compare a decision that has ALREADY been made against ground truth — so the
    decision (`live`) is handed in directly rather than produced by pass one.
    """
    Q = torch.zeros(len(tops), 2, dtype=torch.float32)
    for i, t in enumerate(tops):
        Q[i, 0] = t
        Q[i, 1] = t - 1.0          # the second column is never the max
    C = torch.eye(2, dtype=torch.float32)
    return twopass.audit_decisions(
        Q, C, "dot", torch.tensor(live),
        torch.tensor(thr, dtype=torch.float32),
    )


def test_each_of_the_four_outcomes_is_classified():
    # thr = 2.0 throughout; a row's exact top is its first coordinate.
    r = _slice(
        tops=[1.0, 3.0, 3.0, 1.0],
        thr=[2.0, 2.0, 2.0, 2.0],
        live=[False, False, True, True],
    )
    assert r["checked"] == 4
    assert r["correct_prune"] == 1     # top 1.0 < 2.0, denied.  Right call.
    assert r["false_prune"] == 1       # top 3.0 >= 2.0, denied.  LOST.
    assert r["correct_live"] == 1      # top 3.0 >= 2.0, kept.   Right call.
    assert r["wasted_live"] == 1       # top 1.0 < 2.0, kept.    Pure cost.


def test_a_false_prune_is_recorded_and_disables_the_two_pass():
    """A lost result must not be a statistic the run then ignores.

    The audit is a verification tool, so the useful response to finding a
    pruned row that held a candidate is to stop pruning for the rest of the
    run — every later slice then takes the exact path — and to say so loudly.
    """
    _slice(tops=[5.0], thr=[2.0], live=[False])
    st = twopass.stats()
    assert st["dead_audit_violations"] == 1
    assert st["dead_audit_worst"] == pytest.approx(3.0)   # 5.0 - 2.0
    assert not twopass.enabled()
    assert "decision audit" in (st["unavailable"] or "")


def test_a_row_tying_its_threshold_should_have_lived():
    """Ground truth is `top >= thr`, not `top > thr`.

    The pruning rule is `dead iff upper < thr` STRICTLY, so a candidate that
    exactly ties the threshold can still take the slot on the tie-break. If the
    audit used `>` it would call that prune correct and the one decision most
    likely to be wrong at the boundary would be the one it refused to check.
    """
    assert _slice(tops=[2.0], thr=[2.0], live=[False])["false_prune"] == 1
    assert _slice(tops=[2.0], thr=[2.0], live=[True])["correct_live"] == 1
    twopass.reset()
    # A hair below the threshold is a correct prune, so the boundary is real
    # and the test above is not just asserting that everything is a violation.
    assert _slice(tops=[1.999], thr=[2.0], live=[False])["correct_prune"] == 1


def test_a_slice_that_pruned_nothing_is_still_graded():
    """"Kept every row" is a set of decisions too.

    The earlier `audit_dead_rows` skipped a slice with no prunes — there were
    no dead rows to look at. That silently threw away the whole `wasted_live`
    column, which is exactly the measurement that says how much the bound's
    slack costs.
    """
    r = _slice(tops=[1.0, 1.0, 9.0], thr=[2.0, 2.0, 2.0], live=[True] * 3)
    assert r["correct_prune"] == 0 and r["false_prune"] == 0
    assert r["wasted_live"] == 2 and r["correct_live"] == 1


def test_non_finite_rows_are_excluded_rather_than_counted_as_violations():
    """A NaN score is the two-pass's OTHER safety mechanism, not a bug.

    Rows whose exact top or threshold is not finite are forced live and never
    pruned, so the bound makes no claim about them. Counting them would make
    the audit fire on the one path that is already safe by construction.
    """
    r = _slice(tops=[float("nan"), 3.0], thr=[2.0, 2.0], live=[False, True])
    assert r["checked"] == 1
    assert r["false_prune"] == 0
    assert r["correct_live"] == 1
    assert twopass.stats()["dead_audit_violations"] == 0


def test_an_empty_slice_grades_nothing_instead_of_reporting_a_clean_pass():
    assert _slice(tops=[], thr=[], live=[]) == {}
    assert twopass.stats()["dead_audit_members"] == 0


def test_dead_audit_rate_reads_the_environment(monkeypatch):
    monkeypatch.delenv("NOVA_BF_TWOPASS_DEAD_AUDIT", raising=False)
    assert twopass.dead_audit_rate() == 0          # off by default: it costs
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "4")
    assert twopass.dead_audit_rate() == 4
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "-3")
    assert twopass.dead_audit_rate() == 0
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "not a number")
    assert twopass.dead_audit_rate() == 0


def test_an_allocation_failure_in_certification_leaves_the_run_alive():
    """A transient OOM must not kill a multi-hour rank.

    The certification GEMM is full-height `n_q x corpus_rows` — the largest
    allocation the two-pass makes — and it fires on the first full-width slice
    of every new `cert_key`, i.e. at file transitions and stored-width changes.
    It had no OOM handling, while `approx_rowmax` and `_verify_shape_locked`
    both treat allocation failure as recoverable. An uncaught raise here
    propagates out of `run_compute`, and a rank that dies loses its partial —
    which makes the merge refuse the whole directory.

    `False` is the INCONCLUSIVE verdict the caller already handles: stay
    uncertified, do not prune, retry next slice.
    """
    import torch
    from nova_bf import compute as compute_mod

    real = compute_mod._scores

    def oom_on_the_full_height_call(Q, C, metric, q_norms=None,
                                    scale_in_packer=False):
        raise RuntimeError("CUDA error: out of memory")

    Q = torch.randn(8, 64)
    C = torch.randn(32, 64)
    compute_mod._scores = oom_on_the_full_height_call
    try:
        why = compute_mod._certify_two_pass(
            Q, C, "cosine", C.norm(dim=1).reciprocal(),
            Q.norm(dim=1).reciprocal(), Q.norm(dim=1), torch.float32)
    finally:
        compute_mod._scores = real
    assert why is False, (
        f"an allocation failure must be INCONCLUSIVE (False), not a "
        f"certification verdict; got {why!r}")

    # A non-allocation error is a real error and must still propagate.
    def real_error(*a, **kw):
        raise RuntimeError("something genuinely wrong")

    compute_mod._scores = real_error
    try:
        with pytest.raises(RuntimeError, match="genuinely wrong"):
            compute_mod._certify_two_pass(
                Q, C, "cosine", C.norm(dim=1).reciprocal(),
                Q.norm(dim=1).reciprocal(), Q.norm(dim=1), torch.float32)
    finally:
        compute_mod._scores = real


def test_an_allocation_failure_in_the_audit_skips_the_slice_not_the_run():
    """A verification tool must never be the thing that kills the run.

    The audit's GEMM is the full-height exact one the two-pass exists to
    avoid, run IN ADDITION to the narrowed one — the likeliest allocation in
    the module to fail — and it is opt-in, so its failure says nothing about
    the bound. It skips the slice and counts it.
    """
    import torch
    from nova_bf import compute as compute_mod

    real = compute_mod._scores
    compute_mod._scores = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("CUDA error: out of memory"))
    try:
        got = _slice(tops=[1.0, 3.0], thr=[2.0, 2.0], live=[False, True])
    finally:
        compute_mod._scores = real
    assert got == {}
    st = twopass.stats()
    assert st["dead_audit_oom"] == 1
    assert st["dead_audit_members"] == 0
    assert st["dead_audit_violations"] == 0
    assert twopass.enabled(), "an audit OOM must not disable the two-pass"


def test_certification_refuses_a_negative_infinite_upper_bound():
    """`-inf` is the one non-finite `upper` that is NEVER safe.

    The liveness test is `~(upper < thr)`, so `+inf` and NaN keep a row LIVE
    — the safe direction, and the two-pass's other safety mechanism doing its
    job. `-inf < thr` is true for every finite threshold, so such a row is
    pruned unconditionally, whatever its true score is.

    `torch.isfinite` masks all three out of the certification check
    identically. That made the gate skip exactly the value it most needed to
    look at. `upper_bounds` does force non-finite row maxima to `+inf` today,
    so this is not reachable through it — but a certification gate that
    ASSUMES an invariant instead of checking it is not a gate.
    """
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    twopass.reset()
    Q = torch.randn(6, 64)
    C = torch.randn(16, 64)
    real = twopass.upper_bounds

    def with_a_neg_inf_row(*a, **kw):
        upper, approx, eps = real(*a, **kw)
        upper = upper.clone()
        upper[2] = float("-inf")
        return upper, approx, eps

    twopass.upper_bounds = with_a_neg_inf_row
    try:
        why = compute_mod._certify_two_pass(
            Q, C, "cosine", C.norm(dim=1).reciprocal(),
            Q.norm(dim=1).reciprocal(), Q.norm(dim=1), torch.float32)
    finally:
        twopass.upper_bounds = real

    assert isinstance(why, str), (
        f"a -inf upper bound must REFUSE the configuration, not be excluded "
        f"from the check; got {why!r}")
    assert "-inf" in why


def test_certification_still_tolerates_the_safe_non_finite_uppers():
    """`+inf` and NaN keep a row live, so they are excluded, not refused.

    The fix above must not turn the safe direction into a refusal — that would
    make any corpus with one degenerate row uncertifiable and take the whole
    feature down for a reason that is the guard working correctly.
    """
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    for bad in (float("inf"), float("nan")):
        twopass.reset()
        Q = torch.randn(6, 64)
        C = torch.randn(16, 64)
        real = twopass.upper_bounds

        def poisoned(*a, _b=bad, **kw):
            upper, approx, eps = real(*a, **kw)
            upper = upper.clone()
            upper[2] = _b
            return upper, approx, eps

        twopass.upper_bounds = poisoned
        try:
            why = compute_mod._certify_two_pass(
                Q, C, "cosine", C.norm(dim=1).reciprocal(),
                Q.norm(dim=1).reciprocal(), Q.norm(dim=1), torch.float32)
        finally:
            twopass.upper_bounds = real
        assert why is None, (
            f"upper={bad} keeps the row live, so it must not refuse the "
            f"configuration; got {why!r}")


def _certify_with(top_value=None, upper_value=None):
    """Run certification on a clean slice, optionally poisoning one row of
    `upper` (via `upper_bounds`) or one row of the exact `top` (via
    `_scores`)."""
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    twopass.reset()
    Q = torch.randn(6, 64)
    C = torch.randn(16, 64)
    real_ub, real_sc = twopass.upper_bounds, compute_mod._scores

    def ub(*a, **kw):
        upper, approx, eps = real_ub(*a, **kw)
        if upper_value is not None:
            upper = upper.clone()
            upper[2] = upper_value
        return upper, approx, eps

    def sc(*a, **kw):
        out = real_sc(*a, **kw)
        if top_value is not None:
            out = out.clone()
            out[2, :] = -1e30
            out[2, 0] = top_value
        return out

    twopass.upper_bounds, compute_mod._scores = ub, sc
    try:
        return compute_mod._certify_two_pass(
            Q, C, "cosine", C.norm(dim=1).reciprocal(),
            Q.norm(dim=1).reciprocal(), Q.norm(dim=1), torch.float32)
    finally:
        twopass.upper_bounds, compute_mod._scores = real_ub, real_sc


def test_an_infinite_exact_top_against_a_finite_upper_is_a_violation():
    """The bound provably failed to dominate — and it used to be excluded.

    Masking on `torch.isfinite(top)` dropped exactly this row: an exact top of
    `+inf` with a finite `upper` means pass one saw nothing unusual and would
    prune a row whose true score beats every threshold. Testing
    `~(upper >= top)` rather than `upper < top` catches it, because
    `finite >= +inf` is False.
    """
    why = _certify_with(top_value=float("inf"))
    assert isinstance(why, str), (
        f"an exact top of +inf above a finite upper bound is a violation, "
        f"not a row to skip; got {why!r}")
    assert "does not hold" in why


def test_a_nan_exact_top_against_a_finite_upper_is_inconclusive():
    """Not a violation, and not something to certify around either.

    A finite `upper` means the row is prunable, while the exact pass produces
    NaN for it — a byte-identity difference against the one-pass run. `False`
    leaves the configuration uncertified so it never prunes; the cost is
    speed, which is the trade made everywhere else here.
    """
    why = _certify_with(top_value=float("nan"))
    assert why is False, (
        f"a NaN exact top under a finite upper must be INCONCLUSIVE, neither "
        f"certified nor reported as a bound failure; got {why!r}")


def test_a_clean_slice_still_certifies():
    """The three refusals above must not swallow the ordinary case."""
    assert _certify_with() is None
