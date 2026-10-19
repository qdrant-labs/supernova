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


@pytest.fixture(autouse=True, params=["slice", "tiles"])
def _clean(request, monkeypatch):
    # Both pass-two modes: the audits and the `_twopass_prepare` call sites
    # below must behave the same whichever granularity decides liveness.
    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", request.param)
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

    def _neg_inf_row(res):
        upper, approx, eps = res
        upper = upper.clone()
        upper[2] = float("-inf")
        return upper, approx, eps

    twopass.upper_bounds = _stub_upper_bounds(real, _neg_inf_row)
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

        def _poison(res, _b=bad):
            upper, approx, eps = res
            upper = upper.clone()
            upper[2] = _b
            return upper, approx, eps

        twopass.upper_bounds = _stub_upper_bounds(real, _poison)
        try:
            why = compute_mod._certify_two_pass(
                Q, C, "cosine", C.norm(dim=1).reciprocal(),
                Q.norm(dim=1).reciprocal(), Q.norm(dim=1), torch.float32)
        finally:
            twopass.upper_bounds = real
        assert why is None, (
            f"upper={bad} keeps the row live, so it must not refuse the "
            f"configuration; got {why!r}")


def _stub_upper_bounds(real, mutate):
    """Wrap `upper_bounds` honouring the `with_outcome` contract.

    `upper_bounds` returns `(result, outcome)` when asked for the outcome, so a
    stub that returns the bare result makes `_certify_two_pass` fail to unpack.
    Three separate stubs in this file hand-rolled the old shape; this is the
    one place that knows the contract.
    """
    from nova_bf import twopass

    def wrapped(*a, **kw):
        want = kw.get("with_outcome", False)
        res = real(*a, **{k: v for k, v in kw.items() if k != "with_outcome"})
        res = mutate(res)
        if not want:
            return res
        return res, twopass.PassOneOutcome(twopass.PASS_ONE_CPU_WIDENED)

    return wrapped


def _certify_with(top_value=None, upper_value=None, outcome=None):
    """Run certification on a clean slice, optionally poisoning one row of
    `upper` (via `upper_bounds`), one row of the exact `top` (via `_scores`),
    or the pass-one OUTCOME that `upper_bounds` reports.

    `outcome` is what lets a test drive `_certify_two_pass`'s three-way
    decision directly. Without it the INCONCLUSIVE arm is unreachable on CPU
    (the accumulator is ours there), which is how a full revert of that arm
    passed the entire suite.
    """
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    twopass.reset()
    Q = torch.randn(6, 64)
    C = torch.randn(16, 64)
    real_ub, real_sc = twopass.upper_bounds, compute_mod._scores

    def ub(*a, **kw):
        # Mirror the real contract: with_outcome=True returns (result, outcome).
        want_outcome = kw.get("with_outcome", False)
        got = real_ub(*a, **{k: v for k, v in kw.items() if k != "with_outcome"})
        upper, approx, eps = got
        if upper_value is not None:
            upper = upper.clone()
            upper[2] = upper_value
        res = (upper, approx, eps)
        if not want_outcome:
            return res
        return res, (outcome if outcome is not None
                     else twopass.PassOneOutcome(twopass.PASS_ONE_CPU_WIDENED))

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


# --- certification: INCONCLUSIVE vs a verdict ---------------------------------
#
# `_certify_two_pass` used to infer "cuBLAS ran" from `used_fused == False`,
# which cannot tell "the fused kernel declined" from "no kernel ran at all".
# On a corpus with degenerate rows the guards refuse the probe slice before any
# kernel launches, and that was reported as a cuBLAS verdict -- disabling
# pruning for a whole 102,400-row run over 70 zero-norm rows (nytimes-256).
#
# These are BEHAVIOURAL: they drive the real function. Source-text assertions
# about the call site were mutation-blind -- inverting the condition left them
# green while restoring the bug.

def test_streak_is_separate_from_the_unchecked_shape_streak():
    """Reusing `_UNCHECKED_STREAK` would let two unrelated conditions add up to
    a cap neither reached, and `note_checked_slice` would clear this one on an
    unrelated event."""
    twopass.reset()
    for _ in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        twopass.note_inconclusive_certification("k")
    twopass.note_checked_slice()          # the OTHER streak's reset
    assert twopass.note_inconclusive_certification("k") is True, (
        "note_checked_slice cleared the inconclusive streak; the counters are "
        "shared")


def test_a_conclusive_certification_clears_the_streak():
    """Sporadic degenerate slices must never accumulate toward a disable."""
    twopass.reset()
    for _ in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        assert twopass.note_inconclusive_certification("k") is False
    twopass.note_conclusive_certification("k")
    assert twopass.note_inconclusive_certification("k") is False


def test_the_streak_stops_retrying_after_the_cap():
    """Unbounded retry is worse than the bug it replaced: every slice re-pays
    the full probe battery and the run never prunes."""
    twopass.reset()
    fired = [twopass.note_inconclusive_certification("k")
             for _ in range(twopass.MAX_INCONCLUSIVE_SLICES)]
    assert fired[-1] is True and not any(fired[:-1])


def test_one_configuration_cannot_mask_another():
    """The streak is per cert_key. With one global counter a healthy score
    group resets its sibling's streak, so a systematically uncertifiable
    configuration would retry forever -- the cap guarding it could never fire.
    Production configs carry several searches; the soak carries one, which is
    why this is invisible there."""
    twopass.reset()
    for _ in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        assert twopass.note_inconclusive_certification("cosine") is False
        # A sibling group certifying cleanly on the same slice must NOT clear
        # the unhealthy one's streak.
        twopass.note_conclusive_certification("dot")
    assert twopass.note_inconclusive_certification("cosine") is True


def test_the_cap_counts_per_configuration_not_in_total():
    """N groups per slice must not trip a '16 consecutive slices' cap after
    16/N slices."""
    twopass.reset()
    for _ in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        assert twopass.note_inconclusive_certification("cosine") is False
        assert twopass.note_inconclusive_certification("dot") is False
    assert twopass.note_inconclusive_certification("cosine") is True


def test_reset_clears_the_inconclusive_streak():
    twopass.reset()
    for _ in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        twopass.note_inconclusive_certification("k")
    twopass.reset()
    assert twopass.note_inconclusive_certification("k") is False


def test_a_guard_refusal_records_which_guard_fired():
    """The reason string is computed at every refusal site and used to be
    dropped, so a declined slice said nothing about WHICH guard fired. Working
    that out by hand cost three GPU round-trips and the first answer was wrong.

    Nothing reads this for control flow -- that is taken from the
    `PassOneOutcome` that `upper_bounds` returns. (An earlier version of this
    docstring credited the `slices_unfused` delta, which is exactly the
    mechanism this change removed for producing three defects in a row.)
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(7)
    d, n = 128, 256
    C = rng.standard_normal((n, d)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    Q = rng.standard_normal((n, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    Ct, Qt = torch.from_numpy(C), torch.from_numpy(Q)

    twopass.reset()
    twopass.upper_bounds(Qt, Ct, Ct.norm(dim=1).reciprocal(),
                         Qt.norm(dim=1).reciprocal(), None, metric="cosine",
                         cn=Ct.norm(dim=1), corpus_exact_fp16=False,
                         with_parts=True)
    assert twopass.stats()["last_refusal_reason"] is None, (
        "a clean slice recorded a refusal reason")

    C[0] = 0.0                      # zero norm: below NORM_MIN
    Ct = torch.from_numpy(C)
    cn = Ct.norm(dim=1)
    twopass.reset()
    twopass.upper_bounds(Qt, Ct, cn.clamp_min(1e-30).reciprocal(),
                         Qt.norm(dim=1).reciprocal(), None, metric="cosine",
                         cn=cn, corpus_exact_fp16=False, with_parts=True)
    assert twopass.stats()["last_refusal_reason"] == "corpus norm", (
        f"named the wrong guard: "
        f"{twopass.stats()['last_refusal_reason']!r}")


# --- the three-way certification decision, driven directly --------------------
#
# These exist because a FULL REVERT of the behaviour change -- replacing the
# inconclusive arm with `if False:` -- passed the entire suite. The arm is
# unreachable on CPU (the accumulator is ours there), so nothing exercised it.
# `_certify_with(outcome=...)` injects what `upper_bounds` reports, which is
# the whole point of returning the outcome instead of inferring it.

def test_no_kernel_ran_is_inconclusive_not_a_cublas_verdict():
    """THE headline fix. A guard refusal means nothing was computed, so there
    is no verdict to give about HOW it was computed. Reading it as "cuBLAS
    ran" disabled pruning for a whole 102,400-row run over 70 zero-norm rows.
    """
    from nova_bf import twopass
    why = _certify_with(outcome=twopass.PassOneOutcome(
        twopass.PASS_ONE_GUARD_REFUSED, "corpus norm"))
    assert why is False, (
        f"a slice where no kernel ran must be INCONCLUSIVE (False), got {why!r}")


def test_an_empty_corpus_is_inconclusive():
    from nova_bf import twopass
    assert _certify_with(
        outcome=twopass.PassOneOutcome(twopass.PASS_ONE_EMPTY)) is False


def test_an_allocation_failure_is_inconclusive_not_a_cublas_verdict():
    """`slices_unfused` is incremented BEFORE the unfused GEMM, so an OOM
    raised by that GEMM used to read as "cuBLAS ran" and permanently disable
    the run from one transient allocation failure."""
    from nova_bf import twopass
    assert _certify_with(
        outcome=twopass.PassOneOutcome(twopass.PASS_ONE_OOM)) is False


def test_cublas_actually_running_is_a_refusal_not_inconclusive():
    """The other side: if a kernel DID run and the accumulator was not ours,
    that is a verdict and the run must decline. Without this the inconclusive
    arm could swallow everything and never refuse."""
    from nova_bf import twopass
    why = _certify_with(outcome=twopass.PassOneOutcome(
        twopass.PASS_ONE_UNFUSED_CUBLAS))
    assert isinstance(why, str) and "cuBLAS" in why, (
        f"a real cuBLAS pass one must refuse, got {why!r}")


def test_our_own_accumulator_certifies():
    """And a clean fused/CPU outcome must still pass, or the two tests above
    would be satisfied by a function that always refuses."""
    from nova_bf import twopass
    for kind in (twopass.PASS_ONE_FUSED, twopass.PASS_ONE_CPU_WIDENED):
        assert _certify_with(outcome=twopass.PassOneOutcome(kind)) is None, kind


def test_outcome_properties_partition_the_kinds():
    """`ran_a_kernel` and `accumulator_is_ours` are what the decision reads;
    every kind must be classified by both, with no kind left undecided."""
    from nova_bf import twopass as tp
    expected = {
        tp.PASS_ONE_FUSED: (True, True),
        tp.PASS_ONE_CPU_WIDENED: (True, True),
        tp.PASS_ONE_UNFUSED_CUBLAS: (True, False),
        tp.PASS_ONE_GUARD_REFUSED: (False, False),
        tp.PASS_ONE_OOM: (False, False),
        tp.PASS_ONE_EMPTY: (False, False),
    }
    for kind, (ran, ours) in expected.items():
        o = tp.PassOneOutcome(kind)
        assert (o.ran_a_kernel, o.accumulator_is_ours) == (ran, ours), kind


# --- the PRODUCER side: does `upper_bounds` report the right kind? ------------
#
# The tests above inject an outcome and check the decision. That leaves the
# other half unpinned: mutation showed that relabelling a kind at the
# production site -- OOM as a real pass one, a guard refusal as cuBLAS, a fused
# run as cuBLAS -- passed all of them. Consumer tests cannot catch a producer
# that lies.

def _ub(C, Q=None, **kw):
    """Run the real `upper_bounds` and return just the outcome."""
    import torch
    from nova_bf import twopass

    Q = torch.randn(8, 64) if Q is None else Q
    Q = Q / Q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cn = C.norm(dim=1)
    twopass.reset()
    _, outcome = twopass.upper_bounds(
        Q, C, cn.clamp_min(1e-30).reciprocal(), Q.norm(dim=1).reciprocal(),
        None, metric="cosine", cn=cn, corpus_exact_fp16=False,
        with_parts=True, with_outcome=True, **kw)
    return outcome


def test_a_clean_cpu_slice_reports_our_own_accumulator():
    import torch
    from nova_bf import twopass
    C = torch.randn(32, 64)
    C = C / C.norm(dim=1, keepdim=True)
    o = _ub(C)
    assert o.kind == twopass.PASS_ONE_CPU_WIDENED, o
    assert o.ran_a_kernel and o.accumulator_is_ours


def test_a_degenerate_corpus_row_reports_a_guard_refusal_and_names_it():
    """The nytimes-256 shape. The kind must say no kernel ran, and the reason
    must name WHICH guard -- working that out by hand cost three GPU
    round-trips and the first answer was wrong."""
    import torch
    from nova_bf import twopass
    C = torch.randn(32, 64)
    C = C / C.norm(dim=1, keepdim=True)
    C[0] = 0.0                      # zero norm: below NORM_MIN
    o = _ub(C)
    assert o.kind == twopass.PASS_ONE_GUARD_REFUSED, o
    assert o.reason == "corpus norm", o
    assert not o.ran_a_kernel


def test_an_empty_corpus_reports_empty_not_a_refusal():
    """`-inf` over an empty corpus is a real answer, not a decline."""
    import torch
    from nova_bf import twopass
    o = _ub(torch.zeros(0, 64))
    assert o.kind == twopass.PASS_ONE_EMPTY, o
    assert not o.ran_a_kernel


def test_an_untrusted_accumulator_reports_cublas_not_a_guard_refusal():
    """A kernel DID run and it was not ours: a verdict, not an absence of one.
    Labelling it GUARD_REFUSED would make certification retry forever on a box
    that genuinely cannot be trusted."""
    import torch
    from nova_bf import twopass
    C = torch.randn(32, 64)
    C = C / C.norm(dim=1, keepdim=True)
    real = twopass.accumulator_is_ours
    twopass.accumulator_is_ours = lambda device, used_fused: False
    try:
        o = _ub(C)
    finally:
        twopass.accumulator_is_ours = real
    assert o.kind == twopass.PASS_ONE_UNFUSED_CUBLAS, o
    assert o.ran_a_kernel and not o.accumulator_is_ours


def test_an_allocation_failure_reports_oom_not_a_kernel_run():
    """`slices_unfused` is bumped BEFORE the unfused GEMM, so a counter delta
    read this as 'cuBLAS ran' and permanently disabled the run."""
    import torch
    from nova_bf import twopass
    C = torch.randn(32, 64)
    C = C / C.norm(dim=1, keepdim=True)
    real = twopass.approx_rowmax

    def boom(*a, **kw):
        twopass._STATS["slices_unfused"] += 1      # as the real one does first
        raise twopass.PassOneUnavailable("simulated allocation failure")

    twopass.approx_rowmax = boom
    try:
        o = _ub(C)
    finally:
        twopass.approx_rowmax = real
    assert o.kind == twopass.PASS_ONE_OOM, o
    assert not o.ran_a_kernel, "an OOM must not read as a kernel having run"


def test_a_fused_pass_one_reports_fused_not_cublas():
    """The fused branch is CUDA-only, so it is simulated: `approx_rowmax`
    returns `(out, fused=True)` and the outcome must say FUSED. Without this
    the fused label is untested on any CPU machine, and mislabelling it cuBLAS
    would make every fused slice refuse."""
    import torch
    from nova_bf import twopass
    C = torch.randn(32, 64)
    C = C / C.norm(dim=1, keepdim=True)
    real = twopass.approx_rowmax

    def pretend_fused(Qh, Ch, cs, out_dtype):
        out, _ = real(Qh, Ch, cs, out_dtype)
        return out, True

    twopass.approx_rowmax = pretend_fused
    try:
        o = _ub(C)
    finally:
        twopass.approx_rowmax = real
    assert o.kind == twopass.PASS_ONE_FUSED, o
    assert o.ran_a_kernel and o.accumulator_is_ours


def test_the_input_copy_allocation_failure_also_reports_oom():
    """The OTHER OOM site: `corpus_side` raising before any kernel. Both must
    report OOM -- a mutation of one was invisible while only the other was
    covered."""
    import torch
    from nova_bf import twopass
    C = torch.randn(32, 64)
    C = C / C.norm(dim=1, keepdim=True)
    real = twopass.corpus_side

    def boom(*a, **kw):
        raise torch.cuda.OutOfMemoryError("simulated")

    twopass.corpus_side = boom
    try:
        o = _ub(C)
    finally:
        twopass.corpus_side = real
    assert o.kind == twopass.PASS_ONE_OOM, o
    assert not o.ran_a_kernel


def test_the_budget_key_ignores_the_components_that_churn():
    """REGRESSION. The budget was keyed on `cert_key`, which carries the corpus
    slice height and the per-file `exact_fp16`. Under the DEFAULT
    `dense_batch_size=None`, `step = batch_size or n_rows` makes the slice
    height the file's row count, so shards of differing sizes each presented a
    fresh key with streak 0 and the cap could never fire -- precisely where the
    unbounded retry it guards against is worst.

    This reproduces that shape: the same configuration seen across files of
    different heights and storage dtypes must accumulate ONE budget.
    """
    twopass.reset()
    d, device, out_dtype = 768, "cuda:0", "torch.float32"
    score_key = ("cosine", False)

    import torch
    from nova_bf import compute as compute_mod

    def budget_key_for(slice_height, exact_fp16):
        # The REAL key builder, and a DIFFERENT Q each time: passing one shared
        # tensor made the test blind to any dependence on it, so a key that
        # varied per slice still passed.
        return compute_mod._certify_budget_key(
            score_key, torch.empty(slice_height % 97 + 1, d), device, out_dtype)

    for i in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        # Every iteration is a different file: different height, alternating
        # storage dtype -- i.e. a different `cert_key` every time.
        k = budget_key_for(4096 + i * 137, bool(i % 2))
        assert twopass.note_inconclusive_certification(k) is False
    k = budget_key_for(99991, True)
    assert twopass.note_inconclusive_certification(k) is True, (
        "the budget did not accumulate across files; it is keyed on something "
        "that churns")


def test_the_budget_key_depends_on_d_only_not_on_the_query_height():
    """`_certify_budget_key` receives Q but must use only its WIDTH. The query
    height is constant in production, but depending on the tensor at all (its
    identity, its rows) would reintroduce per-slice churn the moment anything
    reshapes it."""
    import torch
    from nova_bf import compute as compute_mod
    k = compute_mod._certify_budget_key
    a = k(("cosine", False), torch.empty(4096, 768), "cuda:0", "torch.float32")
    b = k(("cosine", False), torch.empty(9991, 768), "cuda:0", "torch.float32")
    assert a == b, "the budget key varies with something other than d"
    c = k(("cosine", False), torch.empty(4096, 384), "cuda:0", "torch.float32")
    assert a != c, "the budget key ignores d, so two dimensions would share one budget"


def test_distinct_configurations_still_get_distinct_budgets():
    """The other direction: keying on the configuration must not collapse
    genuinely different configurations into one budget."""
    twopass.reset()
    import torch
    from nova_bf import compute as compute_mod
    Q = torch.empty(0, 768)
    a = compute_mod._certify_budget_key(("cosine", False), Q, "cuda:0", "torch.float32")
    b = compute_mod._certify_budget_key(("dot", False), Q, "cuda:0", "torch.float32")
    for _ in range(twopass.MAX_INCONCLUSIVE_SLICES - 1):
        assert twopass.note_inconclusive_certification(a) is False
    # `b` is a different configuration and must start from zero.
    assert twopass.note_inconclusive_certification(b) is False
    assert twopass.note_inconclusive_certification(a) is True


# --- the CALL SITE: _twopass_prepare's inconclusive branch --------------------
#
# Mutation showed this branch had ZERO coverage: four separate reverts passed
# the whole suite -- swapping the budget key back to `cert_key`, turning the
# retry cap into `if False:`, deleting the streak reset, and turning the cap's
# `return {}` back into `continue`. The unit tests above drive the twopass
# helpers directly and cannot see which key the caller passes or whether it
# calls them at all.

class _FakeSlice:
    """The minimum `_twopass_prepare` reads off a slice."""

    def __init__(self, cb, exact_fp16=True):
        self.Cb = cb
        self.n_rows = int(cb.shape[0])
        self.exact_fp16 = exact_fp16
        self._n = cb.norm(dim=1).clamp_min(1e-12)

    def col_norms(self):
        return self._n

    def col_norms_raw(self):
        return self._n


def _drive_prepare(n_slices, vary_shape, monkeypatch, groups=("cosine",)):
    """Run `_twopass_prepare` `n_slices` times with certification INCONCLUSIVE.

    `vary_shape` reproduces the default `dense_batch_size=None` layout, where
    one slice per file means the corpus height (and `exact_fp16`) differ
    between files -- the churn that made the cap unfireable when the budget was
    keyed on `cert_key`.
    """
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    twopass.reset()
    monkeypatch.setattr(compute_mod, "_certify_two_pass",
                        lambda *a, **kw: False)
    # Engage on every slice: the live-fraction gate is not what is under test.
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")

    Q = torch.randn(64, 32)
    Q = Q / Q.norm(dim=1, keepdim=True)
    for i in range(n_slices):
        rows = (48 + i) if vary_shape else 48
        C = torch.randn(rows, 32)
        C = C / C.norm(dim=1, keepdim=True)
        sl = _FakeSlice(C, exact_fp16=(not vary_shape) or bool(i % 2))
        tp_groups = {
            (m, False): {"key": (m, False), "Q": Q, "metric": m,
                         "row_scale": None, "q_norms": Q.norm(dim=1)}
            for m in groups
        }
        compute_mod._twopass_prepare(
            tp_groups, sl, {0: float("-inf")}, {0: None}, torch.device("cpu"))
    return twopass.stats()


def test_the_cap_fires_at_the_call_site_despite_per_file_shape_churn(monkeypatch):
    """REGRESSION (verified surviving mutant). With the budget keyed on
    `cert_key` every file presents a fresh key -- because `cert_key` embeds the
    corpus height and the per-file `exact_fp16` -- so the cap never fires in the
    DEFAULT configuration. Swapping `budget_key` back to `cert_key` used to
    pass the entire suite."""
    from nova_bf import twopass
    st = _drive_prepare(twopass.MAX_INCONCLUSIVE_SLICES + 2, vary_shape=True,
                        monkeypatch=monkeypatch)
    assert st["certify_inconclusive"] >= twopass.MAX_INCONCLUSIVE_SLICES
    assert not twopass.enabled(), (
        "the retry cap never fired across files of differing shape; the budget "
        "is keyed on something that churns per file")
    assert "consecutive" in (st["unavailable"] or "")


def test_the_capping_slice_returns_no_plans(monkeypatch):
    """REGRESSION (verified surviving mutant). When the cap disables the run,
    `_twopass_prepare` must return `{}` like every other `disable()` site --
    the caller only clears `tp_groups` for FUTURE slices and still consumes
    whatever this call returns, so a surviving plan prunes a slice the manifest
    reports as unavailable."""
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    twopass.reset()
    monkeypatch.setattr(compute_mod, "_certify_two_pass", lambda *a, **kw: False)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    Q = torch.randn(64, 32)
    Q = Q / Q.norm(dim=1, keepdim=True)
    out = None
    for i in range(twopass.MAX_INCONCLUSIVE_SLICES):
        C = torch.randn(48 + i, 32)
        C = C / C.norm(dim=1, keepdim=True)
        out = compute_mod._twopass_prepare(
            {("cosine", False): {"key": ("cosine", False), "Q": Q,
                                 "metric": "cosine", "row_scale": None,
                                 "q_norms": Q.norm(dim=1)}},
            _FakeSlice(C, exact_fp16=bool(i % 2)), {0: float("-inf")},
            {0: None}, torch.device("cpu"))
    assert not twopass.enabled(), "the cap did not fire"
    assert out == {}, (
        f"the capping slice returned plans ({out!r}); they would be applied "
        f"after the run was declared disabled")


def test_the_capping_slice_stops_processing_further_groups(monkeypatch):
    """REGRESSION (verified surviving mutant). With ONE group, `continue` and
    `return {}` are indistinguishable -- the loop ends either way and `plans`
    is empty. The difference only shows with a SECOND group: `continue` lets it
    certify, build a plan and prune on a slice the run has just been disabled
    on; `return {}` stops immediately.

    So the observable is whether certification is attempted for the second
    group after the cap fires.
    """
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    calls = []

    def counting(*a, **kw):
        calls.append(1)
        return False

    twopass.reset()
    monkeypatch.setattr(compute_mod, "_certify_two_pass", counting)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    Q = torch.randn(64, 32)
    Q = Q / Q.norm(dim=1, keepdim=True)

    def groups():
        return {(m, False): {"key": (m, False), "Q": Q, "metric": m,
                             "row_scale": None, "q_norms": Q.norm(dim=1)}
                for m in ("cosine", "dot")}

    per_slice = []
    for i in range(twopass.MAX_INCONCLUSIVE_SLICES):
        C = torch.randn(48 + i, 32)
        C = C / C.norm(dim=1, keepdim=True)
        n_before = len(calls)
        compute_mod._twopass_prepare(
            groups(), _FakeSlice(C, exact_fp16=bool(i % 2)),
            {0: float("-inf")}, {0: None}, torch.device("cpu"))
        per_slice.append(len(calls) - n_before)

    assert not twopass.enabled(), "the cap did not fire"
    assert per_slice[-1] == 1, (
        f"the capping slice certified {per_slice[-1]} groups; it must return "
        f"immediately so no later group builds a plan after the disable "
        f"(per-slice certifications: {per_slice})")
    assert per_slice[0] == 2, (
        "the fixture never had two groups to distinguish, so this test cannot "
        "see the difference it exists to pin")


def test_the_cap_does_not_fire_below_the_budget(monkeypatch):
    """The other direction, so the test above is not satisfied by a cap that
    always fires."""
    from nova_bf import twopass
    _drive_prepare(twopass.MAX_INCONCLUSIVE_SLICES - 1, vary_shape=True,
                   monkeypatch=monkeypatch)
    assert twopass.enabled(), "the cap fired before the budget was spent"


def test_the_streak_is_cleared_by_a_conclusive_certification(monkeypatch):
    """REGRESSION (verified surviving mutant). Deleting
    `note_conclusive_certification` at the call site meant the streak never
    cleared, so 16 SCATTERED inconclusive slices across a long run would
    spuriously disable the two-pass."""
    import torch
    from nova_bf import compute as compute_mod
    from nova_bf import twopass

    verdicts = []

    def alternating(*a, **kw):
        verdicts.append(1)
        # Inconclusive most of the time, but a real PASS every third slice.
        return None if len(verdicts) % 3 == 0 else False

    twopass.reset()
    monkeypatch.setattr(compute_mod, "_certify_two_pass", alternating)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    Q = torch.randn(64, 32)
    Q = Q / Q.norm(dim=1, keepdim=True)
    for i in range(twopass.MAX_INCONCLUSIVE_SLICES * 3):
        # VARY the shape: a constant one gets certified on the first PASS and
        # then `is_certified` short-circuits every later slice, so nothing
        # reaches the streak at all and the test proves nothing.
        C = torch.randn(48 + i, 32)
        C = C / C.norm(dim=1, keepdim=True)
        try:
            compute_mod._twopass_prepare(
                {("cosine", False): {"key": ("cosine", False), "Q": Q,
                                     "metric": "cosine", "row_scale": None,
                                     "q_norms": Q.norm(dim=1)}},
                _FakeSlice(C, exact_fp16=bool(i % 2)), {0: float("-inf")},
                {0: None}, torch.device("cpu"))
        except Exception:
            # A PASS proceeds into plan building, which this fake slice cannot
            # support; the streak bookkeeping has already happened.
            pass
    assert twopass.enabled(), (
        "scattered inconclusive slices disabled the run; the streak is not "
        "being cleared by the conclusive ones between them")


# --------------------------------------------------------------------------
# The generation allowlist must track MEASUREMENT, not optimism.
# --------------------------------------------------------------------------

def test_no_unmeasured_generation_can_prune():
    """Every admitted capability must have a committed probe result ON DISK.

    Theorem 1 is conditional on (R5). A capability in `_R5_GENERATIONS` that no
    probe has run on is an assumption about silicon we have never touched, and
    the two-pass would prune on it in production. The T4 is why this is not
    paranoia: it measured (8,1) where the published Turing row says (4,0), so
    inheriting a generation from a paper is demonstrably not measuring it.

    The measured set is READ FROM `docs/brute-force/tc-probe/results/*.json`,
    not hardcoded. An earlier version of this test listed the capabilities by
    hand and included (8, 0) and (9, 0) because those runs were expected to
    land -- so it passed while the allowlist carried two generations with no
    evidence behind them, which is precisely the failure it was written to
    catch. A test that restates the intention cannot detect that the intention
    was not met.
    """
    import json
    import pathlib

    from nova_bf import twopass

    results = (pathlib.Path(__file__).resolve().parents[3]
               / "docs" / "brute-force" / "tc-probe" / "results")
    if not results.is_dir():
        pytest.skip(f"probe results are not present at {results}")

    measured = {}
    for f in sorted(results.glob("*.json")):
        try:
            info = json.loads(f.read_text()).get("info", {})
            sm = str(info.get("sm", ""))
        except (json.JSONDecodeError, OSError):
            continue
        if not sm.isdigit():
            continue
        # `sm` is the capability with no separator: "70" -> (7, 0),
        # "120" -> (12, 0).
        cap = (int(sm[:-1]), int(sm[-1]))
        measured.setdefault(cap, []).append(f.name)

    assert measured, f"no usable probe results found in {results}"

    unmeasured = sorted(set(twopass._R5_GENERATIONS) - set(measured))
    assert not unmeasured, (
        f"these capabilities may prune but no probe result in {results} "
        f"measures them: {unmeasured}. Either run the probe on that device and "
        f"commit its result JSON, or remove it from _R5_GENERATIONS. Refusing "
        f"is safe: pass one falls back to the one-pass fp32 path with "
        f"identical results, only slower. "
        f"Measured today: {sorted(measured)}"
    )


def test_a_generation_outside_the_allowlist_is_refused():
    """The gate must actually refuse, not merely warn.

    Blackwell (10, 0) was in the allowlist until 2026-09-18 with nothing behind
    it. If the refusal path ever stops firing, an unmeasured generation starts
    pruning silently, which is the failure this whole harness exists to prevent.
    """
    import sys
    import types
    from nova_bf import twopass

    assert (10, 0) not in twopass._R5_GENERATIONS, \
        "Blackwell is back in the allowlist; this test's premise is stale"

    fake = types.SimpleNamespace(
        device=lambda d: d,
        cuda=types.SimpleNamespace(
            get_device_capability=lambda d: (10, 0),
            get_device_name=lambda d: "NVIDIA B200",
        ),
    )
    real = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        why = twopass.probe_device_generation("cuda:0")
    finally:
        if real is not None:
            sys.modules["torch"] = real
        else:
            sys.modules.pop("torch", None)

    assert why is not None, "an unmeasured generation was admitted"
    assert "B200" in why and "no accumulation model" in why, why


# --- probe A's shortfall diagnostic ------------------------------------------
#
# `shortfall_worst_ratio` is the worst row's `exact max - first-pass max` as a
# fraction of the margin the bound's leading term gives THAT row: C(d) for
# cosine, C(d) * N_q * ceil_pow2(max N_c) for dot. Dividing a raw dot gap by
# C(d) alone measured the norms, not the margin: "fell 12.97x C(d)" at d=128
# on Gaussian vectors whose bound had plenty of slack.


def _probe_a(metric, Q, C, shift=None):
    """`_certify_two_pass` on (Q, C), recording pass one's `approx` and the
    exact tops. `shift(approx, top)` may lower `approx` (keeping `upper`, so
    the bound still holds) to plant a genuine shortfall."""
    from nova_bf import compute as compute_mod

    seen = {}
    real_ub, real_sc = twopass.upper_bounds, compute_mod._scores

    def ub(*a, **kw):
        got = real_ub(*a, **kw)
        (upper, approx, eps), outcome = got
        if shift is not None:
            approx = shift(approx)
        seen["approx"] = approx
        return (upper, approx, eps), outcome

    def sc(*a, **kw):
        out = real_sc(*a, **kw)
        seen["top"] = out.max(dim=1).values
        return out

    if metric == "cosine":
        args = (C.norm(dim=1).reciprocal(), Q.norm(dim=1).reciprocal(),
                Q.norm(dim=1))
    else:
        args = (None, None, None)
    twopass.upper_bounds, compute_mod._scores = ub, sc
    try:
        why = compute_mod._certify_two_pass(Q, C, metric, *args, torch.float32)
    finally:
        twopass.upper_bounds, compute_mod._scores = real_ub, real_sc
    return why, seen


def _c_d(C, d):
    return twopass._cf.C_const(
        d, twopass._cf.U16,
        0.0 if twopass.float32_is_exactly_fp16(C) else twopass._cf.U16)


def _gauss(n_q, n_c, d, scale, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(n_q, d, generator=g) * scale
    C = (torch.randn(n_c, d, generator=g) * scale).half().float()
    return Q, C


@pytest.mark.parametrize("scale", [1.0, 40.0])
def test_a_dot_slice_with_ample_slack_does_not_warn(caplog, scale):
    """Gaussian vectors at d=128 (norms ~11 and ~450): the raw gap over C(d)
    is far past 0.75 -- the old, misleading number -- while the gap as a
    fraction of each row's own margin is small, and nothing warns."""
    import logging

    d = 128
    Q, C = _gauss(256, 512, d, scale)
    with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
        why, seen = _probe_a("dot", Q, C)
    assert why is None, why
    ratio = twopass.stats()["shortfall_worst_ratio"]
    assert ratio is not None and 0.0 < ratio < 0.75, ratio
    raw = float((seen["top"] - seen["approx"]).max()) / _c_d(C, d)
    assert raw > 0.75, ("the fixture no longer shows the norm inflation the "
                        f"scaled ratio exists to remove: raw {raw:.3f}")
    # Exactly the documented formula: per row, gap / (N_q * ceil_pow2(max N_c)).
    cn_key = twopass._cf.ceil_pow2(float(C.norm(dim=1).max()))
    want = float(((seen["top"] - seen["approx"]).double()
                  / (Q.norm(dim=1).double() * cn_key)).max()) / _c_d(C, d)
    assert ratio == pytest.approx(want, rel=1e-12)
    assert not [r for r in caplog.records if "below the exact" in r.message]


def test_on_unit_cosine_vectors_the_ratio_is_the_raw_gap_over_c_d():
    """Cosine is in unit-score units: the scale is 1, and the ratio is the
    same number the probe always reported."""
    d = 128
    Q, C = _gauss(256, 512, d, 1.0, seed=1)
    Q = Q / Q.norm(dim=1, keepdim=True)
    C = (C / C.norm(dim=1, keepdim=True)).half().float()
    why, seen = _probe_a("cosine", Q, C)
    assert why is None, why
    want = float((seen["top"] - seen["approx"]).max()) / _c_d(C, d)
    assert twopass.stats()["shortfall_worst_ratio"] == want


@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_a_shortfall_that_uses_most_of_the_margin_still_warns(caplog, metric):
    """A first pass 0.9 of the row's margin below its exact max (upper
    untouched, so the bound still HOLDS): recorded, warned, not refused."""
    import logging

    d = 128
    Q, C = _gauss(256, 512, d, 3.0, seed=2)
    if metric == "cosine":
        Q = Q / Q.norm(dim=1, keepdim=True)
        C = (C / C.norm(dim=1, keepdim=True)).half().float()
        row_scale = torch.ones(Q.shape[0], dtype=torch.float64)
    else:
        row_scale = (Q.norm(dim=1).double()
                     * twopass._cf.ceil_pow2(float(C.norm(dim=1).max())))
    margin = (0.9 * _c_d(C, d) * row_scale).float()

    def shift(approx):
        out = approx.clone()
        out[7] = out[7] - margin[7]
        return out

    with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
        why, _ = _probe_a(metric, Q, C, shift=shift)
    assert why is None, why
    ratio = twopass.stats()["shortfall_worst_ratio"]
    assert ratio > 0.75, ratio
    assert [r for r in caplog.records if "below the exact" in r.message]


def test_the_dot_scale_reuses_the_bounds_own_query_norms(monkeypatch):
    """`query_side`'s cached binary32 norms, the entry `upper_bounds` just
    built: the diagnostic is a cache hit, not a second norm pass."""
    Q, C = _gauss(64, 128, 128, 5.0)
    built = []
    real = twopass.query_side

    def spy(Q_, row_scale=None):
        before = twopass._QCACHE.get(id(Q_))
        got = real(Q_, row_scale)
        built.append(got is not before)
        return got

    monkeypatch.setattr(twopass, "query_side", spy)
    why, _ = _probe_a("dot", Q, C)
    assert why is None, why
    assert built and built[-1] is False, built
    assert torch.equal(twopass._QCACHE[id(Q)]["qn"], Q.float().norm(dim=1))


def test_an_oom_in_the_shortfall_diagnostic_still_certifies(monkeypatch):
    """The bound held; failing to MEASURE its slack is not a verdict and must
    not raise out of certification."""
    Q, C = _gauss(64, 128, 128, 5.0)
    done = [False]
    real_ub, real_qs = twopass.upper_bounds, twopass.query_side

    def ub(*a, **kw):
        got = real_ub(*a, **kw)
        done[0] = True
        return got

    def qs(*a, **kw):
        if done[0]:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_qs(*a, **kw)

    monkeypatch.setattr(twopass, "upper_bounds", ub)
    monkeypatch.setattr(twopass, "query_side", qs)
    from nova_bf import compute as compute_mod

    why = compute_mod._certify_two_pass(Q, C, "dot", None, None, None,
                                        torch.float32)
    assert why is None, why
    assert twopass.stats()["shortfall_worst_ratio"] is None


def test_an_oom_computing_c_d_still_certifies(monkeypatch):
    """`C(d)`'s storage probe allocates too: inside the same guard."""
    Q, C = _gauss(64, 128, 128, 5.0)
    done = [False]
    real_ub, real_fp16 = twopass.upper_bounds, twopass.float32_is_exactly_fp16

    def ub(*a, **kw):
        got = real_ub(*a, **kw)
        done[0] = True
        return got

    def fp16(*a, **kw):
        if done[0]:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_fp16(*a, **kw)

    monkeypatch.setattr(twopass, "upper_bounds", ub)
    monkeypatch.setattr(twopass, "float32_is_exactly_fp16", fp16)
    from nova_bf import compute as compute_mod

    for metric in ("dot", "cosine"):
        twopass.reset()
        done[0] = False
        args = ((None, None, None) if metric == "dot" else
                (C.norm(dim=1).reciprocal(), Q.norm(dim=1).reciprocal(),
                 Q.norm(dim=1)))
        why = compute_mod._certify_two_pass(Q, C, metric, *args, torch.float32)
        assert why is None, (metric, why)
        assert twopass.stats()["shortfall_worst_ratio"] is None
