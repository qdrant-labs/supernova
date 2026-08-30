"""Comparing two engines' top-K without pretending float reductions agree.

The naive way to compare two top-K lists — assert the id sequences are equal —
fails for a reason that has nothing to do with correctness: three
implementations (nova-bf's tiled matmuls, Qdrant's own reductions, this
harness's Python loops) sum the same products in three different orders, and
float addition is not associative. Two documents whose true scores differ by
less than the accumulated error can legitimately swap, and one sitting exactly
at the K-th position can legitimately fall out of one engine's list and into
another's.

So the comparison here asserts the two properties that actually distinguish a
correct implementation from a broken one:

  1. **Agreement where both engines answered.** For every id in both lists,
     the scores match to tolerance. A wrong metric, a wrong norm, a dropped
     token, a mis-mapped row — all of them move a score far more than a
     reduction-order ULP.
  2. **Disagreement only at the boundary.** An id in one list and not the
     other is accepted ONLY if its score sits within tolerance of the OTHER
     list's K-th score. That is the signature of a near-tie straddling the
     cutoff. An id missing from the middle of the ranking — a filter that
     wrongly excluded a row, a non-candidate wrongly included — is not at the
     boundary and fails.

`assert_same_membership` is the stricter variant for cases where the two sides
must agree exactly on WHICH rows were eligible (filter semantics), independent
of how they ranked them.
"""

from __future__ import annotations

# Per-metric absolute tolerance, scaled by |score| at the comparison site.
# Euclidean is loosest because nova-bf derives it from a shared Gram expansion
# (‖q‖² − 2q·c + ‖c‖²), which cancels catastrophically near zero distance.
TOL = {"dot": 2e-4, "cosine": 2e-5, "euclidean": 2e-3}


def scores_are_reproducible(device: str, vector_type: str,
                            multivector_kernel: str = "torch") -> bool:
    """Can two runs over the same input be expected to produce identical score
    BITS — the precondition for any bit-equality assertion?

    Everywhere except one case, yes. The exception, measured on an A10G:
    **multivector on CUDA with the default `torch` kernel is not
    bit-reproducible run to run.** Its MaxSim sum over query tokens is an
    `index_add_` (`compute.py`), which on CUDA is an atomicAdd, so the order
    float additions land in varies between launches. Three identical runs
    disagreed on ~50 of 200 scores, all ~1e-7 relative — and the ranking was
    identical every time.

    This is a property of the hardware and the reduction, not a defect the
    tests can hold nova-bf to. Measured on the same box, all three runs x3:

      multivector_kernel="torch"                    NOT reproducible (~50/200)
      multivector_kernel="triton_reduce"            bit-reproducible
      "torch" + torch.use_deterministic_algorithms  bit-reproducible

    `CUBLAS_WORKSPACE_CONFIG` does NOT help (tried `:4096:8` and `:16:8`),
    which is the tell that this is the `index_add_` atomics rather than a
    cuBLAS split-k reduction. So a run needing byte-identical multivector
    ground truth has two levers, and `triton_reduce` is the one that costs
    nothing (it is also the faster path).

    Callers use this to pick their assertion: `assert_identical` where bits are
    guaranteed, `assert_same_membership` + `assert_scores_agree` where only the
    ranking is. Asserting bit-equality where it cannot hold does not make the
    code more correct, it just makes the suite fail on the GPU for a reason
    that has nothing to do with what the test is about.
    """
    if vector_type != "multivector" or device != "cuda":
        return True
    return multivector_kernel == "triton_reduce"


def assert_same_ranking(a, b, *, metric: str, label: str,
                        device: str = "cpu", vector_type: str = "dense",
                        multivector_kernel: str = "torch"):
    """`assert_identical` where score bits are reproducible, and the strongest
    available claim where they are not: same ids, in the same order, with
    scores agreeing to the metric's tolerance.

    Note what is NOT weakened even in the non-reproducible case — the id
    sequence is still compared exactly. The measured nondeterminism moves the
    last mantissa bit and has never reordered the result, so a reordering
    remains a failure."""
    if scores_are_reproducible(device, vector_type, multivector_kernel):
        assert_identical(a, b, label=label)
        return
    assert_same_membership(a, b, label=label)
    assert [int(i) for i, _ in a] == [int(i) for i, _ in b], (
        f"{label}: ranking differs (scores here are not bit-reproducible, but "
        f"the ORDER still must not move)")
    assert_scores_agree(a, b, metric=metric, label=label)


def as_dict(hits) -> dict[int, float]:
    return {int(i): float(s) for i, s in hits}


def assert_scores_agree(a, b, *, metric: str, label: str, tol: float | None = None):
    """The full boundary-aware comparison — see this module's docstring.

    `a` and `b` are `[(id, score), …]` lists (or anything `as_dict` accepts)
    from the two engines being compared; neither is privileged.
    """
    tol = TOL[metric] if tol is None else tol
    da, db = as_dict(a), as_dict(b)
    assert da or db, f"{label}: both sides returned nothing"
    assert da, f"{label}: left side returned nothing while right returned {len(db)}"
    assert db, f"{label}: right side returned nothing while left returned {len(da)}"

    for i in sorted(set(da) & set(db)):
        assert abs(da[i] - db[i]) <= tol * (1 + abs(db[i])), (
            f"{label}: id={i} scored {da[i]!r} vs {db[i]!r} "
            f"(tolerance {tol} relative)")

    a_floor, b_floor = min(da.values()), min(db.values())
    for i in sorted(set(da) - set(db)):
        assert abs(da[i] - b_floor) <= tol * (1 + abs(b_floor)), (
            f"{label}: id={i} (score {da[i]!r}) is in the left top-K only, and is "
            f"not at the right side's top-K boundary {b_floor!r} — this is a "
            f"ranking disagreement, not a boundary tie")
    for i in sorted(set(db) - set(da)):
        assert abs(db[i] - a_floor) <= tol * (1 + abs(a_floor)), (
            f"{label}: id={i} (score {db[i]!r}) is in the right top-K only, and is "
            f"not at the left side's top-K boundary {a_floor!r} — this is a "
            f"ranking disagreement, not a boundary tie")


def assert_same_membership(a, b, *, label: str):
    """Exact set equality of returned ids — no boundary allowance.

    Only for comparisons where a difference could not possibly be a float
    artefact: the same run on two devices, or a filter's ELIGIBLE set checked
    against the naive predicate over the whole corpus.
    """
    ia, ib = {int(i) for i, _ in a}, {int(i) for i, _ in b}
    assert ia == ib, (
        f"{label}: id sets differ — only left: {sorted(ia - ib)}, "
        f"only right: {sorted(ib - ia)}")


def assert_identical(a, b, *, label: str, tol: float = 0.0):
    """Same ids, same order, same scores. Used where nothing legitimate may
    differ — e.g. a `rows` subset against the same search without one."""
    la, lb = list(a), list(b)
    assert [int(i) for i, _ in la] == [int(i) for i, _ in lb], (
        f"{label}: ranking differs\n  left:  {[int(i) for i, _ in la][:12]}\n"
        f"  right: {[int(i) for i, _ in lb][:12]}")
    for (ia, sa), (_, sb) in zip(la, lb):
        assert abs(sa - sb) <= tol * (1 + abs(sb)), (
            f"{label}: id={ia} scored {sa!r} vs {sb!r}")
