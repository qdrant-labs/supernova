"""`_gemm_rowmax` — pass one with the row max taken in the GEMM's epilogue.

The kernel replaces a cuBLAS half GEMM plus `_rowmax_scaled`, and it is the
input to the ONE decision the two-pass makes: is this query row dead? A row
max that comes out too LOW makes rows dead that should be live, which loses
ground truth silently — nothing downstream can notice, and the partials still
look perfectly well formed. So these tests attack the value, not the speed:

* against a **float64** row max of the exact product of the same float16
  operands, which is what the float32 accumulator is approximating, with the
  bound the run actually uses as the margin;
* against `_rowmax_scaled` on the unfused path, which is the code the fused
  kernel has to agree with;
* through `upper_bounds`, fused and unfused, against the float32 scores the
  one-pass path really produces — the comparison the run really makes;
* and on the gates: every assumption `fuse_available` makes is violated in
  turn and must produce a fallback, not a wrong answer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nova_bf import twopass

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _fresh():
    twopass.reset()
    yield
    twopass.reset()


def _operands(n_q, w, dim, *, seed, q_scale=1.0, c_scale=1.0, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = (torch.randn(n_q, dim, generator=g) * q_scale).to(dev)
    C = (torch.randn(w, dim, generator=g) * c_scale).to(dev)
    C = C.half().float()                      # as fineweb stores it
    cs = C.norm(dim=1).clamp_min(1e-12).reciprocal().contiguous()
    return Q, C, cs


def _ref64(Qh, Ch, cs):
    """The float64 row max of the EXACT product of the float16 operands."""
    out = torch.empty(Qh.shape[0], dtype=torch.float64, device=Qh.device)
    csd = cs.double()
    step = max(1, (1 << 24) // int(Ch.shape[0]))
    for r0 in range(0, Qh.shape[0], step):
        g = Qh[r0:r0 + step].double() @ Ch.double().T
        out[r0:r0 + step] = (g * csd[None, :]).amax(dim=1)
        del g
    return out


# --- the gates, which run anywhere ------------------------------------------


def test_cpu_tensors_are_never_handed_to_the_kernel():
    Q, C, cs = _operands(64, 32, 128, seed=1, dev="cpu")
    assert twopass.fused_rowmax(Q.half(), C.half(), cs) is None
    got, fused = twopass.approx_rowmax(Q.half(), C.half(), cs, None)
    assert not fused and got.shape == (64,)


def test_the_env_kill_switch_is_honoured(monkeypatch):
    monkeypatch.setenv("NOVA_BF_NO_FUSED_ROWMAX", "1")
    assert twopass.fused_off() == "NOVA_BF_NO_FUSED_ROWMAX"


def test_a_disable_is_sticky_until_reset():
    twopass.fuse_disable("because")
    assert twopass.fused_off() == "because"
    twopass.fuse_disable("something else")
    assert twopass.fused_off() == "because", "the FIRST reason is the one kept"
    twopass.reset()
    assert twopass.fused_off() is None


def test_usage_has_the_manifest_shape():
    u = twopass.fuse_usage()
    assert set(u) >= {"permitted", "launches", "unavailable", "config"}
    assert len(u["config"]) == 6


# --- the value ---------------------------------------------------------------


@cuda
@pytest.mark.parametrize("n_q,w,dim,seed,q_scale,c_scale", [
    (4096, 1024, 768, 0, 1.0, 1.0),
    (4096, 4096, 768, 1, 1.0, 1.0),
    (513, 777, 768, 2, 1.0, 1.0),          # every axis ragged
    (4096, 1024, 768, 3, 300.0, 0.003),    # wildly mismatched norms
    (4096, 1024, 768, 4, 0.004, 250.0),
    (2048, 2048, 384, 5, 1.0, 1.0),        # a dimension that is not 768
    (1024, 33, 768, 6, 1.0, 1.0),          # fewer columns than one tile
])
def test_the_row_max_plus_the_bound_never_falls_below_the_float64_max(
    n_q, w, dim, seed, q_scale, c_scale
):
    """The one inequality the liveness decision rests on, on the fused path.

    `bound()` is evaluated with `half_out=False`, which is what `upper_bounds`
    passes when the kernel ran — so this is the exact margin the run uses, not
    a more generous one.

    BOTH SIDES ARE ROW-SCALED, which they were not before the closed form. The
    kernel returns `max_c (q_h . c_h) / ||c||`, whose magnitude is ~||q||, and
    the old measured-residual `eps` scaled with `Qn` so it tracked that. The
    closed form's `eps` is in COSINE units — Theorem 1 bounds a quantity of
    magnitude at most `Lambda(d)` — so comparing it against an unscaled row max
    would charge a unit-norm bound to a quantity ||q|| times larger. Applying
    `row_scale` to both sides is what `upper_bounds` itself does, and it makes
    this the same inequality the run decides on.
    """
    Q, C, cs = _operands(n_q, w, dim, seed=seed, q_scale=q_scale,
                         c_scale=c_scale)
    rs = Q.norm(dim=1).reciprocal()
    qs = twopass.query_side(Q, rs)
    Ch, sigma_max, cn_min, cn_max = twopass.corpus_side(C, cs)
    got = twopass.fused_rowmax(qs["Qh"], Ch, cs)
    assert got is not None, twopass.fused_off()
    # `fmt_c=EXACT`: `_operands` stores the corpus as fineweb does, so the
    # half copy is exact and Lemma 1 charges the corpus axis nothing.
    eps = torch.from_numpy(twopass.bound(
        qs, dim, sigma_max, False, "cosine",
        fmt_c=twopass._cf.EXACT, cn_max=cn_max)).to(got.device)
    ref = _ref64(qs["Qh"], Ch, cs) * rs.double()
    assert torch.all(got.double() * rs.double() + eps.double() >= ref), (
        (ref - got.double() * rs.double() - eps.double()).max()
    )


@cuda
@pytest.mark.parametrize("seed", range(8))
def test_fuzz_against_the_float64_max(seed):
    """Random shapes and scales, so the tile boundaries land differently every
    time and nothing rests on 4096 being a multiple of anything."""
    g = torch.Generator().manual_seed(1000 + seed)
    n_q = int(torch.randint(17, 6000, (1,), generator=g))
    w = int(torch.randint(1, 5000, (1,), generator=g))
    scale = float(10.0 ** torch.randint(-3, 3, (1,), generator=g))
    Q, C, cs = _operands(n_q, w, 768, seed=seed, q_scale=scale,
                         c_scale=1.0 / scale)
    rs = Q.norm(dim=1).reciprocal()
    qs = twopass.query_side(Q, rs)
    Ch, sigma_max, cn_min, cn_max = twopass.corpus_side(C, cs)
    got = twopass.fused_rowmax(qs["Qh"], Ch, cs)
    assert got is not None, twopass.fused_off()
    eps = torch.from_numpy(twopass.bound(
        qs, 768, sigma_max, False, "cosine",
        fmt_c=twopass._cf.EXACT, cn_max=cn_max)).to(got.device)
    ref = _ref64(qs["Qh"], Ch, cs) * rs.double()
    scaled = got.double() * rs.double()
    ok = ~(torch.isnan(ref) | torch.isnan(eps))
    assert torch.all(scaled[ok] + eps.double()[ok] >= ref[ok])


@cuda
def test_it_agrees_with_the_unfused_row_max_it_replaces():
    """Not bit-identity — the accumulation order differs, and the bound covers
    that — but the two must be the same number to well within the accumulator
    term, or one of them is computing something else."""
    Q, C, cs = _operands(4096, 4096, 768, seed=11)
    qs = twopass.query_side(Q, None)
    Ch, sigma_max, _, _ = twopass.corpus_side(C, cs)
    fused = twopass.fused_rowmax(qs["Qh"], Ch, cs)
    assert fused is not None
    unfused, was_fused = twopass.approx_rowmax(
        qs["Qh"], Ch, cs, torch.float32)
    assert was_fused, "the fused path should be the default on CUDA"
    twopass.fuse_disable("for the comparison")
    unfused, was_fused = twopass.approx_rowmax(
        qs["Qh"], Ch, cs, torch.float32)
    assert not was_fused
    rel = (fused - unfused).abs() / unfused.abs().clamp_min(1e-30)
    assert float(rel.max()) < 4 * twopass.gamma(768), float(rel.max())


@cuda
def test_dot_metric_passes_ones_and_gets_the_plain_row_max():
    Q, C, _ = _operands(2048, 1024, 768, seed=12)
    qs = twopass.query_side(Q, None)
    Ch, sigma_max, _, _ = twopass.corpus_side(C, None)
    got, fused = twopass.approx_rowmax(qs["Qh"], Ch, None, None)
    assert fused
    ref = (qs["Qh"].double() @ Ch.double().T).amax(dim=1)
    assert torch.all((got.double() - ref).abs()
                     <= 4 * twopass.gamma(768) * ref.abs().clamp_min(1e-9))


@cuda
def test_a_column_outside_the_slice_can_never_win_the_maximum():
    """The tail tile wraps rather than masking its loads, so a wrapped column
    computes a REAL value from a real corpus row. If the epilogue's mask were
    dropped, an all-negative slice would take its maximum from a duplicate.
    The width here is deliberately not a multiple of any tile size."""
    # EVERY score must be negative, or this test is vacuous. A masked-off
    # lane contributes exactly 0.0 (the epilogue loads `CS` with `other=0.0`),
    # so an unmasked wrapped lane can only win when the row's true maximum is
    # below zero. The previous fixture set `C[0] = -Q[0] * 1000` on otherwise
    # random data, where the minimum true row max measured +2.83 — a
    # manufactured 0.0 could never win it, and the test passed with the
    # epilogue mask deleted.
    #
    # All-positive queries against an all-negative corpus makes every dot
    # negative by construction. `w = 4097` is kept: not a multiple of any
    # tile size, so the tail tile really wraps onto column 0.
    g = torch.Generator(device="cpu").manual_seed(13)
    Q = (torch.rand(1024, 768, generator=g) + 0.5).to("cuda")
    C = (-(torch.rand(4097, 768, generator=g) + 0.5)).to("cuda").half().float()
    cs = C.norm(dim=1).clamp_min(1e-12).reciprocal().contiguous()
    qs = twopass.query_side(Q, None)
    Ch, sigma_max, _, _ = twopass.corpus_side(C, cs)
    ref = ((qs["Qh"].double() @ Ch.double().T) * cs.double()).amax(dim=1)
    assert bool((ref < 0).all()), (
        f"fixture is vacuous: {int((ref >= 0).sum())} of {len(ref)} rows have "
        f"a non-negative true row max (min {float(ref.min()):.4f}), so a "
        f"wrapped lane's 0.0 could never win and the mask is untested")
    got = twopass.fused_rowmax(qs["Qh"], Ch, cs)
    assert torch.all((got.double() - ref).abs()
                     <= 1e-3 * ref.abs().clamp_min(1e-6))


@cuda
@pytest.mark.parametrize("break_it", [
    "q_float32", "c_float32", "cs_float64", "q_strided", "c_strided",
    "cs_strided", "width_mismatch",
])
def test_every_assumption_is_checked_not_trusted(break_it):
    Q, C, cs = _operands(1024, 512, 768, seed=14)
    Qh, Ch = Q.half().contiguous(), C.half().contiguous()
    if break_it == "q_float32":
        Qh = Q
    elif break_it == "c_float32":
        Ch = C
    elif break_it == "cs_float64":
        cs = cs.double()
    elif break_it == "q_strided":
        Qh = Q.half()[:, ::2]
    elif break_it == "c_strided":
        Ch = C.half()[:, ::2]
    elif break_it == "cs_strided":
        cs = torch.zeros(2 * cs.numel(), device=cs.device)[::2]
    else:
        Ch = C.half()[:, :384].contiguous()
    assert twopass.fuse_available(Qh, Ch, cs) is False
    assert twopass.fused_rowmax(Qh, Ch, cs) is None


@cuda
def test_upper_bounds_still_bounds_the_float32_scores_both_ways():
    """End to end through the entry point the run uses, fused and unfused.

    Fused must also be TIGHTER, because it drops the float16 output term —
    that is the whole reason the bound changed.
    """
    from nova_bf.compute import _scores

    Q, C, cs = _operands(4096, 2048, 768, seed=15)
    qn = Q.norm(dim=1).clamp_min(1e-12)
    rs = qn.reciprocal()
    ref = _scores(Q, C, "cosine", qn, scale_in_packer=False).amax(dim=1)

    fused_ub = twopass.upper_bounds(Q, C, cs, rs, None, metric="cosine")
    assert twopass.stats()["slices_fused"] == 1
    twopass.reset()
    twopass.fuse_disable("for the comparison")
    plain_ub = twopass.upper_bounds(Q, C, cs, rs, None, metric="cosine")
    assert twopass.stats()["slices_unfused"] == 1

    assert torch.all(fused_ub >= ref), float((ref - fused_ub).max())
    assert torch.all(plain_ub >= ref), float((ref - plain_ub).max())
    assert torch.all(fused_ub <= plain_ub + 1e-9), "the fused bound is tighter"
    assert float((plain_ub - fused_ub).min()) > 0.0


@cuda
def test_a_launch_failure_falls_back_instead_of_raising(monkeypatch):
    """A configuration that cannot fit must take the run to the cuBLAS path,
    with the looser bound, rather than killing it."""
    monkeypatch.setenv("NOVA_BF_FUSE_CONFIG", "256,256,128,8,4,8")
    Q, C, cs = _operands(2048, 1024, 768, seed=16)
    qs = twopass.query_side(Q, None)
    Ch, sigma_max, _, _ = twopass.corpus_side(C, cs)
    got, fused = twopass.approx_rowmax(qs["Qh"], Ch, cs, None)
    assert not fused
    assert twopass.fused_off() is not None
    assert got.shape == (2048,)


@cuda
def test_the_kernel_does_not_spill():
    """A spilling tile shape would still be correct and would still be
    reported as the fast path, while running slower than what it replaced."""
    Q, C, cs = _operands(4096, 4096, 768, seed=17)
    assert twopass.fused_rowmax(Q.half().contiguous(),
                                C.half().contiguous(), cs) is not None
    info = twopass.fuse_usage()
    assert info["launches"] == 1
    assert info.get("n_spills") == 0, info
