"""`_gemm_rowmax` executed for real, on CPU, in the Triton interpreter.

`tests/test_fused_rowmax.py` skips 28 of its 32 tests on a machine without a
GPU, and `slices_fused` is 0 in every CI run — so the kernel that decides
liveness on the production path is exercised by nothing that actually runs.
It is also the path whose bound is TIGHTER (no `2**-11 * T` output term,
because the row max is read off the float32 accumulator), so an indexing or
masking mistake in it is not merely slow: it silently makes live rows dead.

`TRITON_INTERPRET=1` runs the real `@triton.jit` body as Python/numpy, which
covers exactly the parts a review cannot check by eye — the pid->tile
swizzle, the `% M` / `% N` wrap, the k-loop masking, and the epilogue's
column mask. It does NOT cover tensor-core numerics; `np.matmul` into a
float32 accumulator stands in for `mma.sync`, so the numeric assertions here
are "the right elements were multiplied and reduced", not "the hardware
rounds this way".

TWO ENVIRONMENT FACTS drive the odd shape of this file.

1. `TRITON_INTERPRET` is read when `triton` is imported, and by the time one
   test module runs, another has almost certainly imported it. So the real
   tests run in a SUBPROCESS; `test_the_interpreted_suite_passes` is the only
   test the parent executes, and it re-invokes pytest on this same file.

2. `nova_bf.twopass` has `from __future__ import annotations`, so the
   kernel's `BLOCK_M: _tl.constexpr` annotation reaches Triton as the STRING
   `"_tl.constexpr"`. The compiler is happy (`KernelParam.is_constexpr` is
   `"constexpr" in annotation`), but the interpreter is not:
   `GridExecutor.__init__` keeps only params whose normalised annotation is
   EXACTLY `"constexpr"`, and `_normalize_ty` strips a leading `"tl."` and
   nothing else. Rather than change production code, `_teach_the_interpreter`
   widens `_normalize_ty` to accept any `*.constexpr` — which is precisely
   what the COMPILER already does, so the shim makes the interpreter agree
   with the GPU instead of inventing behaviour.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

_CHILD = os.environ.get("NOVA_BF_INTERPRET_CHILD") == "1"

# The child currently runs 41 tests. A floor rather than an equality so
# adding a case does not fail the parent, but dropping most of them does.
_MIN_CHILD_TESTS = 30

def test_the_interpreted_suite_passes():
    """Re-run this file with `TRITON_INTERPRET=1` set before `import triton`."""
    if _CHILD:
        pytest.skip("this process IS the interpreted child")
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["NOVA_BF_INTERPRET_CHILD"] = "1"
    # A kill switch left set in the parent's environment would turn the whole
    # child suite into a no-op that still reports success.
    env.pop("NOVA_BF_NO_FUSED_ROWMAX", None)
    env.pop("NOVA_BF_FUSE_CONFIG", None)
    # The child gets the parent's import path, so this works whether the file
    # sits in `tests/` (where pytest's own `pythonpath = ["src"]` applies) or
    # is pointed at from elsewhere.
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(
            os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "-p", "no:cacheprovider", os.path.abspath(__file__)],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, (
        f"the interpreted kernel suite failed:\n{proc.stdout}\n{proc.stderr}")
    # A child that collected nothing — or that skipped nearly everything —
    # is indistinguishable from a child that passed, and this whole file
    # exists to close a coverage hole. So pin the COUNT, not the word:
    # "1 passed, 40 skipped" satisfies `" passed" in stdout` perfectly.
    m = re.search(r"(\d+) passed", proc.stdout)
    assert m, f"the child reported no passing tests:\n{proc.stdout}"
    n_passed = int(m.group(1))
    assert n_passed >= _MIN_CHILD_TESTS, (
        f"the interpreted child ran only {n_passed} tests, expected at least "
        f"{_MIN_CHILD_TESTS} — the kernel coverage this file exists for is "
        f"not actually running:\n{proc.stdout}")


# --- everything below runs only in the child ---------------------------------

if _CHILD:
    assert os.environ.get("TRITON_INTERPRET") == "1"

    def _teach_the_interpreter() -> None:
        """Make the interpreter recognise `_tl.constexpr` as a constexpr.

        Test-side only. The compiler's own test is `"constexpr" in
        annotation`, which `"_tl.constexpr"` passes, so this aligns the
        interpreter with the GPU rather than changing the kernel's meaning.
        `GridExecutor.__init__` does `from .jit import _normalize_ty` at call
        time, so patching the module attribute is enough.
        """
        import triton.runtime.jit as jit

        original = jit._normalize_ty

        def normalize(ty):
            out = original(ty)
            if isinstance(out, str) and out.rsplit(".", 1)[-1] == "constexpr":
                return "constexpr"
            return out

        jit._normalize_ty = normalize

    _teach_the_interpreter()

    import torch                                     # noqa: E402
    import triton                                    # noqa: E402

    from nova_bf import twopass                      # noqa: E402

    @pytest.fixture(autouse=True)
    def _fresh():
        twopass.reset()
        yield
        twopass.reset()

    def _operands(M, N, K, *, seed, q_scale=1.0, c_scale=1.0):
        g = torch.Generator().manual_seed(seed)
        Qh = (torch.randn(M, K, generator=g) * q_scale).half().contiguous()
        Ch = (torch.randn(N, K, generator=g) * c_scale).half().contiguous()
        cs = (torch.rand(N, generator=g) + 0.5).float().contiguous()
        return Qh, Ch, cs

    def _launch(Qh, Ch, cs, bm, bn, bk, gm, *, fill=float("inf")):
        """`fused_rowmax`'s body, verbatim, minus the CUDA-only scaffolding.

        Kept a line-for-line mirror of `twopass.fused_rowmax` on purpose:
        `EVEN_K` is derived the same way, `part` is filled the same way, the
        grid is computed the same way, and the two-tile-count reduction at the
        end is the same expression. Anything this helper gets to decide is
        something the test would stop covering.
        """
        M, K = int(Qh.shape[0]), int(Qh.shape[1])
        N = int(Ch.shape[0])
        n_tiles = triton.cdiv(N, bn)
        part = torch.full((n_tiles, M), fill, dtype=torch.float32)
        grid = (triton.cdiv(M, bm) * n_tiles,)
        twopass._gemm_rowmax[grid](
            Qh, Ch, cs, part, M, N, K,
            Qh.stride(0), Ch.stride(0), part.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
            EVEN_K=(K % bk == 0), num_stages=1, num_warps=4,
        )
        return (part[0] if n_tiles == 1 else part.amax(dim=0)), part

    def _ref64(Qh, Ch, cs):
        return ((Qh.double() @ Ch.double().T) * cs.double()).amax(dim=1)

    def _tol(Qh, Ch, cs, K):
        """`4 * gamma_K` times the row's sum of |products|.

        The accumulator is float32 over K terms in an order the kernel is free
        to choose, so `gamma_K = K*u/(1-K*u)` times the sum of absolute
        products is the honest per-row envelope; the factor of four is slack
        for the epilogue's own multiply and for numpy's blocking.
        """
        weight = (Ch.double().abs() * cs.double().abs()[:, None])
        row = (Qh.double().abs() @ weight.T).amax(dim=1)
        return 4.0 * twopass.gamma(K) * row + 1e-30

    # --- the kernel is really running ------------------------------------

    def test_the_production_kernel_is_the_one_being_interpreted():
        """Guards the whole file: if the shim ever stops working, or the
        module falls back to `_gemm_rowmax = None`, every other test here
        would quietly test nothing."""
        from triton.runtime.interpreter import InterpretedFunction

        assert twopass._gemm_rowmax is not None, twopass._UNAVAILABLE
        assert isinstance(twopass._gemm_rowmax, InterpretedFunction)
        assert (twopass._gemm_rowmax.fn.__module__
                == "nova_bf.twopass")
        Qh, Ch, cs = _operands(20, 24, 32, seed=0)
        got, part = _launch(Qh, Ch, cs, 16, 16, 16, 8)
        assert torch.isfinite(part).all(), "a tile was never written"
        assert torch.allclose(got.double(), _ref64(Qh, Ch, cs), atol=1e-3)

    # --- the row max, against float64 ------------------------------------

    @pytest.mark.parametrize(
        "name,M,N,K,bm,bn,bk,gm,even_k", [
            # K a multiple of BLOCK_K: the unmasked-load branch.
            ("even K", 40, 48, 32, 16, 16, 16, 8, True),
            # K not a multiple: the masked branch, tail block partly live.
            ("ragged K", 40, 48, 40, 16, 16, 16, 8, False),
            ("K below one block", 40, 48, 5, 16, 16, 16, 8, False),
            ("K = 1", 40, 48, 1, 16, 16, 16, 8, False),
            # M and N both smaller than a single tile: every lane of both
            # axes wraps, and the store mask is doing all the work.
            ("M and N below one block", 5, 7, 16, 16, 16, 16, 8, True),
            ("M not a multiple of BLOCK_M", 33, 32, 32, 16, 16, 16, 8, True),
            # The tail column tile wraps onto column 0.
            ("ragged N, tail tile wraps", 40, 17, 32, 16, 16, 16, 8, True),
            ("many N tiles", 24, 97, 32, 16, 16, 16, 8, True),
            # GROUP_M above the M-tile count: `group_m` clamps to 1 or 2 and
            # the whole grid lands in group 0.
            ("GROUP_M above the M-tile count", 17, 48, 32, 16, 16, 16,
             64, True),
            ("GROUP_M above it, one M tile", 9, 48, 32, 16, 16, 16, 8, True),
            # A K that needs several blocks AND is ragged, with a wide N.
            ("ragged K over several blocks", 33, 65, 100, 16, 16, 32, 4,
             False),
        ])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_the_row_max_matches_a_float64_reference(
            name, M, N, K, bm, bn, bk, gm, even_k, seed):
        assert (K % bk == 0) is even_k, (
            f"{name}: this case no longer exercises EVEN_K={even_k}")
        Qh, Ch, cs = _operands(M, N, K, seed=seed)
        got, part = _launch(Qh, Ch, cs, bm, bn, bk, gm)
        assert torch.isfinite(part).all(), (
            f"{name}: {int((~torch.isfinite(part)).sum())} lanes of part were "
            f"never written, so some (row, column tile) was never covered")
        ref = _ref64(Qh, Ch, cs)
        err = (got.double() - ref).abs()
        assert torch.all(err <= _tol(Qh, Ch, cs, K)), (
            f"{name}: worst |fused - float64| = {float(err.max()):.3e}")

    # --- the epilogue's column mask, on a slice where it can be seen ------

    def test_a_wrapped_column_cannot_win_an_all_negative_slice():
        """The one fixture in which the epilogue mask is observable.

        A masked-off lane contributes exactly 0.0 — `CS` is loaded with
        `other=0.0` and the `tl.where` replaces the product with `-inf` — so
        the mask can only ever CHANGE an answer when the row's true maximum is
        below zero. Any fixture whose minimum true row max is positive passes
        with the mask deleted, which is how the GPU version of this test came
        to be vacuous.

        All-positive queries against an all-negative corpus makes every dot
        negative by construction, and `N = 17` with `BLOCK_N = 16` leaves a
        one-column tail tile whose other 15 lanes wrap onto real corpus rows.
        """
        g = torch.Generator().manual_seed(13)
        Qh = (torch.rand(40, 32, generator=g) + 0.5).half().contiguous()
        Ch = (-(torch.rand(17, 32, generator=g) + 0.5)).half().contiguous()
        cs = (torch.rand(17, generator=g) + 0.5).float().contiguous()
        ref = _ref64(Qh, Ch, cs)
        assert bool((ref < -1e-3).all()), (
            f"fixture is vacuous: the LARGEST true row max is "
            f"{float(ref.max()):.4f}, "
            f"so an unmasked lane's 0.0 could never win the maximum")
        got, _ = _launch(Qh, Ch, cs, 16, 16, 16, 8)
        assert bool((got < 0).all()), (
            "a lane outside the slice manufactured a non-negative maximum")
        assert torch.all((got.double() - ref).abs() <= _tol(Qh, Ch, cs, 32))

    def test_a_wrapped_column_cannot_win_when_only_the_tail_lane_is_real():
        """The same mask, with the ONE live lane of the tail tile deliberately
        made the worst column in the slice.

        `N = 17`, `BLOCK_N = 16`: tile 1 holds column 16 plus fifteen lanes
        that wrap onto columns 0..14. Column 16 is scaled down so that it is
        never the row maximum, and every score is negative, so tile 1's
        contribution must be exactly column 16's — if a wrapped lane leaked
        through, tile 1 would report column 0..14's much LARGER value and the
        final `amax` would be wrong in the LIVE (safe) direction, which is
        still a wrong row max and would show up here."""
        g = torch.Generator().manual_seed(29)
        Qh = (torch.rand(24, 32, generator=g) + 0.5).half().contiguous()
        Ch = (-(torch.rand(17, 32, generator=g) + 0.5)).half()
        Ch[16] *= 8.0                          # column 16 is much the worst
        Ch = Ch.contiguous()
        cs = torch.ones(17, dtype=torch.float32)
        _, part = _launch(Qh, Ch, cs, 16, 16, 16, 8)
        exact = (Qh.double() @ Ch.double().T)
        assert part.shape[0] == 2
        assert torch.all((part[1].double() - exact[:, 16]).abs() <= 1e-2), (
            "the tail tile's row max is not column 16 alone")
        assert bool((part[1] < part[0]).all()), "fixture no longer separates"

    # --- the row wrap ----------------------------------------------------

    def test_a_wrapped_row_cannot_overwrite_a_real_row():
        """`M = 33` with `BLOCK_M = 16` gives a final M-tile whose lanes 1..15
        wrap onto rows 0..14. Row 0's scores are made enormous, so if the
        store used the WRAPPED index (or the mask were missing) row 0's value
        would land on rows 33.. or on top of the tail rows."""
        g = torch.Generator().manual_seed(31)
        Qh = (torch.randn(33, 32, generator=g) * 0.01).half()
        Qh[0] = 20.0
        Qh = Qh.contiguous()
        Ch = (torch.rand(24, 32, generator=g) + 0.5).half().contiguous()
        cs = torch.ones(24, dtype=torch.float32)
        got, _ = _launch(Qh, Ch, cs, 16, 16, 16, 8)
        ref = _ref64(Qh, Ch, cs)
        assert float(got[0]) > 100.0, "fixture no longer separates row 0"
        assert torch.all((got.double() - ref).abs() <= _tol(Qh, Ch, cs, 32))
        assert torch.all(got[1:] < 10.0), (
            "row 0's maximum leaked onto a wrapped row")

    # --- an uncovered tile must fail LIVE, not DEAD ----------------------

    @pytest.mark.parametrize("fill,expect", [
        (float("inf"), "live"),
        (-3.0e38, "dead"),
    ])
    def test_an_uncovered_tile_fails_towards_live(fill, expect):
        """`GROUP_M = 0` degenerates the pid->tile mapping and leaves lanes of
        `part` unwritten, raising nothing. This is the one mis-computed input
        in the module — `_config_ok` rejects it, but the fill is what decides
        what an uncovered tile COSTS.

        With `fused_rowmax`'s `+inf` the row max comes back `+inf`, the caller
        reads `~(inf < thr)` as LIVE, and the slice pays for one exact GEMM.
        With a poisoned buffer (what `torch.empty` can hand back) the row max
        comes back hugely negative and the row is called DEAD — ground truth
        lost with no symptom. Both are asserted, so the test states the
        difference rather than only the good half.
        """
        Qh, Ch, cs = _operands(64, 64, 32, seed=5)
        got, part = _launch(Qh, Ch, cs, 16, 16, 16, 0, fill=fill)
        unwritten = (part == fill)
        assert bool(unwritten.any()), (
            "GROUP_M=0 no longer leaves any lane uncovered; this test's "
            "premise is gone")
        if expect == "live":
            assert bool(torch.isinf(got).any()) and bool((got > 0).any())
            assert not bool((got < -1e30).any())
        else:
            assert bool((got < -1e30).any()), (
                "a poisoned buffer no longer produces a dead-looking row max")

    def test_config_ok_rejects_the_group_m_that_loses_coverage():
        assert twopass._config_ok((16, 16, 16, 0, 3, 4)) is False
        assert twopass._config_ok((16, 16, 16, -1, 3, 4)) is False
        assert twopass._config_ok((16, 16, 16, 1, 3, 4)) is True

    # --- fused_rowmax's own post-processing -------------------------------

    @pytest.fixture
    def _pretend_cuda(monkeypatch):
        """Let `fused_rowmax` run on CPU tensors.

        Only the two gates that are literally about the device are lifted —
        `fuse_available`'s `is_cuda` test and the `torch.cuda.device` context
        — so the reduction at the end of `fused_rowmax`, the `+inf` fill, the
        int32 store-offset guard and the exception handling are all the
        production ones.
        """
        import contextlib

        real = twopass.fuse_available

        def available(Qh, Ch, cs):
            if twopass.fused_off() is not None:
                return False
            return bool(real(Qh, Ch, cs)) or _cpu_ok(Qh, Ch, cs)

        def _cpu_ok(Qh, Ch, cs):
            return (Qh.dtype is torch.float16 and Ch.dtype is torch.float16
                    and cs.dtype is torch.float32
                    and Qh.ndim == 2 and Ch.ndim == 2 and cs.ndim == 1
                    and Qh.is_contiguous() and Ch.is_contiguous()
                    and cs.is_contiguous()
                    and Qh.shape[1] == Ch.shape[1]
                    and cs.shape[0] == Ch.shape[0]
                    and Qh.shape[0] > 0 and Ch.shape[0] > 0
                    and Qh.shape[1] > 0
                    and Qh.shape[0] * Qh.shape[1] < 2 ** 31
                    and Ch.shape[0] * Qh.shape[1] < 2 ** 31)

        monkeypatch.setattr(twopass, "fuse_available", available)
        monkeypatch.setattr(torch.cuda, "device",
                            lambda *_a, **_k: contextlib.nullcontext())
        return None

    @pytest.mark.parametrize("N,bn,n_tiles", [
        (16, 16, 1),          # exactly one tile: `part[0]`, no amax
        (9, 16, 1),           # one tile, wrapping
        (17, 16, 2),          # two tiles
        (97, 16, 7),          # many tiles
    ])
    def test_fused_rowmax_reduces_every_tile_count(
            _pretend_cuda, monkeypatch, N, bn, n_tiles):
        monkeypatch.setenv("NOVA_BF_FUSE_CONFIG", f"16,{bn},16,8,1,4")
        Qh, Ch, cs = _operands(40, N, 32, seed=7)
        got = twopass.fused_rowmax(Qh, Ch, cs)
        assert got is not None, twopass.fused_off()
        assert got.shape == (40,)
        assert got.dtype is torch.float32
        ref = _ref64(Qh, Ch, cs)
        assert torch.all((got.double() - ref).abs() <= _tol(Qh, Ch, cs, 32))
        assert twopass.fuse_usage()["launches"] == 1
        assert twopass.fuse_usage()["config"] == [16, bn, 16, 8, 1, 4]

    @pytest.mark.parametrize("K,bk,even_k", [
        (32, 16, True),          # EVEN_K: the unmasked k-loop
        (40, 16, False),         # ragged tail block
        (5, 16, False),          # K below one block
        (1, 16, False),          # a single term
    ])
    def test_fused_rowmax_derives_even_k_from_the_real_shape(
            _pretend_cuda, monkeypatch, K, bk, even_k):
        """`EVEN_K = (K % bk == 0)` is computed by `fused_rowmax`, not by the
        kernel, and it is the promise that licenses the UNMASKED loads in the
        k-loop. Getting it wrong reads past the end of a row (or of the whole
        tensor) and folds a neighbouring row's coordinates into the dot, so
        the derivation has to be exercised through the production call and not
        re-derived by the test.
        """
        assert (K % bk == 0) is even_k
        monkeypatch.setenv("NOVA_BF_FUSE_CONFIG", f"16,16,{bk},8,1,4")
        Qh, Ch, cs = _operands(40, 24, K, seed=21)
        got = twopass.fused_rowmax(Qh, Ch, cs)
        assert got is not None, twopass.fused_off()
        ref = _ref64(Qh, Ch, cs)
        err = (got.double() - ref).abs()
        assert torch.all(err <= _tol(Qh, Ch, cs, K)), (
            f"K={K}, BLOCK_K={bk}: worst error {float(err.max()):.3e}")

    @pytest.mark.parametrize("N,bn", [(16, 16), (97, 16)])
    def test_fused_rowmax_fills_with_inf_so_an_unwritten_lane_is_live(
            _pretend_cuda, monkeypatch, N, bn):
        """The whole point of the `+inf` fill, through the real function and
        for both branches of its reduction.

        The kernel is replaced by a launcher that writes NOTHING, which is the
        worst case of the coverage bug `GROUP_M=0` causes. `fused_rowmax` must
        then return `+inf` for every row — the caller's `~(inf < thr)` reads
        that as LIVE and the slice pays for one exact GEMM. `torch.empty`
        here would return whatever was in the allocator's cache, and a
        sufficiently negative value silently makes a live row DEAD.

        `N = 16` takes the `n_tiles == 1` branch (`part[0]`, returned
        verbatim) and `N = 97` takes `part.amax(dim=0)` over seven rows that
        are entirely `+inf`; both must come back `+inf`, so the assertion
        covers the post-processing for both tile counts.
        """
        class _Noop:
            def __getitem__(self, grid):
                return lambda *a, **k: None

        monkeypatch.setenv("NOVA_BF_FUSE_CONFIG", f"16,{bn},16,8,1,4")
        monkeypatch.setattr(twopass, "_gemm_rowmax", _Noop())
        Qh, Ch, cs = _operands(40, N, 32, seed=9)
        got = twopass.fused_rowmax(Qh, Ch, cs)
        assert got is not None
        assert got.shape == (40,)
        assert bool(torch.isinf(got).all()) and bool((got > 0).all()), got

    # --- the inequality the liveness decision rests on -------------------

    def test_the_row_max_plus_the_bound_never_falls_below_the_float64_max():
        """`bound(..., half_out=False)` — the margin `upper_bounds` uses when
        the kernel ran — added to the interpreted row max, against the float64
        maximum of the exact product of the same float16 operands.

        This is the assertion the GPU suite makes and CI never reaches.
        """
        g = torch.Generator().manual_seed(41)
        for scale in (1.0, 300.0, 0.004):
            Q = (torch.randn(48, 64, generator=g) * scale)
            C = (torch.randn(40, 64, generator=g) / max(scale, 1e-3))
            C = C.half().float()                     # as fineweb stores it
            cs = C.norm(dim=1).clamp_min(1e-12).reciprocal().contiguous()
            rs = Q.norm(dim=1).reciprocal()
            qs = twopass.query_side(Q, rs)
            Ch, sigma_max, _, cn_max = twopass.corpus_side(C, cs)
            got, _ = _launch(qs["Qh"], Ch, cs, 16, 16, 16, 8)
            # Both sides row-scaled: the closed form's `eps` is in cosine
            # units, and the kernel's raw output has magnitude ~||q||. See the
            # same note in `test_fused_rowmax.py`.
            eps = torch.from_numpy(twopass.bound(
                qs, 64, sigma_max, False, "cosine",
                fmt_c=twopass._cf.EXACT, cn_max=cn_max))
            ref = _ref64(qs["Qh"], Ch, cs) * rs.double()
            scaled = got.double() * rs.double()
            assert torch.all(scaled + eps.double() >= ref), (
                float((ref - scaled - eps.double()).max()))

    def test_the_fused_path_reads_a_float32_accumulator():
        """The docstring's load-bearing claim: no float16 rounding of the Gram
        anywhere on the fused path, which is why the `2**-11 * T` term is
        dropped.

        Checked by construction rather than by reading: a dot whose exact
        value is NOT representable in float16 but IS representable in float32
        must come back exactly. `1 + 2**-13` needs 14 significand bits; fp16
        has 11, fp32 has 24. Both operands are exactly representable in fp16,
        so the products are exact and only the accumulator's width decides.
        """
        K = 16
        Qh = torch.zeros(16, K, dtype=torch.float16)
        Ch = torch.zeros(16, K, dtype=torch.float16)
        Qh[:, 0], Ch[:, 0] = 1.0, 1.0
        Qh[:, 1], Ch[:, 1] = 2.0 ** -13, 1.0
        cs = torch.ones(16, dtype=torch.float32)
        got, _ = _launch(Qh, Ch, cs, 16, 16, 16, 8)
        exact = 1.0 + 2.0 ** -13
        assert float(got[0]) == exact, (
            f"{float(got[0])!r} != {exact!r}: the accumulator rounded to "
            f"float16 ({float(torch.tensor(exact).half())!r}), so the bound "
            f"must keep its 2**-11 * T output term on this path")
