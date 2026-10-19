"""Tiles mode's two Triton pieces executed for real, on CPU, in the interpreter.

`_tile_rescore` (tiles-mode pass two) and the per-tile buffer `fused_rowmax`
now hands back (`with_tiles=True`) only run on a GPU in production, and this
suite has to cover them on a machine without one. `TRITON_INTERPRET=1` runs
the real `@triton.jit` bodies as numpy, which covers what a review cannot check
by eye: the persistent loop over the device-side work list, the tile-owner
search, the per-tile query lists, the row/column masks of a partial tile, the
scatter into the `-inf` buffer, and that the fused kernel's per-tile maxima are
what the bound reads. It does NOT cover GPU numerics (`np.matmul` stands in for
the MMA), so numeric checks are "the right products were summed", within a
few ULP of the torch reference; `tests/test_tile_rescore_gpu.py` holds the
hardware side.

Same two environment facts as `test_fused_rowmax_interpret.py`: the variable
must be set before `triton` is imported, so the tests run in a SUBPROCESS; and
`nova_bf.twopass` uses `from __future__ import annotations`, so the child
widens the interpreter's constexpr check exactly as that file does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

_CHILD = os.environ.get("NOVA_BF_TILE_INTERPRET_CHILD") == "1"
_MIN_CHILD_TESTS = 20


def test_the_interpreted_suite_passes():
    """Re-run this file with `TRITON_INTERPRET=1` set before `import triton`."""
    if _CHILD:
        pytest.skip("this process IS the interpreted child")
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["NOVA_BF_TILE_INTERPRET_CHILD"] = "1"
    for k in ("NOVA_BF_NO_FUSED_ROWMAX", "NOVA_BF_FUSE_CONFIG",
              "NOVA_BF_NO_TILE_KERNEL"):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(
            os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "-p", "no:cacheprovider", os.path.abspath(__file__)],
        env=env, capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, (
        f"the interpreted tile suite failed:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"(\d+) passed", proc.stdout)
    assert m, f"the child reported no passing tests:\n{proc.stdout}"
    assert int(m.group(1)) >= _MIN_CHILD_TESTS, (
        f"the interpreted child ran only {m.group(1)} tests:\n{proc.stdout}")


if _CHILD:
    assert os.environ.get("TRITON_INTERPRET") == "1"

    import contextlib

    import triton.runtime.jit as _jit

    _orig_normalize = _jit._normalize_ty

    def _normalize(ty):
        out = _orig_normalize(ty)
        if isinstance(out, str) and out.rsplit(".", 1)[-1] == "constexpr":
            return "constexpr"
        return out

    _jit._normalize_ty = _normalize

    import torch                                     # noqa: E402

    from nova_bf import tile_rescore as tr           # noqa: E402
    from nova_bf import twopass                      # noqa: E402

    @pytest.fixture(autouse=True)
    def _fresh():
        twopass.reset()
        yield
        twopass.reset()

    def _masks(n_tiles, nq, kind, g):
        if kind == "empty":
            return torch.zeros(n_tiles, nq, dtype=torch.bool)
        if kind == "all":
            return torch.ones(n_tiles, nq, dtype=torch.bool)
        if kind == "sparse":
            return torch.rand(n_tiles, nq, generator=g) < 0.1
        if kind == "hot":       # one tile live for most queries (> BQ of them)
            m = torch.rand(n_tiles, nq, generator=g) < 0.02
            m[n_tiles // 2] |= torch.rand(nq, generator=g) < 0.9
            return m
        raise ValueError(kind)

    def _case(nq, n_rows, d, width, kind, seed, metric):
        g = torch.Generator().manual_seed(seed)
        Q = torch.randn(nq, d, generator=g)
        C = torch.randn(n_rows, d, generator=g)
        n_tiles = -(-n_rows // width)
        tl_ = _masks(n_tiles, nq, kind, g)
        idx = tl_.any(dim=0).nonzero().flatten()
        cn = qn = None
        if metric in ("cosine", "cosine_q"):
            cn = C.norm(dim=1).clamp_min(1e-12)
        if metric == "cosine_q":
            qn = Q.norm(dim=1)
        return Q, C, idx, tl_, cn, qn

    def test_the_kernel_is_the_one_being_interpreted():
        from triton.runtime.interpreter import InterpretedFunction

        assert tr._tile_rescore is not None, tr._UNAVAILABLE
        assert isinstance(tr._tile_rescore, InterpretedFunction)
        Q, C, idx, tl_, cn, qn = _case(40, 100, 48, 32, "sparse", 0, "dot")
        _, used = tr.rescore(Q, C, idx, tl_, 32)
        assert used, "the kernel declined, so this file would test the reference"
        assert tr.usage()["launches"] == 1

    SHAPES = [
        # nq, n_rows, d, width: partial last tile, d not a multiple of BK,
        # one query, one tile, more live queries in a tile than BQ (=32).
        (1, 16, 64, 16), (37, 200, 70, 32), (70, 128, 33, 64),
        (90, 300, 96, 16), (5, 10, 64, 16), (64, 257, 40, 128),
    ]

    @pytest.mark.parametrize("nq,n_rows,d,width", SHAPES)
    @pytest.mark.parametrize("kind", ["empty", "sparse", "hot", "all"])
    @pytest.mark.parametrize("metric", ["dot", "cosine", "cosine_q"])
    def test_the_kernel_matches_the_reference(nq, n_rows, d, width, kind, metric):
        Q, C, idx, tl_, cn, qn = _case(nq, n_rows, d, width, kind,
                                       nq * 7 + d, metric)
        got, used = tr.rescore(Q, C, idx, tl_, width, cn=cn, qn=qn)
        want = tr.reference(Q, C, idx, tl_, width, cn=cn, qn=qn)
        assert got.shape == want.shape == (int(idx.numel()), n_rows)
        if not idx.numel():
            return
        assert used
        dead = torch.isneginf(want)
        assert torch.equal(torch.isneginf(got), dead), (
            "the kernel wrote a dead pair, or left a live one at -inf")
        live = ~dead
        # Different summation order only: a few ULP of each score's scale.
        scale = (Q.index_select(0, idx).abs() @ C.abs().T)
        if cn is not None:
            scale = scale / cn[None, :]
        if qn is not None:
            scale = scale / qn.index_select(0, idx)[:, None]
        err = (got - want).abs()[live]
        bound = (4 * twopass.gamma(d) * scale[live]) + 1e-30
        assert bool((err <= bound).all()), float((err - bound).max())

    def test_the_worklist_groups_queries_by_tile_in_order():
        tl_ = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
                           dtype=torch.bool)
        qlist, toff, tcnt, bcum, nblk = tr.build_worklist(tl_, bq=2)
        assert tcnt.tolist() == [3, 0, 2] and toff.tolist() == [0, 3, 3]
        assert qlist[:5].tolist() == [0, 2, 3, 1, 3]
        assert bcum.tolist() == [0, 2, 2, 3] and int(nblk) == 3

    def test_the_kernel_is_deterministic():
        Q, C, idx, tl_, cn, qn = _case(60, 160, 64, 32, "hot", 3, "cosine_q")
        a, _ = tr.rescore(Q, C, idx, tl_, 32, cn=cn, qn=qn)
        b, _ = tr.rescore(Q, C, idx, tl_, 32, cn=cn, qn=qn)
        assert torch.equal(a, b)

    def test_fused_rowmax_returns_its_own_per_tile_buffer(monkeypatch):
        """`with_tiles=True` returns the kernel's `part` -- the per-(BLOCK_N
        tile, row) maxima -- whose column max IS the returned row max, bit
        for bit, and each entry is that tile's scaled maximum."""
        # The production function, minus only its CUDA gates.
        monkeypatch.setattr(twopass, "fuse_available", lambda *a: True)
        monkeypatch.setattr(torch.cuda, "device",
                            lambda *_a: contextlib.nullcontext())
        g = torch.Generator().manual_seed(4)
        M, N, K = 40, 70, 48
        Qh = torch.randn(M, K, generator=g).half().contiguous()
        Ch = torch.randn(N, K, generator=g).half().contiguous()
        cs = (torch.rand(N, generator=g) + 0.5).float().contiguous()
        cfg = (16, 32, 16, 8, 1, 4)
        row, part, bn = twopass.fused_rowmax(Qh, Ch, cs, config=cfg,
                                             with_tiles=True)
        assert bn == 32 and part.shape == (-(-N // 32), M)
        assert torch.equal(part.amax(dim=0), row)
        plain = twopass.fused_rowmax(Qh, Ch, cs, config=cfg)
        assert torch.equal(plain, row), "with_tiles changed the row max"
        ref = twopass.tile_amax(
            (Qh.double() @ Ch.double().T) * cs.double()[None, :], 32).T
        assert torch.allclose(part.double(), ref, rtol=1e-3, atol=1e-3)

    def test_the_self_check_runs_once_per_configuration_and_passes():
        Q, C, idx, tl_, cn, qn = _case(40, 100, 48, 32, "sparse", 5, "cosine_q")
        tr.rescore(Q, C, idx, tl_, 32, cn=cn, qn=qn)
        tr.rescore(Q, C, idx, tl_, 32, cn=cn, qn=qn)
        assert tr.usage()["selfchecks"] == 1
        tr.rescore(Q, C, idx, tl_, 32, cn=cn)               # other divides
        _, _, idx16, tl16, _, _ = _case(40, 100, 48, 16, "sparse", 6, "dot")
        tr.rescore(Q, C, idx16, tl16, 16)                   # other width
        u = tr.usage()
        assert u["selfchecks"] == 3 and u["selfcheck_failures"] == 0
        assert tr.kernel_off() is None

    def test_the_self_check_compares_a_bounded_sample_of_the_real_launch(
            monkeypatch):
        """The first launch's check reads the launch's OWN output rows (no
        second kernel run) and does reference / tolerance work on at most
        SELF_CHECK_ROWS + BQ of them, first and last live query included;
        the per-tile pairs and the whole dead-pair mask pass too."""
        monkeypatch.setattr(tr, "SELF_CHECK_ROWS", 8)
        Q, C, idx, tl_, cn, qn = _case(90, 300, 96, 16, "hot", 9, "cosine_q")
        n_live = int(idx.numel())
        assert n_live > 8 + tr.BQ
        seen, launches = [], []
        real_ref, real_tol = tr.reference, tr.score_tolerance
        real_check = tr._self_check

        def ref(Q_, C_, idx_, *a, **kw):
            seen.append(("ref", idx_.tolist()))
            return real_ref(Q_, C_, idx_, *a, **kw)

        def tol(Q_, C_, idx_, *a, **kw):
            seen.append(("tol", idx_.tolist()))
            return real_tol(Q_, C_, idx_, *a, **kw)

        def check(*a):
            launches.append(tr.usage()["launches"])
            return real_check(*a)

        monkeypatch.setattr(tr, "reference", ref)
        monkeypatch.setattr(tr, "score_tolerance", tol)
        monkeypatch.setattr(tr, "_self_check", check)
        got, used = tr.rescore(Q, C, idx, tl_, 16, cn=cn, qn=qn)
        assert used and launches == [1] and tr.usage()["launches"] == 1
        want = [int(idx[r]) for r in tr._self_check_rows(n_live)]
        assert seen == [("ref", want), ("tol", want)]
        assert want[0] == int(idx[0]) and want[-1] == int(idx[-1])
        assert tr.usage()["selfchecks"] == 1
        assert tr.usage()["selfcheck_failures"] == 0

    def test_a_new_tile_count_bucket_is_self_checked_on_its_first_launch():
        """NT_P2 is a constexpr -- a separate compile per power-of-two tile
        bucket -- so a short last slice's first launch is checked too; a
        second slice in an already-checked bucket is not."""
        for n_rows, checks in ((100, 1), (120, 1), (40, 2), (36, 2)):
            Q, C, idx, tl_, cn, qn = _case(10, n_rows, 32, 32, "all", n_rows,
                                           "dot")
            _, used = tr.rescore(Q, C, idx, tl_, 32)
            assert used and tr.usage()["selfchecks"] == checks, n_rows

    def test_a_misaligned_operand_is_a_new_configuration_and_is_checked():
        """Triton specializes every pointer on 16-byte alignment, so the
        same shapes through a misaligned (contiguous) view compile anew:
        that first launch is self-checked too; an aligned repeat is not."""
        g = torch.Generator().manual_seed(8)
        big = torch.randn(11, 33, generator=g)       # row = 132 bytes
        C = torch.randn(64, 33, generator=g)
        tl_ = torch.ones(2, 10, dtype=torch.bool)
        idx = torch.arange(10)
        for Q, checks in ((big[:-1], 1), (big[1:], 2), (big[:-1].clone(), 2)):
            assert Q.is_contiguous()
            _, used = tr.rescore(Q, C, idx, tl_, 32)
            assert used and tr.usage()["selfchecks"] == checks, checks

    def test_one_compiled_kernel_serves_every_tile_count():
        """`n_tiles` is a runtime argument: only its power-of-two bucket is a
        compile-time constant, so slices of different heights share a kernel."""
        import inspect

        src = inspect.getsource(tr)
        assert "N_TILES: tl.constexpr" not in src and "n_tiles,\n" in src
        for n_rows in (100, 120):                            # 4 tiles each at 32
            Q, C, idx, tl_, cn, qn = _case(10, n_rows, 32, 32, "all", n_rows, "dot")
            got, used = tr.rescore(Q, C, idx, tl_, 32)
            assert used
