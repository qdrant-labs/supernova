"""Tiles-mode pass two on real hardware (CUDA only; skipped elsewhere).

Ported from benchmarks/streaming/tile_rescore/test_tile_rescore.py (dense mode
only -- the prototype's candidate mode is not in nova_bf, see
`nova_bf.tile_rescore`), plus what only a GPU can show: the kernel against the
torch reference with real MMA numerics, the fused pass one's per-tile buffer,
and a whole tiles-mode run against one-pass.

The CPU side of every behaviour is in `test_twopass_tiles.py` (torch
reference) and `test_tile_rescore_interpret.py` (this kernel, interpreted).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from nova_bf import tile_rescore as tr  # noqa: E402
from nova_bf import twopass  # noqa: E402

dev = torch.device("cuda")


@pytest.fixture(autouse=True)
def _fresh():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    twopass.reset()
    yield
    twopass.reset()
    torch.backends.cuda.matmul.allow_tf32 = prev


def unit(n, d, g):
    x = torch.randn(n, d, generator=g, device=dev)
    return (x / x.norm(dim=1, keepdim=True)).contiguous()


def masks(n_tiles, nq, kind, g):
    if kind == "empty":
        return torch.zeros(n_tiles, nq, dtype=torch.bool, device=dev)
    if kind == "all":
        return torch.ones(n_tiles, nq, dtype=torch.bool, device=dev)
    if kind == "sparse":
        return torch.rand(n_tiles, nq, generator=g, device=dev) < 0.01
    if kind == "hot":           # one tile live for most queries, the rest nearly empty
        m = torch.rand(n_tiles, nq, generator=g, device=dev) < 0.002
        m[n_tiles // 2] |= torch.rand(nq, generator=g, device=dev) < 0.8
        return m
    raise ValueError(kind)


def reference64(Q, C, cn=None, qn=None):
    s = Q.double() @ C.double().T
    if cn is not None:
        s = s / cn.double()[None, :]
    if qn is not None:
        s = s / qn.double()[:, None]
    return s


SHAPES = [  # nq, n_rows, d  (partial last tile, d not a multiple of BK, tiny nq)
    (1, 128, 64), (17, 8192, 768), (300, 8000, 100), (2000, 8192, 2560),
    (20000, 4096, 768),
]
KINDS = ["empty", "sparse", "hot", "all"]


@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("nq,n_rows,d", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_dense_mode_matches_reference(nq, n_rows, d, kind, width):
    g = torch.Generator(device=dev).manual_seed(nq * 7 + d)
    Q, C = unit(nq, d, g), unit(n_rows, d, g)
    cn = (torch.rand(n_rows, generator=g, device=dev) + 0.5).contiguous()
    qn = (torch.rand(nq, generator=g, device=dev) + 0.5).contiguous()
    n_tiles = -(-n_rows // width)
    tl_ = masks(n_tiles, nq, kind, g)
    idx = tl_.any(dim=0).nonzero().flatten()
    buf, used = tr.rescore(Q, C, idx, tl_, width, cn=cn, qn=qn)
    assert buf.shape == (int(idx.numel()), n_rows)
    if not idx.numel():
        return
    assert used, f"the kernel declined: {tr.kernel_off()}"
    ref = reference64(Q[idx], C, cn, qn[idx]).float()
    keep = tl_[:, idx].T.repeat_interleave(width, dim=1)[:, :n_rows]
    assert torch.isneginf(buf[~keep]).all(), "dead tiles must stay -inf"
    err = (buf[keep] - ref[keep]).abs().max().item() if keep.any() else 0.0
    assert err < 1e-5, f"max abs err {err}"
    # And against the torch reference (cuBLAS + the same divides): the
    # documented ULP-level difference of summation order, nothing more.
    want = tr.reference(Q, C, idx, tl_, width, cn=cn, qn=qn)
    assert torch.equal(torch.isneginf(buf), torch.isneginf(want))
    assert (buf[keep] - want[keep]).abs().max().item() < 1e-5


def test_deterministic_run_to_run():
    g = torch.Generator(device=dev).manual_seed(2)
    Q, C = unit(3000, 768, g), unit(8192, 768, g)
    tl_ = masks(64, 3000, "hot", g)
    idx = tl_.any(dim=0).nonzero().flatten()
    a, _ = tr.rescore(Q, C, idx, tl_, 128)
    b, _ = tr.rescore(Q, C, idx, tl_, 128)
    assert torch.equal(a, b)


def test_the_worklist_never_syncs_the_host():
    g = torch.Generator(device=dev).manual_seed(3)
    tl_ = masks(64, 5000, "sparse", g)
    tr.build_worklist(tl_)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tr.build_worklist(tl_)
    finally:
        torch.cuda.set_sync_debug_mode(0)


def test_worklist_groups_queries_by_tile_in_order():
    tl_ = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
                       dtype=torch.bool, device=dev)
    qlist, toff, tcnt, bcum, nblk = tr.build_worklist(tl_, bq=2)
    assert tcnt.tolist() == [3, 0, 2] and toff.tolist() == [0, 3, 3]
    assert qlist[:5].tolist() == [0, 2, 3, 1, 3]
    assert bcum.tolist() == [0, 2, 2, 3] and nblk.item() == 3


def test_the_fused_pass_one_hands_back_its_per_tile_maxima():
    """`with_tiles=True` returns the kernel's own buffer and BLOCK_N; its
    column max is the row max bit for bit, and the tile bounds built from it
    cover every exact score of their tile."""
    g = torch.Generator(device=dev).manual_seed(4)
    Q, C = unit(4096, 768, g), unit(8192, 768, g)
    Qh, Ch = Q.half().contiguous(), C.half().contiguous()
    cs = torch.ones(8192, device=dev)
    got = twopass.fused_rowmax(Qh, Ch, cs, with_tiles=True)
    assert got is not None, twopass.fused_off()
    row, part, bn = got
    assert part.shape == (-(-8192 // bn), 4096)
    assert torch.equal(part.amax(dim=0), row)

    info = {}
    cn = C.norm(dim=1)
    upper = twopass.upper_bounds(Q, C, cn.clamp_min(1e-12).reciprocal(),
                                 Q.norm(dim=1).clamp_min(1e-12).reciprocal(),
                                 None, "cosine", tiles=info)
    if not info:
        pytest.skip(f"pass one refused the slice: {twopass.stats()['last_refusal_reason']}")
    assert info["width"] == bn
    exact = (Q @ C.T) / cn.clamp_min(1e-12)[None, :] / Q.norm(dim=1)[:, None]
    texact = twopass.tile_amax(exact, bn).T
    assert bool((info["upper"] >= texact).all())
    assert bool((info["upper"] <= upper[None, :]).all())


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_a_tiles_run_agrees_with_one_pass_on_cuda(tmp_path, monkeypatch, metric):
    """End to end on the device: fused pass one -> tile bounds -> the Triton
    re-score -> the fold. Tie-aware: tiles mode is not byte-identical."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nova_bf.compute import run_compute
    from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                ParamsConfig, QueriesConfig, SearchSpec)
    from twopass_compare import assert_same_results

    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 256)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    rng = np.random.default_rng(0)
    cdir = tmp_path / "c"
    cdir.mkdir()
    for fi in range(6):
        v = rng.standard_normal((3000, 128)).astype(np.float16).astype(np.float32)
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), type=pa.list_(pa.float32())),
            "id": [f"c{fi}_{r}" for r in range(3000)]}), str(cdir / f"f{fi}.parquet"))
    qp = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(rng.standard_normal((512, 128)).astype(np.float32).tolist(),
                                    type=pa.list_(pa.float32())),
        "qid": [f"q{i}" for i in range(512)]}), str(qp))

    def run(mode, out):
        params = {"dense_batch_size": 1024}
        params.update({"two_pass": "off"} if mode == "off" else {"two_pass_rescore": mode})
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qp), id_column="qid"),
            output=OutputConfig(path=str(tmp_path / out)),
            params=ParamsConfig(**params),
            # One metric per run: two would share a Gram, which the two-pass
            # refuses.
            searches=[SearchSpec(name=metric, metric=metric, k=10)])
        (tmp_path / out).mkdir()
        return {n: pq.read_table(p) for n, p in run_compute(cfg).items()}, twopass.stats()

    off, _ = run("off", "off")
    ti, st = run("tiles", "tiles")
    assert st["slices_tiles"] > 0 and st["tiles_kernel"] > 0, st
    assert st["audit_violations"] == 0 and st["tiles_fallbacks"] == 0, st
    from twopass_compare import norm_product, tolerance

    tol = tolerance(128, 1.0 if metric == "cosine" else norm_product(cdir, qp))
    for name in off:
        assert_same_results(ti[name], off[name], abs_tol=tol, label=name)


@pytest.mark.parametrize("width", [16, 32, 256])
@pytest.mark.parametrize("divides", ["dot", "cn_only", "cn_qn"])
def test_the_largest_admitted_tile_count_and_every_divide_branch(width, divides):
    """~4096 tiles (the most the kernel holds), each width, and the three
    epilogues: no divide (dot), the column divide only (cosine with packer
    scaling), and both."""
    g = torch.Generator(device=dev).manual_seed(width)
    n_tiles = 4096 if width <= 32 else 1024
    n_rows, nq, d = n_tiles * width - 3, 96, 64
    Q, C = unit(nq, d, g), unit(n_rows, d, g)
    cn = (torch.rand(n_rows, generator=g, device=dev) + 0.5) if divides != "dot" else None
    qn = (torch.rand(nq, generator=g, device=dev) + 0.5) if divides == "cn_qn" else None
    tl_ = masks(-(-n_rows // width), nq, "sparse", g)
    tl_[0, 0] = True
    idx = tl_.any(dim=0).nonzero().flatten()
    assert tr.decline_reason(Q, C, tl_, width, cn, qn) is None
    buf, used = tr.rescore(Q, C, idx, tl_, width, cn=cn, qn=qn)
    assert used
    want = tr.reference(Q, C, idx, tl_, width, cn=cn, qn=qn)
    dead = torch.isneginf(want)
    assert torch.equal(torch.isneginf(buf), dead)
    tol = tr.score_tolerance(Q, C, idx, cn, qn)
    assert bool(((buf.double() - want.double()).abs()[~dead] <= tol[~dead]).all())


def test_more_tiles_than_the_kernel_holds_is_declined_with_a_reason():
    Q, C = unit(8, 64, torch.Generator(device=dev).manual_seed(0)), \
        torch.randn(16 * 5000, 64, device=dev)
    tl_ = torch.ones(5000, 8, dtype=torch.bool, device=dev)
    why = tr.decline_reason(Q, C, tl_, 16)
    assert why is not None and "tiles in the slice" in why


def test_the_divides_are_correctly_rounded():
    """Catches `div_rn` replaced by Triton's approximate `/`: with
    small-integer operands every fp32 dot is EXACT in any order, so the only
    rounding left is the divide(s), and the kernel must match torch's IEEE
    division bit for bit."""
    g = torch.Generator(device=dev).manual_seed(7)
    nq, n_rows, d = 256, 4096, 64
    Q = torch.randint(-3, 4, (nq, d), generator=g, device=dev).float()
    C = torch.randint(-3, 4, (n_rows, d), generator=g, device=dev).float()
    cn = torch.rand(n_rows, generator=g, device=dev) * 3 + 0.7
    qn = torch.rand(nq, generator=g, device=dev) * 3 + 0.7
    tl_ = torch.ones(n_rows // 128, nq, dtype=torch.bool, device=dev)
    idx = torch.arange(nq, device=dev)
    buf, used = tr.rescore(Q, C, idx, tl_, 128, cn=cn, qn=qn)
    assert used
    exact_dot = (Q.double() @ C.double().T).float()      # exact: small ints
    want = (exact_dot / cn[None, :]) / qn[:, None]       # IEEE fp32 divides
    assert torch.equal(buf, want), (
        f"{int((buf != want).sum())} scores differ: an approximate divide")


def test_a_warm_rescore_never_syncs_the_host():
    """After the first launch (whose self-check reads back), the whole
    `rescore` call -- worklist, buffer, launch -- is free of host syncs."""
    g = torch.Generator(device=dev).manual_seed(3)
    Q, C = unit(5000, 768, g), unit(8192, 768, g)
    tl_ = masks(64, 5000, "sparse", g)
    idx = tl_.any(dim=0).nonzero().flatten()
    cn = C.norm(dim=1).clamp_min(1e-12)
    tr.rescore(Q, C, idx, tl_, 128, cn=cn)               # compile + self-check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        tr.rescore(Q, C, idx, tl_, 128, cn=cn)
    finally:
        torch.cuda.set_sync_debug_mode(0)


def test_the_first_launch_self_check_is_bounded_at_production_scale(
        monkeypatch):
    """A first launch with thousands of live queries: the self-check's
    reference and fp64 tolerance GEMMs see SELF_CHECK_ROWS + BQ of them (the
    full-height fp64 GEMM was ~1.8 s on an A10G), and the real kernel's
    sampled rows pass."""
    g = torch.Generator(device=dev).manual_seed(5)
    Q, C = unit(5000, 768, g), unit(8192, 768, g)
    tl_ = masks(64, 5000, "sparse", g)
    idx = tl_.any(dim=0).nonzero().flatten()
    assert int(idx.numel()) > 4 * tr.SELF_CHECK_ROWS
    cn = C.norm(dim=1).clamp_min(1e-12)
    rows = []
    real = tr.score_tolerance

    def tol(Q_, C_, idx_, *a, **kw):
        rows.append(int(idx_.numel()))
        return real(Q_, C_, idx_, *a, **kw)

    monkeypatch.setattr(tr, "score_tolerance", tol)
    _, used = tr.rescore(Q, C, idx, tl_, 128, cn=cn)
    assert used and rows == [len(tr._self_check_rows(int(idx.numel())))]
    assert rows[0] <= tr.SELF_CHECK_ROWS + tr.BQ
    assert tr.usage()["selfchecks"] == 1
    assert tr.usage()["selfcheck_failures"] == 0


@pytest.mark.parametrize("d", [768, 1024])
def test_a_realistic_tiles_run_agrees_with_one_pass(tmp_path, monkeypatch, d):
    """Production-like shape: d=768/1024, k=100, 8192-row slices, the fused
    pass one's own tile width; compared within the gamma_d bound."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nova_bf.compute import run_compute
    from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                ParamsConfig, QueriesConfig, SearchSpec)
    from twopass_compare import assert_same_results, tolerance

    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 2048)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    rng = np.random.default_rng(d)
    cdir = tmp_path / "c"
    cdir.mkdir()
    centers = rng.standard_normal((64, d)).astype(np.float32)
    for fi in range(8):
        v = centers[rng.integers(0, 64, 16384)] + 0.6 * rng.standard_normal((16384, d))
        v = v.astype(np.float16).astype(np.float32)
        pq.write_table(pa.table({
            "dense_embedding": pa.array(list(v), type=pa.list_(pa.float32())),
            "id": [f"c{fi}_{r}" for r in range(16384)]}), str(cdir / f"f{fi}.parquet"))
    q = centers[rng.integers(0, 64, 4096)] + 0.6 * rng.standard_normal((4096, d))
    qp = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(list(q.astype(np.float32)), type=pa.list_(pa.float32())),
        "qid": [f"q{i}" for i in range(4096)]}), str(qp))

    def run(mode, out):
        params = {"dense_batch_size": 8192}
        params.update({"two_pass": "off"} if mode == "off" else {"two_pass_rescore": mode})
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qp), id_column="qid"),
            output=OutputConfig(path=str(tmp_path / out)),
            params=ParamsConfig(**params),
            searches=[SearchSpec(name="cos", metric="cosine", k=100)])
        (tmp_path / out).mkdir()
        twopass.reset()
        return {n: pq.read_table(p) for n, p in run_compute(cfg).items()}, twopass.stats()

    off, _ = run("off", "off")
    ti, st = run("tiles", "tiles")
    assert st["slices_tiles"] > 0 and st["tiles_kernel"] > 0, st
    assert st["tiles_selfcheck_failures"] == 0 and st["audit_violations"] == 0, st
    assert_same_results(ti["cos"], off["cos"], abs_tol=tolerance(d, 1.0), label="cos")
