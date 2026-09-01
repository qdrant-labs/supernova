"""Per-query top-K pruning, and the two silent ways it can go wrong.

Deep into a corpus scan, most slices hold nothing that can enter a given
query's top-K: the running state's weakest packed key (`spec_thr`) is a bar the
slice's best candidate must at least tie (`tiebreak.live_rows`). Rows that
cannot are skipped by both the pre-top-K (`topk_triton._cutfill`) and the fold
(`merge_triton._fold`). The skip is exact — a dead row's candidates all pack
strictly below the state's weakest key — so the ANSWER must be bit-identical
with pruning on, off, kernel, or portable.

Neither failure mode raises, so both directions are pinned here:

  * an over-eager skip silently drops a real candidate (wrong ground truth);
  * a skip that stops engaging is a silent perf regression.

These tests run the portable paths on CPU and the Triton kernels wherever
`run_compute` selects CUDA, so the same file covers both on a GPU box.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import compute as compute_mod
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig, QueriesConfig,
    SearchSpec,
)
from nova_bf.tiebreak import (
    SENTINEL_KEY, TIE_WORST, live_rows, pack, pack_topk, sentinel_key,
)

DIM = 8


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------


def test_sentinel_key_constant_matches_sentinel_key():
    """`SENTINEL_KEY` (the plain-int fill value) must be `sentinel_key`'s
    value bit for bit, or the sanitize in `_merge_topk` writes keys that do
    not compare like sentinels."""
    assert int(sentinel_key((1,), "cpu").item()) == SENTINEL_KEY


def test_real_keys_strictly_outrank_the_sentinel():
    """The prune's under-filled-state guarantee leans on this: ordinals are
    capped at U32-1 (`MAX_ROWS_PER_WORKER` rejects a slice that would use
    U32), so even a real `-inf` candidate packs strictly above the sentinel
    and an empty slot always fills."""
    import torch

    worst_real = pack(
        torch.tensor([[float("-inf")]], dtype=torch.float32),
        torch.tensor([TIE_WORST - 1], dtype=torch.int64),
    )
    assert int(worst_real.item()) > SENTINEL_KEY


def test_live_rows_rule_boundaries():
    """live iff the row's best SCORE half >= thr's score half: strictly-below
    is dead, an exact score tie is live (the ordinal may still win), and a
    sentinel threshold (under-filled state) keeps everything live, `-inf`
    candidates included."""
    import torch

    o = torch.arange(1, dtype=torch.int64)
    # A full state whose weakest entry scores 0.5 with ordinal 5.
    state = pack(torch.tensor([[0.5, 0.7]], dtype=torch.float32),
                 torch.tensor([5, 6], dtype=torch.int64))
    thr = state.min(dim=1).values.expand(4).contiguous()
    cand = pack(torch.tensor(
        [[0.49999], [0.5], [0.50001], [float("-inf")]], dtype=torch.float32), o)
    assert live_rows(cand, thr).tolist() == [0, 1, 1, 0]

    # Under-filled: the sentinel min must keep even -inf rows live.
    thr_sent = sentinel_key((2,), "cpu")
    cand2 = pack(torch.tensor([[float("-inf")], [-1e38]], dtype=torch.float32), o)
    assert live_rows(cand2, thr_sent).tolist() == [1, 1]


def test_pack_topk_live_matches_the_portable_rule():
    """Whichever path `pack_topk` takes must make the SAME live decision the
    one-place rule (`live_rows` over the full packed row) makes. On CUDA this
    pins the kernel's in-register decision to the portable one; on CPU it
    pins the portable path to itself (still catches a drifted rule)."""
    import torch

    from nova_bf import topk_triton

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        g = torch.Generator(device="cpu").manual_seed(7)
        sc = torch.randn(64, 33, generator=g).to(dev)
        # quantize hard so exact cross-row score ties actually occur
        sc = (sc * 4).round() / 4
        od = torch.arange(33, dtype=torch.int64, device=dev)
        k = 5
        # threshold from a plausible state: the k-th best of the first half
        half_keys = pack(sc[:, :16].contiguous(), od[:16])
        thr = torch.topk(half_keys, k=k, dim=1).values.min(dim=1).values.contiguous()

        keys, idx, live = pack_topk(sc.contiguous(), od, k, thr=thr)
        want = live_rows(pack(sc, od), thr)
        assert torch.equal(live.cpu(), want.cpu()), f"live diverged on {dev}"
        # live rows must carry the exact selection; dead rows' keys are
        # unspecified but their idx must stay gatherable
        full = pack(sc, od)
        want_keys = torch.topk(full, k=k, dim=1, sorted=False).values
        lv = live.bool()
        assert torch.equal(
            keys[lv].sort(dim=1).values, want_keys[lv].sort(dim=1).values
        ), f"a live row's keys diverged on {dev}"
        assert int(idx.min()) >= 0 and int(idx.max()) < 33


# ---------------------------------------------------------------------------
# the fold under pruning
# ---------------------------------------------------------------------------


def _real_state(n_q, k, device, seed=0):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    sc = torch.randn(n_q, k, generator=g).float().to(device) + 10.0  # all real, high
    key = pack(sc, torch.arange(k, dtype=torch.int64, device=device))
    enc = torch.arange(n_q * k, dtype=torch.int64, device=device).reshape(n_q, k)
    return key, enc


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_merge_topk_never_reads_a_dead_row(device):
    """The sharpest mutation catch: dead rows' part keys are POISONED with
    int64 max — if any path (kernel skip, multi-part sanitize, portable
    sanitize) reads one as a key, it wins the fold and the state is visibly
    wrong. Covers single-part, multi-part with differing masks, and a
    mask-less part mixed in."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    n_q, k = 6, 4
    key, enc = _real_state(n_q, k, device)
    thr = key.min(dim=1).values.contiguous()
    thr0 = thr.clone()
    poison_val = 2**63 - 1  # outranks every real key here (no NaNs in play)
    poison = torch.full((n_q, 3), poison_val, dtype=torch.int64, device=device)

    # single part, alternating live; live rows carry one real improving key
    live = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.uint8, device=device)
    good = pack(torch.full((n_q, 3), 99.0, device=device),
                torch.arange(3, dtype=torch.int64, device=device))
    pk = torch.where(live.bool().unsqueeze(1), good, poison)
    pe = torch.full((n_q, 3), 7, dtype=torch.int64, device=device)
    nk, ne = compute_mod._merge_topk(key.clone(), enc.clone(), [(pk, pe, live)], k, thr=thr)
    assert not (nk == poison_val).any(), "a poisoned dead row's key entered the state"
    assert torch.equal(thr[live == 0], thr0[live == 0]), "a skipped row's thr moved"
    assert (ne[live.bool()] == 7).any(axis=1).all(), "a live row's candidate was lost"
    # The state is unordered within a row (the portable fold re-selects and
    # may permute; the kernel skip copies through) — compare as key/enc pairs.
    dead = ~live.bool()
    sn, on_ = nk[dead].sort(dim=1)
    so, oo = key[dead].sort(dim=1)
    assert torch.equal(sn, so), "a dead row's state changed"
    assert torch.equal(ne[dead].gather(1, on_), enc[dead].gather(1, oo)), \
        "a dead row's ids changed"
    assert torch.equal(thr, nk.min(dim=1).values), "thr is no longer the state min"

    # multi part: masks differ per part, plus one mask-less (all-valid) part
    key2, enc2 = _real_state(n_q, k, device, seed=1)
    thr2 = key2.min(dim=1).values.contiguous()
    lA = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8, device=device)
    lB = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.uint8, device=device)
    pA = torch.where(lA.bool().unsqueeze(1), good, poison)
    pB = torch.where(lB.bool().unsqueeze(1), good, poison)
    pC = pack(torch.full((n_q, 1), 100.0, device=device),  # beats every other candidate
              torch.tensor([9], dtype=torch.int64, device=device))  # live=None part
    parts = [
        (pA, torch.full((n_q, 3), 11, dtype=torch.int64, device=device), lA),
        (pB, torch.full((n_q, 3), 22, dtype=torch.int64, device=device), lB),
        (pC, torch.full((n_q, 1), 33, dtype=torch.int64, device=device), None),
    ]
    nk2, ne2 = compute_mod._merge_topk(key2.clone(), enc2.clone(), parts, k, thr=thr2)
    assert not (nk2 == poison_val).any(), "poison crossed a multi-part flush"
    assert torch.equal(thr2, nk2.min(dim=1).values)
    # the mask-less part's candidate (score 100, the global best) must appear on EVERY row
    assert (ne2 == 33).any(axis=1).all(), "a mask-less part's rows were skipped"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_merge_topk_pruned_equals_unpruned(device):
    """With valid (portable-packed) parts, thr on and thr off must produce the
    identical state — pruning may only skip what cannot matter."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    g = torch.Generator(device="cpu").manual_seed(3)
    n_q, k, w = 32, 8, 5
    key, enc = _real_state(n_q, k, device, seed=2)
    sc = (torch.randn(n_q, w, generator=g) * 8 + 6).float().to(device)
    pk = pack(sc, torch.arange(w, dtype=torch.int64, device=device))
    pe = torch.arange(w, dtype=torch.int64, device=device) + 10_000
    thr = key.min(dim=1).values.contiguous()
    live = live_rows(pk, thr)
    assert 0 < int(live.sum()) < n_q, "fixture no longer exercises both sides"

    ak, ae = compute_mod._merge_topk(key.clone(), enc.clone(),
                                     [(pk.clone(), pe, live)], k, thr=thr)
    bk, be = compute_mod._merge_topk(key.clone(), enc.clone(),
                                     [(pk.clone(), pe, None)], k)
    sa, oa = ak.sort(dim=1)
    sb, ob = bk.sort(dim=1)
    assert torch.equal(sa, sb)
    assert torch.equal(ae.gather(1, oa), be.gather(1, ob))


# ---------------------------------------------------------------------------
# end-to-end: answer preserved, and the skip actually engages
# ---------------------------------------------------------------------------


def _oracle_topk_ids(qvec, cvecs, k, ids):
    """Independent f64 cosine top-k, best first, ties by ascending id."""
    q = qvec.astype(np.float64)
    c = cvecs.astype(np.float64)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    s = (cn @ q) / np.linalg.norm(q)
    order = sorted(range(len(ids)), key=lambda i: (-s[i], ids[i]))
    return [ids[i] for i in order[:k]]


def _many_file_corpus(tmp_path, n_files, per_file, dim=DIM, seed=0):
    """Several files so the running top-K fills early and later slices mostly
    cannot contribute — the regime where the prune actually fires."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "c"
    cdir.mkdir(exist_ok=True)
    vecs, ids = [], []
    for f in range(n_files):
        v = rng.normal(size=(per_file, dim)).astype(np.float32)
        i = [f"c{f:02d}_{j:04d}" for j in range(per_file)]
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), pa.list_(pa.float32())),
            "sid": pa.array(i),
        }), str(cdir / f"f{f:02d}.parquet"))
        vecs.append(v)
        ids.extend(i)
    return cdir, np.concatenate(vecs), ids


def _run_dense(tmp_path, cdir, qpath, out_name, k, batch):
    out = tmp_path / out_name
    out.mkdir(exist_ok=True)
    return pq.read_table(run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=batch, tiebreak="id"),
        searches=[SearchSpec(name="s", k=k, metric="cosine")],
    ))["s"]).to_pydict()


def _write_queries(tmp_path, qv):
    pq.write_table(pa.table({
        "dense_embedding": pa.array(qv.tolist(), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(len(qv))]),
    }), str(tmp_path / "q.parquet"))
    return tmp_path / "q.parquet"


def test_prune_is_answer_preserving(tmp_path):
    """Against an independent f64 oracle, on a corpus deep enough that rows
    really are being skipped."""
    cdir, cvecs, ids = _many_file_corpus(tmp_path, n_files=8, per_file=200, seed=5)
    rng = np.random.default_rng(99)
    qv = rng.normal(size=(6, DIM)).astype(np.float32)
    qpath = _write_queries(tmp_path, qv)

    t = _run_dense(tmp_path, cdir, qpath, "o", k=10, batch=32)
    for row, qid in enumerate(t["query_id"]):
        want = _oracle_topk_ids(qv[int(qid[1:])], cvecs, 10, ids)
        assert t["hit_ids"][row] == want, f"{qid}: skipped a live candidate"


def test_prune_matches_prune_disabled(tmp_path, monkeypatch):
    """NOVA_BF_NO_PRUNE must be a pure perf knob: bit-identical output tables
    either way."""
    cdir, _, _ = _many_file_corpus(tmp_path, n_files=6, per_file=150, seed=11)
    rng = np.random.default_rng(4)
    qv = rng.normal(size=(5, DIM)).astype(np.float32)
    qpath = _write_queries(tmp_path, qv)

    on = _run_dense(tmp_path, cdir, qpath, "on", k=12, batch=64)
    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    off = _run_dense(tmp_path, cdir, qpath, "off", k=12, batch=64)
    assert on["hit_ids"] == off["hit_ids"]
    assert on["hit_scores"] == off["hit_scores"]


def test_prune_actually_engages(tmp_path, monkeypatch):
    """If the skip stops firing this whole feature becomes a silent no-op.
    Spies on `_merge_topk`'s parts: on a 10-file corpus with k=5, some flush
    must arrive with at least one dead row."""
    import torch

    # This test asserts pruning HAPPENS, so an ambient kill switch would fail
    # it spuriously rather than reveal anything.
    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)

    seen = {"dead": 0, "flushes_with_masks": 0}
    real = compute_mod._merge_topk

    def spy(top_key, top_enc, parts, k, thr=None):
        for _, _, live in parts:
            if live is not None:
                seen["flushes_with_masks"] += 1
                seen["dead"] += int((live == 0).sum())
        return real(top_key, top_enc, parts, k, thr=thr)

    monkeypatch.setattr(compute_mod, "_merge_topk", spy)
    cdir, cvecs, ids = _many_file_corpus(tmp_path, n_files=10, per_file=300, seed=7)
    rng = np.random.default_rng(3)
    qv = rng.normal(size=(4, DIM)).astype(np.float32)
    qpath = _write_queries(tmp_path, qv)
    t = _run_dense(tmp_path, cdir, qpath, "o2", k=5, batch=64)
    assert seen["flushes_with_masks"] > 0, "no part ever carried a live mask"
    assert seen["dead"] > 0, (
        "no row was ever prunable on a 10-file corpus with k=5 — the prune "
        "has stopped engaging"
    )
    # and the pruned answer is still the oracle's
    for row, qid in enumerate(t["query_id"]):
        want = _oracle_topk_ids(qv[int(qid[1:])], cvecs, 5, ids)
        assert t["hit_ids"][row] == want


def test_skip_respects_the_tiebreak_ordinal(tmp_path):
    """All scores EQUAL, ids DESCENDING in scan order: the last file read
    carries the LOWEST ids, so under tiebreak='id' the winners arrive last,
    into a state already full of same-score entries. A skip that compares
    score halves with `>` instead of `>=` (or decides on scores while
    ignoring that ordinals may still win) drops them silently. With ids
    ascending in scan order every later tied candidate legitimately loses,
    so 'skip them all' would accidentally be right — which is why this
    fixture inverts the ids."""
    vec = [1.0] + [0.0] * (DIM - 1)
    n_files, per_file = 6, 50
    cdir = tmp_path / "c"
    cdir.mkdir(exist_ok=True)
    for f in range(n_files):
        pq.write_table(pa.table({
            "dense_embedding": pa.array([vec] * per_file, pa.list_(pa.float32())),
            "sid": pa.array([f"c{n_files - 1 - f:02d}_{j:04d}" for j in range(per_file)]),
        }), str(cdir / f"f{f:02d}.parquet"))
    qpath = _write_queries(tmp_path, np.array([vec], dtype=np.float32))

    t = _run_dense(tmp_path, cdir, qpath, "o3", k=12, batch=16)
    got = t["hit_ids"][0]
    assert len(got) == 12
    # `tiebreak="id"` must return the 12 lowest ids overall — which live in
    # the file scanned LAST.
    assert got == sorted(
        f"c{f:02d}_{j:04d}" for f in range(n_files) for j in range(per_file)
    )[:12]
    assert got[0].startswith("c00_"), "the last-scanned file's ids should win"


def test_underfilled_state_is_never_skipped(tmp_path):
    """k above the whole corpus: every row must land in the output, however
    late (and low-scoring) its file arrives. An over-eager skip against a
    sentinel-filled state loses the tail files entirely."""
    cdir, cvecs, ids = _many_file_corpus(tmp_path, n_files=5, per_file=8, seed=13)
    rng = np.random.default_rng(8)
    qv = rng.normal(size=(3, DIM)).astype(np.float32)
    qpath = _write_queries(tmp_path, qv)

    t = _run_dense(tmp_path, cdir, qpath, "o4", k=64, batch=16)
    for row in range(3):
        assert sorted(t["hit_ids"][row]) == sorted(ids), (
            "an under-filled top-K lost rows — a skip fired against a "
            "sentinel-filled state"
        )
