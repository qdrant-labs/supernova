"""Per-query pruning through the search paths that are NOT plain dense cosine.

`test_topk_prune.py` and `test_prune_gpu_contract.py` are dense-cosine only.
The prune interacts with each vector type differently, and the corpora in
`test_tiebreak_paths.py` cannot be reused as-is: they are built so every
candidate scores EXACTLY equal, which is the one input where nothing is ever
prunable (every row ties the threshold and stays live). The fixtures here are
deliberately varied-score, deep enough for the running top-K to fill early so
later slices really are skippable.

Each test runs under `gpu_contract.install()`, so dead rows carry a key that
outranks everything — a path that reads one produces a visibly wrong answer
rather than passing on values the GPU would never have written. Each asserts
`dead_rows_seen > 0` so it cannot pass vacuously.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

import gpu_contract
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig, QueriesConfig,
    RowSelector, SearchSpec,
)

SPARSE_TYPE = pa.struct([
    pa.field("indices", pa.list_(pa.uint32())),
    pa.field("values", pa.list_(pa.float32())),
])
DIM = 8
VOCAB = 32
# "native" runs the REAL kernel and is meaningful only where there is one:
# on CPU there is no Triton path and dead rows are already valid, so it would
# duplicate `test_topk_prune.py` rather than add coverage.
MODES = ["fold", "decline", "nofold"]
if __import__("torch").cuda.is_available():
    MODES.append("native")


def _read(paths: dict) -> dict:
    return {name: pq.read_table(p).to_pydict() for name, p in paths.items()}


def _assert_same(a: dict, b: dict, what: str) -> None:
    assert a.keys() == b.keys(), what
    for name in a:
        assert a[name]["query_id"] == b[name]["query_id"], f"{what}/{name} qids"
        assert a[name]["hit_ids"] == b[name]["hit_ids"], f"{what}/{name} ids"
        assert a[name]["hit_scores"] == b[name]["hit_scores"], f"{what}/{name} scores"


# ---------------------------------------------------------------------------
# sparse — the -inf zero-overlap cell is the interesting input
# ---------------------------------------------------------------------------


def _sparse_corpus(tmp_path, n_files=6, per_file=180, seed=0):
    """Varied weights, and every third row shares NO term with any query, so
    the no-overlap gate `-inf`s whole cells. Those are exactly the rows that
    stay live under a sentinel threshold, and the ones a wrong prune would
    silently drop."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "csparse"
    cdir.mkdir(exist_ok=True)
    for f in range(n_files):
        rows, ids = [], []
        for j in range(per_file):
            if j % 3 == 0:
                # Terms drawn from a disjoint high range: no query overlap.
                idx = sorted(rng.choice(range(VOCAB, VOCAB * 2), 3, replace=False))
            else:
                idx = sorted(rng.choice(range(VOCAB), 4, replace=False))
            rows.append({"indices": [int(i) for i in idx],
                         "values": rng.uniform(0.1, 3.0, len(idx)).astype(np.float32)
                         .tolist()})
            ids.append(f"s{f:02d}_{j:04d}")
        pq.write_table(pa.table({
            "sparse_embedding": pa.array(rows, SPARSE_TYPE),
            "sid": pa.array(ids, pa.string()),
        }), str(cdir / f"f{f:02d}.parquet"))

    qrows = []
    for _ in range(5):
        idx = sorted(rng.choice(range(VOCAB), 6, replace=False))
        qrows.append({"indices": [int(i) for i in idx],
                      "values": rng.uniform(0.5, 2.0, len(idx)).astype(np.float32)
                      .tolist()})
    qpath = tmp_path / "qsparse.parquet"
    pq.write_table(pa.table({
        "sparse_embedding": pa.array(qrows, SPARSE_TYPE),
        "qid": pa.array([f"q{i}" for i in range(len(qrows))]),
    }), str(qpath))
    return cdir, qpath


def _sparse_cfg(cdir, qpath, out, k=10, batch=64):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding",
                            id_column="sid"),
        queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding",
                              id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, sparse_batch_size=batch,
                            tiebreak="id"),
        searches=[SearchSpec(name="sp", k=k, metric="dot", vector_type="sparse")],
    )


@pytest.mark.parametrize("mode", MODES)
def test_sparse_prune_matches_prune_disabled(tmp_path, monkeypatch, mode):
    cdir, qpath = _sparse_corpus(tmp_path, seed=3)

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_sparse_cfg(cdir, qpath, tmp_path / f"sb{mode}")))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    got = _read(run_compute(_sparse_cfg(cdir, qpath, tmp_path / f"sg{mode}")))

    assert state["dead_rows_seen"] > 0, f"sparse/{mode}: prune never fired"
    _assert_same(got, base, f"sparse/{mode}")


def _sparse_sparse_query_corpus(tmp_path, seed=21):
    """A query overlapping only a HANDFUL of corpus rows, with k far larger.
    Its running top-K can never fill, so its threshold stays the -inf sentinel
    for the whole scan — the regime `live_rows` documents as always-live and
    the one a wrong prune would silently truncate."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "crare"
    cdir.mkdir(exist_ok=True)
    rare = VOCAB * 4  # a term only a few corpus rows carry
    for f in range(4):
        rows, ids = [], []
        for j in range(120):
            if f == 0 and j < 3:
                idx = [rare]
            else:
                idx = sorted(rng.choice(range(VOCAB), 4, replace=False))
            rows.append({"indices": [int(i) for i in idx],
                         "values": rng.uniform(0.1, 3.0, len(idx)).astype(np.float32)
                         .tolist()})
            ids.append(f"r{f:02d}_{j:04d}")
        pq.write_table(pa.table({
            "sparse_embedding": pa.array(rows, SPARSE_TYPE),
            "sid": pa.array(ids, pa.string()),
        }), str(cdir / f"f{f:02d}.parquet"))
    qpath = tmp_path / "qrare.parquet"
    pq.write_table(pa.table({
        "sparse_embedding": pa.array(
            [{"indices": [rare], "values": [1.0]},
             {"indices": sorted(int(i) for i in rng.choice(range(VOCAB), 5, False)),
              "values": [1.0] * 5}], SPARSE_TYPE),
        "qid": pa.array(["qrare", "qdense"]),
    }), str(qpath))
    return cdir, qpath


@pytest.mark.parametrize("mode", MODES)
def test_sparse_underfilled_state_keeps_its_rows_live(tmp_path, monkeypatch, mode):
    """k=50 against a query with 3 real candidates: the state never fills, so
    its threshold stays the sentinel and nothing may be pruned away from it."""
    cdir, qpath = _sparse_sparse_query_corpus(tmp_path)

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_sparse_cfg(cdir, qpath, tmp_path / f"ub{mode}", k=50)))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    gpu_contract.install(monkeypatch, mode=mode)
    got = _read(run_compute(_sparse_cfg(cdir, qpath, tmp_path / f"ug{mode}", k=50)))

    rare_hits = dict(zip(got["sp"]["query_id"], got["sp"]["hit_ids"]))["qrare"]
    assert len([h for h in rare_hits if h.startswith("r00_")]) >= 3, \
        "an under-filled query lost its real candidates"
    _assert_same(got, base, f"sparse_underfilled/{mode}")


# ---------------------------------------------------------------------------
# multivector — token-budget tiling, so slice position != file row
# ---------------------------------------------------------------------------


def _mv_array(docs: list[np.ndarray]) -> pa.Array:
    tok = [len(d) for d in docs]
    total = sum(tok)
    flat = (np.concatenate([d.reshape(-1) for d in docs if len(d)])
            if total else np.empty(0, np.float32))
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, total * DIM + 1, DIM, dtype=np.int32)),
        pa.array(flat.astype(np.float32), pa.float32()),
    )
    outer = np.zeros(len(docs) + 1, dtype=np.int32)
    np.cumsum(tok, out=outer[1:])
    return pa.ListArray.from_arrays(pa.array(outer), inner)


def _mv_corpus(tmp_path, n_files=10, per_file=90, seed=0):
    """Random token vectors and RAGGED token counts, so maxsim scores vary and
    the token-budget tiling produces row-misaligned slices."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "cmv"
    cdir.mkdir(exist_ok=True)
    for f in range(n_files):
        docs, ids = [], []
        for j in range(per_file):
            n_tok = 1 + ((j + f) % 4)
            docs.append(rng.normal(size=(n_tok, DIM)).astype(np.float32))
            ids.append(f"m{f:02d}_{j:04d}")
        pq.write_table(pa.table({"multivector_embedding": _mv_array(docs),
                                 "sid": pa.array(ids, pa.string())}),
                       str(cdir / f"f{f:02d}.parquet"))
    qpath = tmp_path / "qmv.parquet"
    qdocs = [rng.normal(size=(3, DIM)).astype(np.float32) for _ in range(5)]
    pq.write_table(pa.table({"multivector_embedding": _mv_array(qdocs),
                             "qid": pa.array([f"q{i}" for i in range(len(qdocs))])}),
                   str(qpath))
    return cdir, qpath


def _mv_cfg(cdir, qpath, out, k=5, budget=None):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir),
                            multivector_column="multivector_embedding",
                            id_column="sid"),
        queries=QueriesConfig(path=str(qpath),
                              multivector_column="multivector_embedding",
                              id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, tiebreak="id",
                            multivector_token_budget=budget),
        searches=[SearchSpec(name="mv", k=k, metric="dot",
                             vector_type="multivector")],
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("budget", [None, 17])
def test_multivector_prune_matches_prune_disabled(tmp_path, monkeypatch,
                                                  mode, budget):
    """A ragged token budget is the tiling most likely to mis-associate a
    live mask with the wrong corpus rows."""
    cdir, qpath = _mv_corpus(tmp_path, seed=5)
    tag = f"{mode}{budget}"

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_mv_cfg(cdir, qpath, tmp_path / f"mb{tag}",
                                     budget=budget)))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    got = _read(run_compute(_mv_cfg(cdir, qpath, tmp_path / f"mg{tag}",
                                    budget=budget)))

    assert state["dead_rows_seen"] > 0, f"mv/{tag}: prune never fired"
    _assert_same(got, base, f"multivector/{tag}")


# ---------------------------------------------------------------------------
# shared Gram — several dense metrics scored from ONE raw Gram
# ---------------------------------------------------------------------------


def _dense_corpus(tmp_path, n_files=6, per_file=200, seed=0, extra_cols=None):
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "cdense"
    cdir.mkdir(exist_ok=True)
    for f in range(n_files):
        v = rng.normal(size=(per_file, DIM)).astype(np.float32)
        cols = {
            "dense_embedding": pa.array(v.tolist(), pa.list_(pa.float32())),
            "sid": pa.array([f"d{f:02d}_{j:04d}" for j in range(per_file)]),
        }
        pq.write_table(pa.table(cols), str(cdir / f"f{f:02d}.parquet"))
    qv = rng.normal(size=(6, DIM)).astype(np.float32)
    qcols = {
        "dense_embedding": pa.array(qv.tolist(), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(len(qv))]),
    }
    if extra_cols:
        qcols.update(extra_cols(len(qv)))
    qpath = tmp_path / "qdense.parquet"
    pq.write_table(pa.table(qcols), str(qpath))
    return cdir, qpath


def _multi_metric_cfg(cdir, qpath, out, k=10):
    """Three metrics over one dense column: `share_gram` turns on when a batch
    group has more than one metric, so all three are scored from one Gram."""
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=64, tiebreak="id"),
        searches=[
            SearchSpec(name="cos", k=k, metric="cosine"),
            SearchSpec(name="dot", k=k, metric="dot"),
            SearchSpec(name="euc", k=k, metric="euclidean"),
        ],
    )


@pytest.mark.parametrize("mode", MODES)
def test_shared_gram_prune_matches_prune_disabled(tmp_path, monkeypatch, mode):
    """Each metric keeps its OWN threshold and live mask off a shared score
    source; a mask leaking across specs would show as a wrong answer in one
    of the three."""
    cdir, qpath = _dense_corpus(tmp_path, seed=7)

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_multi_metric_cfg(cdir, qpath, tmp_path / f"gb{mode}")))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    got = _read(run_compute(_multi_metric_cfg(cdir, qpath, tmp_path / f"gg{mode}")))

    assert state["dead_rows_seen"] > 0, f"gram/{mode}: prune never fired"
    assert set(got) == {"cos", "dot", "euc"}
    _assert_same(got, base, f"shared_gram/{mode}")


# ---------------------------------------------------------------------------
# SearchSpec.rows — `thr`'s height is the SPEC's query count, not the file's
# ---------------------------------------------------------------------------


def _halves_cfg(cdir, qpath, out, k=10):
    """Two specs splitting the query file in half BY DESIGN.

    The halves cover every row on purpose: `RowSelector`'s docstring warns
    that a union which is a strict SUBSET of the file shortens the query
    matrix, changing the matmul's accumulation order and moving scores by
    ~1 ULP. Full coverage keeps the matrix at full height and stays
    bit-exact, so any difference this test sees is the prune's fault.
    """
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid",
                              payload_fields=["half"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=64, tiebreak="id"),
        searches=[
            SearchSpec(name="lo", k=k, metric="cosine",
                       rows=RowSelector(column="half", isin=["lo"])),
            SearchSpec(name="hi", k=k, metric="cosine",
                       rows=RowSelector(column="half", isin=["hi"])),
        ],
    )


@pytest.mark.parametrize("mode", MODES)
def test_spec_rows_prune_matches_prune_disabled(tmp_path, monkeypatch, mode):
    """Specs of one vector_type share a query matrix over the UNION of their
    `rows`, but each spec slices its own rows out AFTER scoring — so a spec's
    `thr` is indexed by ITS query count, not the shared matrix's. An off-by-a-
    subset there would prune against another spec's threshold."""
    cdir, qpath = _dense_corpus(
        tmp_path, seed=13,
        extra_cols=lambda n: {"half": pa.array(
            ["lo" if i < n // 2 else "hi" for i in range(n)])})

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    base = _read(run_compute(_halves_cfg(cdir, qpath, tmp_path / f"rb{mode}")))

    monkeypatch.delenv("NOVA_BF_NO_PRUNE", raising=False)
    state = gpu_contract.install(monkeypatch, mode=mode)
    got = _read(run_compute(_halves_cfg(cdir, qpath, tmp_path / f"rg{mode}")))

    assert state["dead_rows_seen"] > 0, f"rows/{mode}: prune never fired"
    assert set(got) == {"lo", "hi"}
    for name in ("lo", "hi"):
        assert got[name]["query_id"], f"{name} selected no rows"
    _assert_same(got, base, f"spec_rows/{mode}")


def test_spec_rows_halves_cover_every_query(tmp_path):
    """Guard the guard: if the halves ever stopped covering the file, the
    comparison above would be measuring a ~1 ULP matrix-height artifact
    rather than the prune."""
    cdir, qpath = _dense_corpus(
        tmp_path, seed=13,
        extra_cols=lambda n: {"half": pa.array(
            ["lo" if i < n // 2 else "hi" for i in range(n)])})
    got = _read(run_compute(_halves_cfg(cdir, qpath, tmp_path / "rcov")))
    covered = set(got["lo"]["query_id"]) | set(got["hi"]["query_id"])
    assert covered == set(pq.read_table(qpath).to_pydict()["qid"])
