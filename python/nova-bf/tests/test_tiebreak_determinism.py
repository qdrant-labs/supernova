"""The property the tie-break exists for, checked end to end.

Every corpus here is built so that MANY candidates score EXACTLY equal — most
of them share one vector — which makes the tie-break, not the score, decide the
whole result. Under the baseline that made the answer depend on
`dense_batch_size`, on `--num-jobs`, and on how candidates happened to be
grouped into slices; here it must not.

The guarantee is conditional on the scores being bit-identical, so every search
uses `dot` on values that are exactly representable in float32 — no re-tiling
can move them, which isolates the tie-break as the only thing under test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import run_merge

VEC = [1.0, 0.0, 0.0, 0.0]


def _corpus(tmp, per_file_ids, dtype=pa.string(), payload=None):
    """Every row carries the SAME vector, so every score ties exactly."""
    cdir = tmp / "corpus"
    cdir.mkdir(exist_ok=True)
    for fi, ids in enumerate(per_file_ids):
        data = {
            "dense_embedding": pa.array([VEC] * len(ids), pa.list_(pa.float32())),
            "sid": pa.array(ids, dtype),
        }
        if payload is not None:
            data["lang"] = pa.array([payload(i) for i in ids])
        pq.write_table(pa.table(data), str(cdir / f"f{fi}.parquet"))
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([VEC], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp / "q.parquet"),
    )
    return cdir


def _cfg(cdir, tmp, out, k, tiebreak, batch=None, filt=None):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, tiebreak=tiebreak, dense_batch_size=batch),
        searches=[SearchSpec(name="t", k=k, metric="dot", filter=filt)],
    )


def _run(cdir, tmp, tag, k, tiebreak, n_jobs=None, batch=None, filt=None):
    out = tmp / tag
    out.mkdir(exist_ok=True)
    cfg = _cfg(cdir, tmp, out, k, tiebreak, batch, filt)
    if n_jobs is None:
        t = pq.read_table(run_compute(cfg)["t"]).to_pydict()
    else:
        for r in range(n_jobs):
            run_compute(cfg, num_jobs=n_jobs, job_rank=r)
        t = pq.read_table(run_merge(cfg)["t"]).to_pydict()
    return dict(zip(t["query_id"], t["hit_ids"]))["q0"]


# --------------------------------------------------------------------------
# the rules mean what they say
# --------------------------------------------------------------------------

# Ids deliberately ANTI-correlated with corpus order, so the two rules disagree
# and neither can pass by accident.
ANTI = [["z9", "z8"], ["z7", "z6"], ["z5", "z4"]]
ANTI_FLAT = [i for f in ANTI for i in f]


def test_id_mode_takes_the_lowest_id(tmp_path):
    cdir = _corpus(tmp_path, ANTI)
    assert _run(cdir, tmp_path, "id", 3, "id") == ["z4", "z5", "z6"]


def test_ordinal_mode_takes_the_earliest_corpus_row(tmp_path):
    cdir = _corpus(tmp_path, ANTI)
    assert _run(cdir, tmp_path, "ord", 3, "ordinal") == ANTI_FLAT[:3]


def test_the_two_rules_really_disagree(tmp_path):
    """Guards the tests above from becoming vacuous."""
    cdir = _corpus(tmp_path, ANTI)
    assert _run(cdir, tmp_path, "a", 3, "id") != _run(cdir, tmp_path, "b", 3, "ordinal")


# --------------------------------------------------------------------------
# invariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("n_jobs", [None, 1, 2, 3, 5])
def test_the_answer_does_not_move_with_the_shard_count(tmp_path, tiebreak, n_jobs):
    """`--num-jobs` above the file count is included: a worker that draws no
    files at all still has to write a partial with the right schema."""
    cdir = _corpus(tmp_path, ANTI)
    expected = sorted(ANTI_FLAT)[:3] if tiebreak == "id" else ANTI_FLAT[:3]
    assert _run(cdir, tmp_path, f"s{tiebreak}{n_jobs}", 3, tiebreak, n_jobs) == expected


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("batch", [None, 1, 2, 3, 4, 7])
def test_the_answer_does_not_move_with_the_batch_size(tmp_path, tiebreak, batch):
    """Slicing decides which candidates meet in a pre-top-K — and that pre-top-K
    DISCARDS PERMANENTLY, so a batch-dependent tie would be unrecoverable."""
    cdir = _corpus(tmp_path, ANTI)
    expected = sorted(ANTI_FLAT)[:3] if tiebreak == "id" else ANTI_FLAT[:3]
    assert _run(
        cdir, tmp_path, f"b{tiebreak}{batch}", 3, tiebreak, batch=batch
    ) == expected


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_every_shard_and_batch_combination_agrees(tmp_path, tiebreak):
    cdir = _corpus(tmp_path, ANTI)
    answers = {
        tuple(_run(cdir, tmp_path, f"x{tiebreak}{nj}_{bs}", 4, tiebreak, nj, bs))
        for nj in (None, 2, 3)
        for bs in (None, 1, 3)
    }
    assert len(answers) == 1, f"{len(answers)} distinct answers: {answers}"


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_k_larger_than_the_corpus_keeps_every_row_once(tmp_path, tiebreak):
    cdir = _corpus(tmp_path, ANTI)
    got = _run(cdir, tmp_path, f"big{tiebreak}", 50, tiebreak, n_jobs=3)
    assert sorted(got) == sorted(ANTI_FLAT)
    assert len(got) == len(ANTI_FLAT), "no duplicates across the reduce"


# --------------------------------------------------------------------------
# the id rule over the shapes a projection could not have keyed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [None, 2])
@pytest.mark.parametrize("ids,dtype,expect", [
    # numeric: "10" would precede "9" as text
    ([[9, 10], [100, 2]], pa.int64(), ["2", "9", "10"]),
    # far from zero: a value-into-32-bits projection collapses these
    ([[1_700_000_000_000_000_003, 1_700_000_000_000_000_001],
      [1_700_000_000_000_000_002]], pa.int64(),
     ["1700000000000000001", "1700000000000000002", "1700000000000000003"]),
    # negative
    ([[-1, -500], [3, -499]], pa.int64(), ["-500", "-499", "-1"]),
    # uint64 above 2**63
    ([[2**64 - 1, 5], [2**63]], pa.uint64(), ["5", str(2**63), str(2**64 - 1)]),
    # long shared head, entropy only after it
    ([["<urn:uuid:0000000c>", "<urn:uuid:0000000a>"], ["<urn:uuid:0000000b>"]],
     pa.string(),
     ["<urn:uuid:0000000a>", "<urn:uuid:0000000b>", "<urn:uuid:0000000c>"]),
    # zero-padded, entropy in the LAST byte
    ([["id00000009", "id00000007"], ["id00000008"]], pa.string(),
     ["id00000007", "id00000008", "id00000009"]),
])
def test_id_shapes_that_defeat_a_prefix_or_window_key(tmp_path, ids, dtype, expect, n_jobs):
    cdir = _corpus(tmp_path, ids, dtype=dtype)
    assert _run(cdir, tmp_path, f"sh{n_jobs}", 3, "id", n_jobs) == expect


@pytest.mark.parametrize("n_jobs", [None, 2])
def test_duplicate_ids_fall_back_to_corpus_position(tmp_path, n_jobs):
    """Two rows with the same id denote the same point, but WHICH survives must
    still be fixed rather than arbitrary."""
    cdir = _corpus(tmp_path, [["b", "a"], ["a", "b"]])
    assert _run(cdir, tmp_path, f"dup{n_jobs}", 2, "id", n_jobs) == ["a", "a"]


# --------------------------------------------------------------------------
# the paths where rows are not simply "the whole file in order"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("n_jobs", [None, 2])
def test_a_filter_does_not_shift_the_ordinals(tmp_path, tiebreak, n_jobs):
    """A filtered batch is COMPACTED to surviving rows, so a row's position in
    the batch is no longer its position in the file. The ordinal must follow the
    file, not the batch — otherwise the same row keys differently depending on
    which filter ran."""
    cdir = _corpus(
        tmp_path, ANTI, payload=lambda i: "eng" if i in ("z9", "z7", "z5") else "fra"
    )
    filt = Filter(must=[FilterCondition(field="lang", match="eng")])
    got = _run(cdir, tmp_path, f"f{tiebreak}{n_jobs}", 2, tiebreak, n_jobs, filt=filt)
    assert got == (["z5", "z7"] if tiebreak == "id" else ["z9", "z7"])


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_empty_files_do_not_shift_the_ordinals(tmp_path, tiebreak):
    """An empty shard contributes no rows but must still advance nothing — and
    must not break the coalesce path, which concatenates several files' rows."""
    cdir = _corpus(tmp_path, [["z9", "z8"], [], ["z7"], [], ["z6"]])
    expected = ["z6", "z7"] if tiebreak == "id" else ["z9", "z8"]
    assert _run(cdir, tmp_path, f"e{tiebreak}", 2, tiebreak, n_jobs=2) == expected


def test_a_null_id_is_still_an_ordinary_row_under_ordinal(tmp_path):
    """`ordinal` never looks at the id, so a null one is just a corpus row —
    here the second, which beats the third row's `a`. It surfaces as the literal
    string `"None"` because that is how `resolve_id` renders it, which is
    pre-existing behaviour and orthogonal to the tie-break."""
    cdir = _corpus(tmp_path, [["b", None], ["a"]])
    assert _run(cdir, tmp_path, "nord", 2, "ordinal", n_jobs=2) == ["b", "None"]


@pytest.mark.parametrize("n_jobs", [None, 2])
def test_a_null_id_is_rejected_under_id(tmp_path, n_jobs):
    """It cannot be ordered consistently: within a worker a null sorts last, but
    `hit_ids` render it as `"None"`, which `merge` sorts BEFORE every lowercase
    id. Left alone the two sides of the reduce disagree and the winner moves
    with `--num-jobs` — the exact failure the tie-break exists to remove."""
    cdir = _corpus(tmp_path, [["b", None], ["a"]])
    with pytest.raises(ValueError, match="null value"):
        _run(cdir, tmp_path, f"nid{n_jobs}", 2, "id", n_jobs)


# --------------------------------------------------------------------------
# fuzz: an independent oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_against_an_independent_oracle(tmp_path, seed):
    """Ranks the whole corpus in plain Python by the rule's definition and
    compares. Scores are drawn from a tiny set so ties are dense."""
    rng = np.random.default_rng(seed)
    n_files = int(rng.integers(1, 5))
    sizes = [int(rng.integers(0, 7)) for _ in range(n_files)]
    ids, groups = [], []
    for n in sizes:
        g = [f"k{int(rng.integers(0, 40)):03d}" for _ in range(n)]
        groups.append(g)
        ids.extend(g)
    if not ids:
        pytest.skip("empty corpus")
    scores = [float(rng.integers(0, 3)) for _ in ids]

    cdir = tmp_path / "corpus"
    cdir.mkdir()
    off = 0
    for fi, g in enumerate(groups):
        vecs = [[scores[off + j], 0.0, 0.0, 0.0] for j in range(len(g))]
        off += len(g)
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array(vecs, pa.list_(pa.float32())),
                "sid": pa.array(g, pa.string()),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )

    k = int(rng.integers(1, len(ids) + 2))
    for tiebreak in ("ordinal", "id"):
        key = (
            (lambda t: (-t[1], t[0], t[2])) if tiebreak == "id"
            else (lambda t: (-t[1], t[2]))
        )
        want = [t[0] for t in sorted(zip(ids, scores, range(len(ids))), key=key)][:k]
        for n_jobs in (None, 2, 3):
            got = _run(cdir, tmp_path, f"o{seed}{tiebreak}{n_jobs}", k, tiebreak, n_jobs)
            assert got == want, f"{tiebreak} n_jobs={n_jobs}"


# --------------------------------------------------------------------------
# ordering of the emitted hits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_jobs", [None, 3])
def test_hits_come_out_in_tie_break_order(tmp_path, n_jobs):
    """Every `topk` on the way in runs `sorted=False`, so the ONLY thing that
    orders the output is the decode-time sort. If that ever regressed, hit_ids
    would come out in whatever order the last fold happened to leave — which no
    test of set membership would catch."""
    cdir = _corpus(tmp_path, ANTI)
    got = _run(cdir, tmp_path, f"ord{n_jobs}", 6, "id", n_jobs)
    # every row scores identically here, so the tie-break alone fixes the order
    assert got == sorted(ANTI_FLAT), got


def test_scores_are_emitted_descending(tmp_path):
    """Distinct scores: the decode sort must put them in descending order."""
    cdir = tmp_path / "c"
    cdir.mkdir()
    ids = [f"s{i}" for i in range(6)]
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(
                [[float(i), 0.0, 0.0, 0.0] for i in range(6)], pa.list_(pa.float32())
            ),
            "sid": pa.array(ids),
        }),
        str(cdir / "f0.parquet"),
    )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([[1.0, 0.0, 0.0, 0.0]], pa.list_(pa.float32())),
            "qid": pa.array(["q0"]),
        }),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / "desc"
    out.mkdir()
    cfg = _cfg(cdir, tmp_path, out, 6, "ordinal")
    t = pq.read_table(run_compute(cfg)["t"]).to_pydict()
    scores = t["hit_scores"][0]
    assert scores == sorted(scores, reverse=True), scores
    assert t["hit_ids"][0] == ["s5", "s4", "s3", "s2", "s1", "s0"]


def test_the_decode_sort_is_exact_however_it_is_chunked(tmp_path, monkeypatch):
    """The decode sort is chunked over query rows to bound its transient (it is
    ~3x the top-K state live at once, and it scales with QUERY count — 100k
    queries at k=1000 would be ~3 GiB unchunked). Per-row sorts are independent,
    so chunking must be exact, not an approximation. Forced to one row per chunk
    here so every boundary is crossed."""
    import nova_bf.compute as cp

    cdir = _corpus(tmp_path, ANTI)
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array([VEC] * 5, pa.list_(pa.float32())),
            "qid": pa.array([f"q{i}" for i in range(5)]),
        }),
        str(tmp_path / "q.parquet"),
    )
    whole = _run(cdir, tmp_path, "whole", 4, "id")
    monkeypatch.setattr(cp, "DECODE_CHUNK_SLOTS", 4)   # -> 1 row per chunk at k=4
    chunked = _run(cdir, tmp_path, "chunked", 4, "id")
    assert whole == chunked, "chunking changed the decoded output"
