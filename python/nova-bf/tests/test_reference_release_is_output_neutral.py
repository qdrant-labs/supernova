"""The reader's reference releases must not change a single output byte.

`reader()` carries three pure-lifetime optimisations: releasing the raw
inputs after their last use (`table`, `masks`, `mask`), releasing the hand-off
locals after `fq.put` (`arrs`, `batches`, `keeps`, ...), and narrowing the
table to just the filter columns before `evaluate`, its peak phase. Together
they are worth roughly -20 GiB of peak host RSS, and all three are only safe
because Arrow buffers are refcounted: whatever is still referenced downstream
keeps exactly the buffers it needs alive.

That safety argument is language-level, not empirical, so this pins it
empirically. The statements are inline with nothing to monkeypatch, so the
only honest A/B is a second package tree with the blocks stripped. If a future
edit reformats them, the anchors below stop matching and this test FAILS with
that message — it never silently strips nothing and passes vacuously.

The fixture deliberately covers every post-narrowing consumer of `table`, not
just the easy one:

  * `evaluate` (CPU fallback) via `match_text_from_query` / `match_from_query`
    / a static `range`;
  * `_corpus_leaf_array` (the GPU-leaf front) via `range_from_query` — on a
    float column AND on a date column, the latter being the one column whose
    identity is mutated between read and narrowing by
    `convert_table_date_columns` (rfc3339 -> int64 epoch);
  * an explicit empty `Filter()`, and separately a config where NO spec has a
    filter at all, so `filter_cols == []` and the guard's own branch runs;
  * multivector, filtered and unfiltered.

It also runs both corpus dense dtypes, because the narrowing's safety argument
is REFCOUNTING, not copying, and which one happens depends on the dtype: fp16
is widened to fp32 by `dense_to_2d` (a real copy), while fp32 is a zero-copy
VIEW into the Arrow buffer. Production is fp16; fp32 is the case where a
mistaken in-place mutation of `arrs` would corrupt a live Arrow buffer.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

import nova_bf

# Reverse patches that undo each lifetime optimisation, applied to a COPY of
# compute.py to build the "before" package. Each `(new, old)` pair must match
# exactly once; a mismatch fails the test rather than silently reverting
# nothing. Order matters only in that each must be independently unique.
REVERSE_PATCHES = [
    # 1. release the raw inputs after their last use
    ("                table = masks = mask = None\n", ""),
    # 2. release the hand-off locals after `fq.put`
    ("                arrs = batches = batch_orig_rows = raw_stats = None\n"
     "                keeps = leaf_arrays = ids = b = union = None\n", ""),
    # 3. narrow the table to the filter columns before `evaluate`
    ("                n_rows_file = len(table)\n"
     "                if filter_cols:\n"
     "                    table = table.select(filter_cols)\n", ""),
    ("                n_rows = n_rows_file\n",
     "                n_rows = len(table)\n"),
]

SPARSE_TYPE = pa.struct([
    pa.field("indices", pa.list_(pa.uint32())),
    pa.field("values", pa.list_(pa.float32())),
])
DIM, VOCAB = 6, 24
WORDS = ["physics", "dna", "pottery", "baking"]


def _make_data(root: pathlib.Path, dense_dtype: str = "float32") -> pathlib.Path:
    """Small, but covering every consumer of the released references: dense
    (`arrs`), sparse (`arrs` tuple), multivector (`arrs` ragged), an
    `id_column` resolved from `ids` (a view into the dropped `table`), and
    filter columns reached by BOTH post-narrowing fronts — `evaluate` (text /
    match / static range) and `_corpus_leaf_array` (`range_from_query`, on a
    float column and on a date column).

    `dense_dtype` selects the corpus dense storage type: "float16" is
    production and makes `dense_to_2d` copy; "float32" makes it a zero-copy
    view. Queries stay fp32 either way, as in production.
    """
    rng = np.random.default_rng(5)
    pa_dense = pa.float16() if dense_dtype == "float16" else pa.float32()
    np_dense = np.float16 if dense_dtype == "float16" else np.float32
    cdir = root / "c"
    cdir.mkdir(parents=True)
    for fi in range(3):
        n = 18 + fi * 5                      # ragged
        rows, mv = [], []
        for j in range(n):
            k = 2 + (j % 3)
            idx = sorted(int(x) for x in rng.choice(VOCAB, k, replace=False))
            rows.append({"indices": idx,
                         "values": rng.uniform(0.1, 2.0, k).astype(np.float32).tolist()})
            mv.append(rng.normal(size=(1 + (j % 4), DIM)).astype(np.float32).tolist())
        pq.write_table(pa.table({
            "dense_embedding": pa.array(
                rng.normal(size=(n, DIM)).astype(np_dense).tolist(),
                pa.list_(pa_dense)),
            "sparse_embedding": pa.array(rows, SPARSE_TYPE),
            "multivector_embedding": pa.array(mv, pa.list_(pa.list_(pa.float32()))),
            "sid": pa.array([f"f{fi}r{j}" for j in range(n)]),
            "text": pa.array([f"{WORDS[(fi + j) % 4]} doc {fi}{j}" for j in range(n)]),
            "tenant": pa.array(["A" if j % 2 else "B" for j in range(n)]),
            # `score` feeds both a static `range` and a `range_from_query`;
            # `published_at` is the rfc3339 column that becomes int64 epoch at
            # read time and must survive `select` in its CONVERTED form.
            "score": pa.array(rng.uniform(0, 100, n).astype(np.float64)),
            "published_at": pa.array(
                [f"20{20 + ((fi + j) % 5):02d}-0{1 + (j % 9)}-15T00:00:00Z"
                 for j in range(n)]),
        }), str(cdir / f"f{fi}.parquet"))

    nq = 6
    qs, qmv = [], []
    for _ in range(nq):
        idx = sorted(int(x) for x in rng.choice(VOCAB, 5, replace=False))
        qs.append({"indices": idx,
                   "values": rng.uniform(0.2, 1.5, 5).astype(np.float32).tolist()})
        qmv.append(rng.normal(size=(2, DIM)).astype(np.float32).tolist())
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.normal(size=(nq, DIM)).astype(np.float32).tolist(),
            pa.list_(pa.float32())),
        "sparse_embedding": pa.array(qs, SPARSE_TYPE),
        "multivector_embedding": pa.array(qmv, pa.list_(pa.list_(pa.float32()))),
        "qid": pa.array([f"q{i}" for i in range(nq)]),
        "kw": pa.array([WORDS[i % 4] for i in range(nq)]),
        "tenant_want": pa.array(["A" if i % 2 else "B" for i in range(nq)]),
        "min_score": pa.array((np.arange(nq) * 13.0)),
        "after": pa.array([f"20{20 + (i % 4):02d}-01-01T00:00:00Z" for i in range(nq)]),
    }), str(root / "q.parquet"))
    return cdir


_DRIVER = textwrap.dedent("""
    import sys
    from nova_bf.compute import run_compute
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
        ParamsConfig, QueriesConfig, RangeCondition, RangeFromQuery, SearchSpec,
    )
    root, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]

    # Unfiltered specs only. In "nofilter" mode these are the WHOLE config, so
    # `filter_cols == []` and the narrowing guard's own branch is exercised —
    # the one path the original fixture never reached.
    plain = [
        SearchSpec(name="dense_cos", vector_type="dense", metric="cosine", k=5),
        SearchSpec(name="sparse_dot", vector_type="sparse", metric="dot", k=5),
        SearchSpec(name="mv", vector_type="multivector", metric="dot", k=5),
    ]
    filtered = [
        # explicit-but-empty filter: contributes no fields to `filter_cols`
        SearchSpec(name="empty_filter", vector_type="dense", metric="dot", k=5,
                   filter=Filter()),
        # -> evaluate (CPU fallback)
        SearchSpec(name="text_pq", vector_type="dense", metric="cosine", k=5,
                   filter=Filter(must=[FilterCondition(
                       field="text", match_text_from_query="kw")])),
        SearchSpec(name="tenant_pq", vector_type="dense", metric="dot", k=5,
                   filter=Filter(must=[FilterCondition(
                       field="tenant", match_from_query="tenant_want")])),
        # -> _corpus_leaf_array (GPU-leaf front), incl. the date column that
        #    convert_table_date_columns rewrote to int64 before the narrowing
        SearchSpec(name="range_pq", vector_type="sparse", metric="dot", k=5,
                   filter=Filter(must=[
                       FilterCondition(field="score",
                                       range_from_query=RangeFromQuery(gte="min_score")),
                       FilterCondition(field="published_at",
                                       range_from_query=RangeFromQuery(gte="after"))])),
        SearchSpec(name="static_range", vector_type="dense", metric="dot", k=5,
                   filter=Filter(must=[FilterCondition(
                       field="score", range=RangeCondition(gte=20.0, lt=90.0))])),
        SearchSpec(name="mv_filtered", vector_type="multivector", metric="dot", k=5,
                   filter=Filter(must=[FilterCondition(field="tenant", match="A")])),
    ]
    run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/c", id_column="sid",
                            dense_column="dense_embedding",
                            sparse_column="sparse_embedding",
                            multivector_column="multivector_embedding",
                            date_fields=["published_at"]),
        queries=QueriesConfig(path=f"{root}/q.parquet", id_column="qid",
                              dense_column="dense_embedding",
                              sparse_column="sparse_embedding",
                              multivector_column="multivector_embedding",
                              payload_fields=["kw", "tenant_want", "min_score", "after"],
                              date_fields=["after"]),
        output=OutputConfig(path=out),
        params=ParamsConfig(io_workers=3, tiebreak="id"),
        searches=plain + (filtered if mode == "full" else []),
    ))
""")

# mode -> how many output tables that config must produce
MODES = {"full": 9, "nofilter": 3}


def _run(root, out, mode, pkg_root=None):
    env = dict(os.environ)
    env.pop("NOVA_BF_NO_PRUNE", None)
    if pkg_root is not None:
        env["PYTHONPATH"] = str(pkg_root)
    r = subprocess.run([sys.executable, "-c", _DRIVER, str(root), str(out), mode],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, (
        f"run failed (mode={mode}, pkg_root={pkg_root}):\n"
        f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


@pytest.mark.parametrize("dense_dtype", ["float32", "float16"])
def test_stripping_the_reference_releases_changes_no_output_byte(tmp_path, dense_dtype):
    src = pathlib.Path(nova_bf.__file__).resolve().parent.parent
    stripped = tmp_path / "pre_src"
    shutil.copytree(src, stripped)
    cp = stripped / "nova_bf" / "compute.py"
    text = cp.read_text()
    for new_src, old_src in REVERSE_PATCHES:
        assert text.count(new_src) == 1, (
            "optimisation not found exactly once in compute.py — it was "
            "reformatted or removed, so this test can no longer A/B it:\n"
            f"{new_src!r}")
        text = text.replace(new_src, old_src)
    cp.write_text(text)

    root = tmp_path / "data"
    _make_data(root, dense_dtype)

    for mode, n_out in MODES.items():
        post, pre = tmp_path / f"post_{mode}", tmp_path / f"pre_{mode}"
        _run(root, post, mode)                        # current package
        _run(root, pre, mode, pkg_root=stripped)      # releases stripped

        names = sorted(p.name for p in post.rglob("*.parquet"))
        assert len(names) == n_out, (
            f"mode={mode}: expected one output per search, got {names}")
        for name in names:
            a = pq.read_table(next(post.rglob(name)))
            b = pq.read_table(next(pre.rglob(name)))
            assert a.schema.equals(b.schema, check_metadata=False), \
                f"mode={mode} {name}: schema"
            if not a.equals(b):
                da, db = a.to_pydict(), b.to_pydict()
                diff = [c for c in da if da[c] != db[c]]
                pytest.fail(
                    f"mode={mode} {name} (dense={dense_dtype}): "
                    f"releasing references changed {diff}")
