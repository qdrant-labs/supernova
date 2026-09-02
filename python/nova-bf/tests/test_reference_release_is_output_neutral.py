"""The reader's reference releases must not change a single output byte.

`reader()` sets two groups of locals to `None` — the raw inputs after their
last use (`table`, `masks`, `mask`) and the hand-off locals after `fq.put`
(`arrs`, `batches`, `keeps`, `leaf_arrays`, `ids`, ...). Both are pure
lifetime changes measured at roughly -19 GiB of peak host RSS combined, and
both are only safe because Arrow buffers are refcounted: whatever is still
referenced downstream keeps exactly the buffers it needs alive.

That safety argument is language-level, not empirical, so this pins it
empirically. The statements are inline with nothing to monkeypatch, so the
only honest A/B is a second package tree with the blocks stripped. If a future
edit reformats them, the anchors below stop matching and this test FAILS with
that message — it never silently strips nothing and passes vacuously.
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

# Exact source of each release, and how many times each must appear.
RELEASE_BLOCKS = [
    "                table = masks = mask = None\n",
    "                arrs = batches = batch_orig_rows = raw_stats = None\n"
    "                keeps = leaf_arrays = ids = b = union = None\n",
]

SPARSE_TYPE = pa.struct([
    pa.field("indices", pa.list_(pa.uint32())),
    pa.field("values", pa.list_(pa.float32())),
])
DIM, VOCAB = 6, 24
WORDS = ["physics", "dna", "pottery", "baking"]


def _make_data(root: pathlib.Path) -> pathlib.Path:
    """Small, but covering every consumer of the released references: dense
    (`arrs`), sparse (`arrs` tuple), an `id_column` resolved from `ids` (a view
    into the dropped `table`), and a CPU-fallback text filter (the unpacked
    `masks`)."""
    rng = np.random.default_rng(5)
    cdir = root / "c"
    cdir.mkdir(parents=True)
    for fi in range(3):
        n = 18 + fi * 5                      # ragged
        rows = []
        for j in range(n):
            k = 2 + (j % 3)
            idx = sorted(int(x) for x in rng.choice(VOCAB, k, replace=False))
            rows.append({"indices": idx,
                         "values": rng.uniform(0.1, 2.0, k).astype(np.float32).tolist()})
        pq.write_table(pa.table({
            "dense_embedding": pa.array(
                rng.normal(size=(n, DIM)).astype(np.float32).tolist(),
                pa.list_(pa.float32())),
            "sparse_embedding": pa.array(rows, SPARSE_TYPE),
            "sid": pa.array([f"f{fi}r{j}" for j in range(n)]),
            "text": pa.array([f"{WORDS[(fi + j) % 4]} doc {fi}{j}" for j in range(n)]),
            "tenant": pa.array(["A" if j % 2 else "B" for j in range(n)]),
        }), str(cdir / f"f{fi}.parquet"))

    nq = 6
    qs = []
    for _ in range(nq):
        idx = sorted(int(x) for x in rng.choice(VOCAB, 5, replace=False))
        qs.append({"indices": idx,
                   "values": rng.uniform(0.2, 1.5, 5).astype(np.float32).tolist()})
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.normal(size=(nq, DIM)).astype(np.float32).tolist(),
            pa.list_(pa.float32())),
        "sparse_embedding": pa.array(qs, SPARSE_TYPE),
        "qid": pa.array([f"q{i}" for i in range(nq)]),
        "kw": pa.array([WORDS[i % 4] for i in range(nq)]),
        "tenant_want": pa.array(["A" if i % 2 else "B" for i in range(nq)]),
    }), str(root / "q.parquet"))
    return cdir


_DRIVER = textwrap.dedent("""
    import sys
    from nova_bf.compute import run_compute
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
        ParamsConfig, QueriesConfig, SearchSpec,
    )
    root, out = sys.argv[1], sys.argv[2]
    run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/c", id_column="sid",
                            dense_column="dense_embedding",
                            sparse_column="sparse_embedding"),
        queries=QueriesConfig(path=f"{root}/q.parquet", id_column="qid",
                              dense_column="dense_embedding",
                              sparse_column="sparse_embedding",
                              payload_fields=["kw", "tenant_want"]),
        output=OutputConfig(path=out),
        params=ParamsConfig(io_workers=3, tiebreak="id"),
        searches=[
            SearchSpec(name="dense_cos", vector_type="dense", metric="cosine", k=5),
            SearchSpec(name="sparse_dot", vector_type="sparse", metric="dot", k=5),
            SearchSpec(name="filtered", vector_type="dense", metric="cosine", k=5,
                       filter=Filter(must=[FilterCondition(
                           field="text", match_text_from_query="kw")])),
            SearchSpec(name="tenant", vector_type="dense", metric="dot", k=5,
                       filter=Filter(must=[FilterCondition(
                           field="tenant", match_from_query="tenant_want")])),
        ],
    ))
""")


def _run(root, out, pkg_root=None):
    env = dict(os.environ)
    env.pop("NOVA_BF_NO_PRUNE", None)
    if pkg_root is not None:
        env["PYTHONPATH"] = str(pkg_root)
    r = subprocess.run([sys.executable, "-c", _DRIVER, str(root), str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, (
        f"run failed (pkg_root={pkg_root}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def test_stripping_the_reference_releases_changes_no_output_byte(tmp_path):
    src = pathlib.Path(nova_bf.__file__).resolve().parent.parent
    stripped = tmp_path / "pre_src"
    shutil.copytree(src, stripped)
    cp = stripped / "nova_bf" / "compute.py"
    text = cp.read_text()
    for block in RELEASE_BLOCKS:
        assert text.count(block) == 1, (
            "release block not found exactly once in compute.py — it was "
            "reformatted or removed, so this test can no longer A/B it:\n"
            f"{block!r}")
        text = text.replace(block, "")
    cp.write_text(text)

    root = tmp_path / "data"
    _make_data(root)
    _run(root, tmp_path / "post")                    # current package
    _run(root, tmp_path / "pre", pkg_root=stripped)  # releases stripped

    names = sorted(p.name for p in (tmp_path / "post").rglob("*.parquet"))
    assert len(names) == 4, f"expected one output per search, got {names}"
    for name in names:
        a = pq.read_table(next((tmp_path / "post").rglob(name)))
        b = pq.read_table(next((tmp_path / "pre").rglob(name)))
        assert a.schema.equals(b.schema, check_metadata=False), f"{name}: schema"
        if not a.equals(b):
            da, db = a.to_pydict(), b.to_pydict()
            diff = [c for c in da if da[c] != db[c]]
            pytest.fail(f"{name}: releasing references changed {diff}")
