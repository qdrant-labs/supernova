"""The run manifest each phase writes next to its outputs (see manifest.py),
and the run fingerprint that decides whether partials may be merged at all.

The parquet's schema metadata says what the ground truth IS; the manifest says
what the RUN was — ranks, files, timings, hardware, the filter each search
actually used. These cover the parts a consumer parses: that a manifest lands
for every phase (one per rank when sharded), that it names the outputs it
describes, and that the search-level detail survives from config to JSON.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    RowSelector,
    SearchSpec,
)
from nova_bf.io import ParquetFile
from nova_bf import manifest as run_manifest
from nova_bf.manifest import corpus_fingerprint, gpu_peak
from nova_bf.merge import run_merge

DIM, K = 8, 3


def _write(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("manifest")
    cdir = tmp / "corpus"
    cdir.mkdir()
    g = 0
    for fi, n in enumerate((5, 4)):
        rows = rng.standard_normal((n, DIM)).astype(np.float32)
        _write(
            cdir / f"f{fi}.parquet", rows,
            id=[f"c{g + r}" for r in range(n)],
            language=["eng" if (g + r) % 2 == 0 else "fra" for r in range(n)],
        )
        g += n
    qpath = tmp / "queries.parquet"
    _write(qpath, rng.standard_normal((4, DIM)).astype(np.float32), qid=[f"q{i}" for i in range(4)])
    return {"cdir": str(cdir), "qpath": str(qpath)}


def _cfg(ds, out) -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2),
        searches=[
            SearchSpec(name="plain", metric="dot", k=K),
            SearchSpec(
                name="eng", metric="dot", k=K,
                filter=Filter(must=[FilterCondition(field="language", match="eng")]),
            ),
        ],
    )


def _read(path) -> dict:
    return json.loads(path.read_text())


def test_single_node_manifest_describes_the_run(ds, tmp_path):
    out = tmp_path / "single"
    out.mkdir()
    cfg = _cfg(ds, out)
    results = run_compute(cfg)

    doc = _read(out / "_bf_manifest_queries_compute.json")
    assert doc["phase"] == "compute"
    assert doc["destination"] == str(out)
    assert doc["source"]["corpus"]["path"] == ds["cdir"]
    assert doc["source"]["queries"]["path"] == ds["qpath"]
    assert doc["tiebreak"] == "ordinal"
    # CLI/param overrides land as what RAN, not what the YAML said.
    assert doc["params"]["io_workers"] == 2
    assert doc["sharding"] == {
        "num_jobs": None, "job_rank": None,
        "corpus_files_total": 2, "corpus_files_this_worker": 2,
        "max_files": None, "partial_slice": False,
    }
    # Which code produced it, and where it ran — an orphan manifest can't be
    # traced back to a rank's log.
    assert doc["code"]["python_version"]
    assert doc["code"]["numpy_version"] and doc["code"]["pyarrow_version"]
    if "git_commit" in doc["code"]:  # absent for a wheel install / no git
        assert len(doc["code"]["git_commit"]) == 40
        assert doc["code"]["git_describe"] and doc["code"]["git_branch"]
        assert isinstance(doc["code"]["git_dirty"], bool)
    if "supernova_version" in doc["code"]:  # absent outside a repo checkout
        assert doc["code"]["supernova_version"].count(".") == 2
    assert doc["job"]["hostname"] and doc["job"]["pid"] > 0

    # The corpus as an ORDERED file list: hit ids are derived from the global
    # file index, so the order is part of the id scheme.
    fp = doc["source"]["corpus"]["fingerprint"]
    assert fp["files"] == 2
    assert len(fp["sha256"]) == 64
    assert fp["first"].endswith("f0.parquet") and fp["last"].endswith("f1.parquet")

    # `params` is what RAN: sizes resolved at runtime, not the configured nulls.
    assert doc["params"]["batch_size_by_vector_type"] == {"dense": None}

    assert doc["counts"]["queries_in_file"] == 4
    assert doc["counts"]["queries_searched"] == 4  # no search subsets its rows
    assert doc["counts"]["corpus_rows_scanned"] == 9
    assert doc["timing"]["elapsed_seconds"] >= 0

    # Every search it computed, with the filter that defined it.
    by_name = {s["name"]: s for s in doc["searches"]}
    assert set(by_name) == {"plain", "eng"}
    assert by_name["plain"]["filter"] is None
    assert by_name["eng"]["filter"]["must"][0]["match"] == "eng"
    assert by_name["plain"]["k"] == K
    assert by_name["plain"]["corpus_dtype"] == "float32"
    assert by_name["plain"]["queries"] == 4
    # A whole-corpus run can report short top-Ks; the outputs it names exist.
    assert by_name["plain"]["queries_short_of_k"] == 0
    assert sorted(doc["output_files"]) == sorted(
        p.rsplit("/", 1)[-1] for p in results.values()
    )
    for name in doc["output_files"]:
        assert (out / name).exists()


def test_sharded_run_writes_one_manifest_per_rank_and_a_merge_manifest(ds, tmp_path):
    out = tmp_path / "sharded"
    out.mkdir()
    cfg = _cfg(ds, out)
    num_jobs = 2
    for rank in range(num_jobs):
        run_compute(cfg, num_jobs=num_jobs, job_rank=rank)

    mdir = out / "_bf_manifest_queries_compute"
    ranks = sorted(mdir.glob("rank*.json"))
    assert [p.name for p in ranks] == ["rank000.json", "rank001.json"]
    per_rank = [_read(p) for p in ranks]
    # Every rank fingerprints the corpus BEFORE its stride slice, so the hash is
    # about the corpus, not the slice — ranks that disagree here disagreed about
    # the corpus itself (an include/exclude drift), which is the failure this
    # is meant to expose.
    fps = {json.dumps(d["source"]["corpus"]["fingerprint"], sort_keys=True) for d in per_rank}
    assert len(fps) == 1
    for rank, doc in enumerate(per_rank):
        assert doc["sharding"]["job_rank"] == rank
        assert doc["sharding"]["num_jobs"] == num_jobs
        assert doc["sharding"]["corpus_files_total"] == 2
        assert doc["sharding"]["corpus_files_this_worker"] == 1
        # A partial's short-of-k count would describe a stride slice, not the
        # ground truth, so it is deliberately absent.
        assert "queries_short_of_k" not in doc["searches"][0]
        for name in doc["output_files"]:
            assert (out / name).exists()
    # The ranks together cover the corpus exactly once.
    assert sum(d["counts"]["corpus_rows_scanned"] for d in per_rank) == 9

    run_merge(cfg)
    doc = _read(out / "_bf_manifest_queries_merge.json")
    assert doc["phase"] == "merge"
    assert doc["counts"]["partials_merged"] == num_jobs * len(cfg.searches)
    assert doc["counts"]["queries"] == 4
    # Merge never opens the corpus, so it has no file list to fingerprint —
    # the compute manifests own that, and `partial_dir` traces back to them.
    assert "fingerprint" not in doc["source"]["corpus"]
    by_name = {s["name"]: s for s in doc["searches"]}
    assert by_name["eng"]["partials"] == num_jobs
    assert by_name["eng"]["filter"]["must"][0]["match"] == "eng"
    # Storage dtypes are carried off the partials, not re-derived from a corpus
    # merge never opens.
    assert by_name["eng"]["corpus_dtype"] == "float32"
    for name in doc["output_files"]:
        assert (out / name).exists()


def test_manifest_failure_does_not_fail_the_run(ds, tmp_path, monkeypatch, caplog):
    """A manifest records outputs that already landed — losing it must not lose
    them (a multi-hour GT run is not worth an S3 hiccup on a 4 KB PUT)."""
    import nova_bf.io as bf_io

    out = tmp_path / "boom"
    out.mkdir()

    def explode(self, filename, data):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(bf_io.Store, "write_bytes", explode)
    results = run_compute(_cfg(ds, out))

    assert not list(out.glob("_bf_manifest*"))
    for path in results.values():  # the parquets are still there and readable
        assert pq.read_table(path).num_rows == 4


def test_corpus_fingerprint_is_order_sensitive():
    """Reordering the corpus is not the same corpus: `make_point_id` and the
    ordinal tie-break both key off the global file index, so two runs over the
    same files in a different order produce different hit ids."""
    files = [ParquetFile(read_path=f"bucket/c/f{i}.parquet", key=f"c/f{i}.parquet") for i in range(3)]
    base = corpus_fingerprint(files)
    assert base["files"] == 3
    assert corpus_fingerprint(list(files)) == base  # stable across calls
    assert corpus_fingerprint(files[::-1])["sha256"] != base["sha256"]
    assert corpus_fingerprint(files[:2])["sha256"] != base["sha256"]
    # Hashed over the loader key (what make_point_id consumes), not read_path,
    # so the same logical corpus fingerprints the same from a different mount.
    moved = [ParquetFile(read_path=f"other-bucket/{f.key}", key=f.key) for f in files]
    assert corpus_fingerprint(moved)["sha256"] == base["sha256"]
    assert corpus_fingerprint([]) == {"files": 0, "sha256": corpus_fingerprint([])["sha256"],
                                      "first": None, "last": None}


def test_gpu_peak_is_empty_without_cuda():
    assert gpu_peak("cpu") == {}
    assert gpu_peak(None) == {}


def test_query_counts_distinguish_the_file_from_what_was_searched(ds, tmp_path):
    """With `rows` subsets a search covers fewer queries than the file holds, so
    the two counts must not be one field: a reader taking `queries` for "queries
    searched" would overstate every subsetted run."""
    out = tmp_path / "subset"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2),
        searches=[
            SearchSpec(name="a", metric="dot", k=K,
                       rows=RowSelector(column="qid", isin=["q0", "q1"])),
            # overlaps `a` on q1: the union is 3 rows, not the 4 a sum would give
            SearchSpec(name="b", metric="dot", k=K,
                       rows=RowSelector(column="qid", isin=["q1", "q2"])),
        ],
    )
    run_compute(cfg)

    doc = _read(out / "_bf_manifest_queries_compute.json")
    assert doc["counts"]["queries_in_file"] == 4
    assert doc["counts"]["queries_searched"] == 3
    by_name = {s["name"]: s for s in doc["searches"]}
    assert by_name["a"]["queries"] == 2 and by_name["b"]["queries"] == 2
    assert by_name["a"]["rows"] == {"column": "qid", "isin": ["q0", "q1"]}


# ---------------------------------------------------------------------------
# provenance the document is supposed to carry
# ---------------------------------------------------------------------------


def test_code_versions_reports_git_for_an_in_repo_checkout():
    info = run_manifest.code_versions()
    assert info.get("git_commit"), "in-repo run should still report a commit"
    assert len(info["git_commit"]) >= 7


def test_code_versions_omits_git_when_the_repo_is_not_ours(monkeypatch, tmp_path):
    """A wheel install inside an unrelated checkout: git walks upward and
    answers about that repo. A plausible sha for the wrong code is worse than
    no sha, so the whole git block must drop out."""
    # Patch the module's own __file__ (which `pkg_dir` is derived from) rather
    # than os.path.abspath — patching that also changes os.path.realpath, which
    # calls it internally, so the check would compare two fake paths and agree.
    pkg = tmp_path / "site-packages" / "nova_bf"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(run_manifest, "__file__", str(pkg / "manifest.py"))

    class R:
        def __init__(self, out): self.stdout = out

    def fake_run(cmd, **kw):
        if "--show-toplevel" in cmd:
            return R("/some/other/repo\n")     # does NOT contain pkg_dir
        return R("deadbeef\n")

    monkeypatch.setattr(run_manifest.subprocess, "run", fake_run)
    info = run_manifest.code_versions()
    assert "git_commit" not in info, (
        "git answered about a repo that does not contain this package")
    assert info.get("python_version"), "non-git fields must still be reported"


# --- 18 + today's additions: the manifest records what RAN ----------------


def test_kernel_switches_report_resolved_booleans(monkeypatch):
    for var in ("NOVA_BF_NO_PRUNE", "NOVA_BF_NO_FOLD_KERNEL",
                "NOVA_BF_NO_TOPK_KERNEL"):
        monkeypatch.delenv(var, raising=False)
    assert run_manifest.kernel_switches() == {
        "prune": True, "fold_kernel": True, "topk_kernel": True}

    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    monkeypatch.setenv("NOVA_BF_NO_FOLD_KERNEL", "1")
    assert run_manifest.kernel_switches() == {
        "prune": False, "fold_kernel": False, "topk_kernel": True}

    # `""` is unset-shaped and must not read as "disabled"
    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "")
    assert run_manifest.kernel_switches()["prune"] is True


# --- peak host RSS, and which sparse paths actually ran -------------------


def test_host_peak_reports_bytes_not_kib():
    """Linux `ru_maxrss` is KiB, macOS is bytes. The field sits next to
    `peak_gpu_allocated_bytes`, so a 1024x misread is easy and silent."""
    import resource

    got = run_manifest.host_peak()["peak_host_rss_bytes"]
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    expect = raw if os.sys.platform == "darwin" else raw * 1024
    assert got == expect
    assert got > 1_000_000, "a live process cannot peak under 1 MB"


def test_sparse_branches_are_counted_and_reset_per_run(tmp_path):
    """The swapped and cuSPARSE paths have DIFFERENT run-to-run determinism
    (~1/10,000 vs ~1423/10,000), so which one ran decides whether two
    artifacts may legitimately differ. Counters are process-global, so they
    must also reset — otherwise run two reports run one's branches too."""
    pytest.importorskip("torch")
    import json

    from nova_bf import compute as C
    from nova_bf.compute import run_compute
    from test_prune_search_paths import _sparse_cfg, _sparse_corpus

    cdir, qpath = _sparse_corpus(tmp_path, n_files=3, per_file=60, seed=4)
    run_compute(_sparse_cfg(cdir, qpath, tmp_path / "r1", k=5))
    first = dict(C._SPARSE_BRANCHES)
    assert sum(first.values()) > 0, "no sparse branch was recorded"
    # positive-weight fixture -> the cheap gate, never the structural one
    assert first["gate_zero"] > 0 and first["gate_structural"] == 0

    run_compute(_sparse_cfg(cdir, qpath, tmp_path / "r2", k=5))
    second = dict(C._SPARSE_BRANCHES)
    assert second == first, f"counters accumulated across runs: {first} -> {second}"


def test_params_record_the_resolved_cpu_thread_count(ds, tmp_path):
    """`params` is what RAN. `cpu_thread_count: 0` means "use os.cpu_count()",
    so the configured value answers nothing when you are reading a manifest
    later asking why one rank was slow — the resolved number has to be here.
    """
    import os as _os

    out = tmp_path / "cpu"
    out.mkdir()
    run_compute(_cfg(ds, out))

    doc = _read(out / "_bf_manifest_queries_compute.json")
    got = doc["params"]["cpu_thread_count"]
    assert got == (_os.cpu_count() or 1), (
        f"expected the RESOLVED thread count, got {got!r}")
    assert got > 0


def test_sparse_branches_reach_the_manifest(tmp_path):
    pytest.importorskip("torch")
    import json

    from nova_bf.compute import run_compute
    from test_prune_search_paths import _sparse_cfg, _sparse_corpus

    out = tmp_path / "m"
    cdir, qpath = _sparse_corpus(tmp_path, n_files=2, per_file=50, seed=9)
    run_compute(_sparse_cfg(cdir, qpath, out, k=4))

    manifests = list(out.rglob("*manifest*.json"))
    assert manifests, "no manifest was written"
    doc = json.loads(manifests[0].read_text())
    assert "sparse_branches" in doc["params"], doc["params"].keys()
    assert doc["compute"].get("peak_host_rss_bytes", 0) > 0


# --- 15: never invent a run fingerprint ------------------------------------
