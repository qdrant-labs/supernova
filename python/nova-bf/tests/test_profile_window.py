"""`NOVA_BF_PROFILE_FILES` — the steady-state profiling window.

`prof_gpu.py` profiles from file 1, which is the wrong regime entirely: the
top-K state is still filling, nearly every row is live and the two-pass has not
engaged. This window runs the scan normally and records only the files asked
for, so what lands in the trace is the steady state.

The tests below check the parsing, that the window opens and closes at the
right FILE and writes both artifacts, and — the property that matters — that
having it set does not change the run's results.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import compute as C  # noqa: F401
from nova_bf import profiling
from nova_bf.compute import run_compute
from nova_bf.config import load_config


def _cfg(tmp_path, n_files=6, rows=32, dim=8):
    rng = np.random.default_rng(0)
    (tmp_path / "corpus").mkdir(parents=True)
    for f in range(n_files):
        v = rng.standard_normal((rows, dim)).astype(np.float32)
        pq.write_table(pa.table({
            "id": pa.array([f"d{f}-{i}" for i in range(rows)]),
            "emb": pa.array(list(v.tolist()), type=pa.list_(pa.float32(), dim)),
        }), tmp_path / "corpus" / f"c{f:03d}.parquet")
    (tmp_path / "queries").mkdir()
    q = rng.standard_normal((5, dim)).astype(np.float32)
    pq.write_table(pa.table({
        "qid": pa.array([f"q{i}" for i in range(5)]),
        "emb": pa.array(list(q.tolist()), type=pa.list_(pa.float32(), dim)),
    }), tmp_path / "queries" / "q.parquet")
    (tmp_path / "out").mkdir()
    p = tmp_path / "cfg.yaml"
    p.write_text(f"""
corpus: {{path: {tmp_path / 'corpus'}, dense_column: emb, id_column: id}}
queries: {{path: {tmp_path / 'queries'}, dense_column: emb, id_column: qid}}
output: {{path: {tmp_path / 'out'}}}
params: {{io_workers: 2}}
searches:
  - {{name: dense_cosine, vector_type: dense, metric: cosine, k: 3}}
""")
    return load_config(str(p))


@pytest.mark.parametrize("raw,want", [
    ("", None), ("   ", None), ("60:62", (60, 62)), ("1:1", (1, 1)),
])
def test_window_parsing(monkeypatch, raw, want):
    monkeypatch.setenv("NOVA_BF_PROFILE_FILES", raw)
    assert profiling.parse_window() == want


@pytest.mark.parametrize("raw", ["60", "a:b", "62:60", "0:3", "60:62:64", ""])
def test_a_malformed_window_is_refused_loudly(monkeypatch, raw):
    """A silently-ignored window means a profiling run that produces nothing
    and is only noticed after the node time is spent."""
    if raw == "":
        pytest.skip("empty means 'no window', covered above")
    monkeypatch.setenv("NOVA_BF_PROFILE_FILES", raw)
    with pytest.raises(ValueError, match="NOVA_BF_PROFILE_FILES"):
        profiling.parse_window()


def test_the_window_writes_both_artifacts_and_only_once(tmp_path, monkeypatch,
                                                        caplog):
    # the parent: the profiler now logs under `nova_bf.profiling`,
    # the scan still under `nova_bf.compute`
    caplog.set_level("INFO", logger="nova_bf")
    out = tmp_path / "prof"
    monkeypatch.setenv("NOVA_BF_PROFILE_FILES", "3:4")
    monkeypatch.setenv("NOVA_BF_PROFILE_OUT", str(out))
    run_compute(_cfg(tmp_path, n_files=6))
    assert (out / "C_kernels.txt").stat().st_size > 0
    assert (out / "C_trace.json.gz").stat().st_size > 0
    assert not (out / "C_trace.json").exists(), "the raw trace must be cleaned up"
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("torch.profiler ON" in m for m in msgs) == 1
    # The file it OPENED on, not just the window it was asked for. Mutating
    # `start`'s `n != w[0]` to open one file late used to leave every assertion
    # here passing while the trace held half the slices.
    assert any("ON at file 3" in m for m in msgs), msgs[-6:]
    assert sum("torch.profiler OFF" in m for m in msgs) == 1
    assert any("OFF after file 4" in m for m in msgs), msgs[-6:]


def test_the_window_does_not_change_results(tmp_path, monkeypatch):
    """The whole point: the run proceeds normally, so the profiled run's
    ground truth is the unprofiled run's."""
    a = tmp_path / "a"
    a.mkdir()
    plain = _cfg(a, n_files=5)
    run_compute(plain)
    ref = pq.read_table(str(list((a / "out").rglob("*.parquet"))[0]))

    b = tmp_path / "b"
    b.mkdir()
    monkeypatch.setenv("NOVA_BF_PROFILE_FILES", "2:3")
    monkeypatch.setenv("NOVA_BF_PROFILE_OUT", str(tmp_path / "prof2"))
    run_compute(_cfg(b, n_files=5))
    got = pq.read_table(str(list((b / "out").rglob("*.parquet"))[0]))
    assert got.equals(ref)


def test_no_window_leaves_the_profiler_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("NOVA_BF_PROFILE_FILES", raising=False)
    run_compute(_cfg(tmp_path, n_files=3))
    assert profiling._PROF["window"] is None
    assert profiling._PROF["active"] is False
    assert profiling._PROF["prof"] is None


def test_slice_marker_is_a_no_op_when_the_window_is_shut(monkeypatch):
    """It wraps a quarter of a million slice calls per rank, so it must cost
    nothing when nobody is profiling."""
    monkeypatch.setitem(profiling._PROF, "active", False)
    with profiling.slice_mark("bf_slice") as m:
        assert m._rf is None


def test_read_timing_does_not_leak_into_a_later_run(tmp_path, monkeypatch):
    """The split accumulator is module-global. Two `run_compute` calls in one
    process — a sweep, a test session, a notebook — used to make the SECOND
    report the first's numbers, because the reset was gated on the flag while
    the manifest field was not. A manifest claiming timings it never measured
    is worse than one with none."""
    import glob
    import json

    monkeypatch.setenv("NOVA_BF_READ_TIMING", "1")
    run_compute(_cfg(tmp_path / "a", n_files=3))
    monkeypatch.delenv("NOVA_BF_READ_TIMING")
    run_compute(_cfg(tmp_path / "b", n_files=3))

    def has_split(root):
        m = glob.glob(str(root / "out" / "_bf_manifest_*compute.json"))
        assert m, f"no manifest under {root}"
        return "read_split" in (json.loads(open(m[0]).read()).get("timing") or {})

    assert has_split(tmp_path / "a"), "the timed run should record its split"
    assert not has_split(tmp_path / "b"), "the untimed run inherited a split"


def test_a_window_that_never_closes_does_not_leak_into_the_next_run(
        tmp_path, monkeypatch):
    """A window whose END is past the rank's last file never reaches `stop`, so
    `active` stayed set and the next run's `slice_mark` attached to a dead
    profiler (torch: "Requested callback is not found"). `configure` must EXIT
    the stale profiler, not just drop it — releasing an entered
    `torch.profiler.profile` to the garbage collector segfaults the
    interpreter."""
    monkeypatch.setenv("NOVA_BF_PROFILE_FILES", "2:99")   # END past the last file
    monkeypatch.setenv("NOVA_BF_PROFILE_OUT", str(tmp_path / "prof"))
    run_compute(_cfg(tmp_path / "a", n_files=3))
    assert profiling._PROF["active"] is True, "precondition: the window is stuck open"

    monkeypatch.delenv("NOVA_BF_PROFILE_FILES")
    monkeypatch.delenv("NOVA_BF_PROFILE_OUT")
    run_compute(_cfg(tmp_path / "b", n_files=3))

    assert profiling._PROF["active"] is False
    assert profiling._PROF["prof"] is None
    with profiling.slice_mark("x") as m:
        assert m._rf is None, "slice_mark attached in an unprofiled run"


def test_the_marks_actually_appear_in_the_trace(tmp_path, monkeypatch):
    """The `_rf is None` assertions above cannot catch a broken gate.

    `_NoMark._rf` is a class constant, so `m._rf is None` holds no matter what
    `slice_mark` decided — a `slice_mark` that ignored the window and always
    returned the shared no-op passed this whole file. Only the trace itself can
    tell recording from not-recording.
    """
    import gzip
    import json as _json

    out = tmp_path / "prof"
    monkeypatch.setenv("NOVA_BF_PROFILE_FILES", "2:3")
    monkeypatch.setenv("NOVA_BF_PROFILE_OUT", str(out))
    run_compute(_cfg(tmp_path, n_files=4))

    with gzip.open(out / "C_trace.json.gz", "rt") as fh:
        events = _json.load(fh)["traceEvents"]
    names = [e.get("name") for e in events]
    marked = {n: names.count(n) for n in ("bf_slice", "bf_transfer")}
    assert marked["bf_slice"] > 0, (
        "no bf_slice in the trace: the window opened but nothing was marked, "
        f"so slice_mark never built a record_function. bf_* names seen: "
        f"{sorted(n for n in set(names) if n and n.startswith('bf_'))}")
    # On this (unbuffered) path the two marks wrap the same statements, so they
    # come in pairs. The other paths use their own names and need a GPU.
    assert marked["bf_transfer"] == marked["bf_slice"], marked
