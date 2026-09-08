"""`NOVA_BF_READ_TIMING` — the read-phase split.

The premise of this instrumentation is that it must not change what a run
PRODUCES, only how long the stages appear to take. That claim was untested:
nothing pinned the alternate read path against the production one, nothing
exercised the ON path at all, and nothing pinned the accumulator reset. The
alternate path fully buffers a file before decoding, so it is a genuinely
different read, and "it happens to agree today" is not the same as "it agrees".
"""
from __future__ import annotations

import glob
import json
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import io as io_mod
from nova_bf import profiling
from nova_bf.compute import run_compute
from nova_bf.io import Store

from test_profile_window import _cfg


def _wide_file(tmp_path):
    """A file whose columns differ wildly in size, so a column subset is a much
    smaller read than the whole file."""
    n = 2000
    rng = np.random.default_rng(0)
    t = pa.table({
        "sid": pa.array([f"id{i}" for i in range(n)]),
        "dense": pa.array(rng.random((n, 64)).astype(np.float32).tolist(),
                          pa.list_(pa.float32())),
        "text": pa.array(["lorem ipsum " * 8] * n),
    })
    p = str(tmp_path / "wide.parquet")
    pq.write_table(t, p, compression="snappy", row_group_size=137)
    return p


@pytest.mark.parametrize("columns", [None, ["sid"], ["dense", "sid"], ["sid", "dense"]])
@pytest.mark.parametrize("ranged", [False, True])
def test_the_timing_read_returns_exactly_the_production_table(
        tmp_path, monkeypatch, columns, ranged):
    """`fetch_and_decode` must be `read_columns` with a stopwatch, for every
    shape the scan asks for -- including a REORDERED subset and both
    `ranged_get` branches."""
    path = _wide_file(tmp_path)
    if ranged:
        monkeypatch.setattr(io_mod, "_RANGED_GET_MIN_BYTES", 0)
    store = Store(str(tmp_path), ranged_get=ranged)

    want = store.read_columns(path, columns)
    got, rt = profiling.fetch_and_decode(store, path, columns)

    assert got.equals(want)
    assert got.schema.equals(want.schema, check_metadata=True)
    assert [c.num_chunks for c in got.columns] == [c.num_chunks for c in want.columns]
    # the keys this function itself fills; `total` is the caller's
    assert rt["fetch"] >= 0 and rt["decode_parquet"] >= 0
    assert rt["bytes"] > 0
    assert rt["fetch_mode"] == (1.0 if ranged else 0.0)


def test_the_split_does_not_leak_into_a_run_that_did_not_ask_for_it(
        tmp_path, monkeypatch):
    """The accumulator is module-global; the reset must not be gated on the flag
    or a later untimed run reports the earlier one's numbers."""
    monkeypatch.setenv("NOVA_BF_READ_TIMING", "1")
    run_compute(_cfg(tmp_path / "on", n_files=3))
    monkeypatch.delenv("NOVA_BF_READ_TIMING")
    run_compute(_cfg(tmp_path / "off", n_files=3))

    def timing(root):
        m = glob.glob(str(root / "out" / "_bf_manifest_*compute.json"))
        assert m, f"no manifest under {root}"
        return json.loads(open(m[0]).read())["timing"]

    assert timing(tmp_path / "on")["read_timing"] is True
    assert "read_split" in timing(tmp_path / "on")
    assert timing(tmp_path / "off")["read_timing"] is False
    assert "read_split" not in timing(tmp_path / "off")


def test_the_marker_records_the_FLAG_not_whether_anything_was_measured(
        tmp_path, monkeypatch):
    """A rank whose slice of the corpus is empty still asked for read timing.

    `bool(split)` passes the ordinary test -- a rank that read files has a
    non-empty split either way -- so only a rank with NO files can tell the
    proxy from the flag. With `num_jobs` above the file count the last ranks get
    nothing, and reporting `false` there would contradict their siblings in the
    same run.
    """
    monkeypatch.setenv("NOVA_BF_READ_TIMING", "1")
    run_compute(_cfg(tmp_path / "empty", n_files=2), num_jobs=8, job_rank=7)
    # A sharded rank writes `_bf_manifest_<stem>_compute/rank007.json` -- the
    # word "manifest" is in the DIRECTORY, not the file name.
    m = glob.glob(str(tmp_path / "empty" / "out" / "**" / "*.json"),
                  recursive=True)
    assert m, "no manifest for the empty rank"
    timing = json.loads(open(m[0]).read())["timing"]
    assert timing["read_timing"] is True, (
        "the empty rank reported read_timing=false while its siblings report "
        f"true: {timing}")
    assert "read_split" not in timing, (
        "it measured nothing, so there should be no split to report")


def test_a_timed_run_is_marked_as_such_in_the_manifest(tmp_path, monkeypatch):
    """Every read-derived figure in the manifest is inflated by the
    instrumentation (whole-file fetch + a re-decode per column group), so a
    timed run must be identifiable or it reads as a throughput regression."""
    monkeypatch.setenv("NOVA_BF_READ_TIMING", "1")
    run_compute(_cfg(tmp_path / "t", n_files=3))
    m = glob.glob(str(tmp_path / "t" / "out" / "_bf_manifest_*compute.json"))
    t = json.loads(open(m[0]).read())["timing"]
    assert t["read_timing"] is True
    assert t["read_seconds_summed"] > 0


def test_the_split_names_each_stage_it_actually_times(tmp_path):
    """One field per operation, and only seconds in the seconds dict.

    `dense_cast` used to time `multivector_to_ragged` too, so on an MV run the
    label was simply wrong. `sparse` covered decode + norms + zero-gate +
    remap, which scale with file bytes, nnz, nnz and QUERY VOCAB — one number
    could not say which was the cost. And `bytes`/`fetch_mode` were summed into
    the same mapping, making `read_split.fetch_mode` a count of files inside a
    dict the log prints as seconds.
    """
    import os
    import subprocess
    import sys

    import test_reference_release_is_output_neutral as R

    root, out = tmp_path / "data", tmp_path / "out"
    R._make_data(root)                       # dense + sparse + multivector
    env = dict(os.environ)
    env["NOVA_BF_READ_TIMING"] = "1"
    r = subprocess.run([sys.executable, "-c", R._DRIVER, str(root), str(out), "full"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]

    m = glob.glob(str(out / "**" / "_bf_manifest_*compute.json"), recursive=True)
    assert m, "no compute manifest"
    timing = json.loads(open(m[0]).read())["timing"]
    split = timing["read_split"]

    for stage in ("dense_cast", "multivector_cast",
                  "sparse_decode", "sparse_norms", "sparse_gate", "sparse_remap"):
        assert stage in split, f"{stage} missing from {sorted(split)}"
    assert "sparse" not in split, "the merged sparse field is back"

    # counts and bytes are their own fields, not entries in a seconds mapping
    assert not {"bytes", "fetch_mode"} & set(split), sorted(split)
    assert timing["read_bytes"] > 0
    assert timing["ranged_files"] >= 0


def test_the_split_is_reset_before_any_reader_reports(tmp_path, monkeypatch):
    """Pins WHERE the reset happens, not just that it happens.

    `reset_read_split()` has to run before the reader threads start. Moving it
    into `profiling.configure()` looks like tidying -- that function already
    resets stale profiler state -- but `configure` runs ~100 lines and one
    thread-launch later, so it would throw away timings files have already
    reported. Observed with most of the split silently lost, with the same field
    names and plausible values.

    The existing tests only pin that the reset is UNCONDITIONAL, which the moved
    version still satisfies. This pins the order, and cross-checks the surviving
    total against what the per-file log lines actually reported.
    """
    order: list[str] = []
    real_reset, real_add = profiling.reset_read_split, profiling.read_split_add

    def spy_reset():
        order.append("reset")
        return real_reset()

    def spy_add(parts):
        order.append("add")
        return real_add(parts)

    monkeypatch.setattr(profiling, "reset_read_split", spy_reset)
    monkeypatch.setattr(profiling, "read_split_add", spy_add)
    monkeypatch.setenv("NOVA_BF_READ_TIMING", "1")

    # WIDEN THE RACE. `configure()` runs only ~100 lines after the reader
    # threads start, with no I/O between, so on a fast local fixture no reader
    # has reported yet and a reset moved into `configure` would still land
    # first -- the test would pass and prove nothing (it did, 3/3, before this).
    # Delaying `configure` gives the readers time to report, which is the state
    # a real 10B-row run is always in.
    real_configure = profiling.configure

    def slow_configure():
        time.sleep(0.5)
        return real_configure()

    monkeypatch.setattr(profiling, "configure", slow_configure)

    import logging

    records: list[str] = []

    class Grab(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    lg = logging.getLogger("nova_bf.compute")
    h = Grab()
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        run_compute(_cfg(tmp_path, n_files=8))
    finally:
        lg.removeHandler(h)

    assert "reset" in order, "the split was never reset"
    assert "add" in order, "no reader reported a split, so this proves nothing"
    assert order.index("reset") < order.index("add"), (
        f"a reader reported before the reset: {order[:6]} — the reset has moved "
        "after the reader threads start, so early files are discarded")
    assert order.count("reset") == 1, f"reset called {order.count('reset')}x"

    # Outcome cross-check: every file logs its own `total`, and the manifest
    # holds the sum. A reset that lands mid-run leaves the manifest short.
    per_file = [float(m.group(1)) for m in
                (re.search(r"read-split .*? total=([0-9.]+)s", r) for r in records)
                if m]
    assert len(per_file) == 8, f"expected 8 per-file lines, got {len(per_file)}"
    m = glob.glob(str(tmp_path / "out" / "**" / "*.json"), recursive=True)
    total = json.loads(open(m[0]).read())["timing"]["read_split"]["total"]
    assert total == pytest.approx(sum(per_file), abs=0.01 * len(per_file)), (
        f"manifest total {total:.3f} vs {sum(per_file):.3f} summed from the "
        f"{len(per_file)} per-file lines — files were dropped from the split")
