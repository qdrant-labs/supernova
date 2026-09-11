"""The per-file progress row `run_compute` emits.

One line per corpus file carrying that file's deltas of wall, consumer GPU
enqueue and consumer io_wait, plus the file's own reader read/filter seconds.
It exists so a long run's two regimes — the live-heavy start and the steady
state a rank projection depends on — can be separated after the fact, so the
thing worth testing is that the line is WELL FORMED and its numbers are
self-consistent, not what they happen to be on a toy corpus.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf.compute import run_compute
from nova_bf.config import load_config

LINE = re.compile(
    r"per-file file=(?P<gidx>\d+) n=(?P<n>\d+)/(?P<total>\d+) "
    r"t=(?P<t>[\d.]+) dwall=(?P<dwall>[\d.]+) dgpu=(?P<dgpu>[\d.]+) "
    r"dio_wait=(?P<dio>[\d.]+) read=(?P<read>[\d.]+) filter=(?P<filter>[\d.]+) "
    r"rows=(?P<rows>\d+)(?P<rest>.*)$"
)

# The adjacent two-pass row, emitted only for files some of whose dense slices
# actually took the two-pass.
TP_LINE = re.compile(
    r"twopass file=(?P<gidx>\d+) slices=(?P<tp>\d+)/(?P<total>\d+) "
    r"live=(?P<live>[\d.]+) padded=(?P<padded>[\d.]+)(?P<rest>.*)$"
)


# `dim` is 64, not 8: the closed-form bound's P6 requires `64 <= d <= 2**20`
# (Lemma 2' needs `d >= 64` for its block count), so below it `upper_bounds`
# refuses every row, the two-pass never engages, and no progress rows exist to
# take a delta of.
def _corpus(root, n_files=4, rows=32, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for f in range(n_files):
        v = rng.standard_normal((rows, dim)).astype(np.float32)
        pq.write_table(pa.table({
            "id": pa.array([f"d{f}-{i}" for i in range(rows)]),
            "emb": pa.array(list(v.tolist()), type=pa.list_(pa.float32(), dim)),
        }), root / f"c{f:03d}.parquet")


def _queries(root, n=6, dim=64, seed=1):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    pq.write_table(pa.table({
        "qid": pa.array([f"q{i}" for i in range(n)]),
        "emb": pa.array(list(v.tolist()), type=pa.list_(pa.float32(), dim)),
    }), root / "q.parquet")


def _cfg(tmp_path, n_files):
    _corpus(tmp_path / "corpus", n_files=n_files)
    _queries(tmp_path / "queries")
    (tmp_path / "out").mkdir()
    text = f"""
corpus:
  path: {tmp_path / 'corpus'}
  dense_column: emb
  id_column: id
queries:
  path: {tmp_path / 'queries'}
  dense_column: emb
  id_column: qid
output:
  path: {tmp_path / 'out'}
params:
  io_workers: 2
searches:
  - name: dense_cosine
    vector_type: dense
    metric: cosine
    k: 3
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return load_config(str(p))


def _slow_reads(monkeypatch, seconds=0.02):
    """Inject a KNOWN per-file read delay and return it.

    Without it this fixture runs in ~0.1 s, every timing field prints 0.00, and
    the assertions below compare zero to zero -- which is how `dwall` as a
    running total and a 3x-inflated `read` both passed.
    """
    from nova_bf.io import Store

    # With NOVA_BF_READ_TIMING set the reader takes `profiling.fetch_and_decode`
    # instead of `Store.read_columns`, so the sleep would never fire and the
    # guard would (correctly) refuse. Do not inherit it from the shell.
    monkeypatch.delenv("NOVA_BF_READ_TIMING", raising=False)
    real_read = Store.read_columns

    def slow(self, path, cols):
        time.sleep(seconds)
        return real_read(self, path, cols)

    monkeypatch.setattr(Store, "read_columns", slow)
    return seconds


def _min_expected_wall(delay, n_files, io_workers):
    """Half the wall the injected delay must produce, as a floor for the guard.

    The sleep is in the reader POOL, so the run is about
    `ceil(n / io_workers) * delay`. Deriving it from `io_workers` rather than
    hard-coding 0.5 matters: at io_workers 3, 4 and 8 a fixed factor tuned for
    2 becomes a coin flip.
    """
    return delay * math.ceil(n_files / max(io_workers, 1)) * 0.5


@pytest.mark.parametrize("n_files", [1, 5])
def test_one_well_formed_line_per_file(tmp_path, caplog, n_files, monkeypatch):
    caplog.set_level("INFO", logger="nova_bf.compute")
    delay = _slow_reads(monkeypatch)
    cfg = _cfg(tmp_path, n_files)
    run_compute(cfg)
    lines = [m for m in (LINE.search(r.getMessage()) for r in caplog.records) if m]
    assert len(lines) == n_files, f"expected {n_files} rows, got {len(lines)}"
    prev_t = 0.0
    dwalls: list[float] = []
    for i, m in enumerate(lines, start=1):
        assert int(m["n"]) == i, "n counts files consumed, in order"
        assert int(m["total"]) == n_files
        assert int(m["rows"]) == 32
        t = float(m["t"])
        assert t >= prev_t, "t is monotonic seconds since the scan started"
        # dwall is the gap since the previous row, so the two must agree.
        # Tight: both fields print as %.2f, so two roundings is all the slack
        # this needs. The old 0.05 exceeded the entire toy run.
        assert float(m["dwall"]) == pytest.approx(t - prev_t, abs=0.011)
        prev_t = t
        dwalls.append(float(m["dwall"]))
        # every delta is a non-negative slice of a monotone counter
        for k in ("dgpu", "dio", "read", "filter"):
            assert float(m[k]) >= 0.0, k
        # the consumer cannot spend more time enqueueing plus waiting than the
        # file's whole wall clock, give or take the log call itself
        assert float(m["dgpu"]) + float(m["dio"]) <= float(m["dwall"]) + 0.05
    # The per-row check above allows 0.05, which on a toy run exceeds the whole
    # scan -- so `dwall = now - wall0` (a running TOTAL, not a delta) passed it.
    # Across the run the deltas must add up to the last `t`, which a total
    # cannot do for more than one file. Tolerance is the printing: %.2f per row
    # plus %.2f on `t`.
    assert prev_t >= _min_expected_wall(delay, n_files, cfg.params.io_workers), (
        f"the run was too fast ({prev_t:.3f}s) for the timing checks below to "
        "discriminate; the injected delay did not take effect")
    assert sum(dwalls) == pytest.approx(prev_t, abs=0.005 * (len(lines) + 1)), (
        f"dwall values {dwalls} sum to {sum(dwalls):.3f}, but the last t is "
        f"{prev_t:.3f} — dwall is not a per-file delta")


def test_the_line_carries_the_live_fractions(tmp_path, caplog):
    """They are on the same row so one line has everything a block summary
    needs — no joining against a second log line by file index."""
    caplog.set_level("INFO", logger="nova_bf.compute")
    run_compute(_cfg(tmp_path, 3))
    rest = [m["rest"] for m in
            (LINE.search(r.getMessage()) for r in caplog.records) if m]
    assert len(rest) == 3, f"no progress rows matched, so nothing was checked: {rest}"
    assert all("dense_cosine=" in r for r in rest), rest

    # The NUMBER, not just the label -- checking only that the name appears let
    # a halved fraction through.
    import json
    import statistics

    got = [float(r.split("dense_cosine=")[1].split()[0]) for r in rest]
    assert all(0.0 <= v <= 1.0 for v in got), got
    assert any(v > 0.0 for v in got), f"every file reported nothing live: {got}"
    doc = json.loads(next((tmp_path / "out").rglob("*manifest*.json")).read_text())
    overall = doc["params"]["kernels"]["prune"]["by_search"]["dense_cosine"]

    # TRUE FOR ANY WEIGHTING: the run total is a weighted mean of the per-file
    # fractions, so it cannot sit outside their range.
    assert min(got) - 1e-9 <= overall["live_fraction"] <= max(got) + 1e-9, (
        got, overall)

    # The stronger `fmean == total` law needs EQUAL WEIGHTS, and the weight is
    # (queries x slices per file) -- not rows, as an earlier version of this
    # comment claimed. Unequal row counts do not break it; a configured
    # `*_batch_size` does, by giving files different slice counts (observed:
    # fmean 0.477 vs a manifest 0.319 at dense_batch_size=16). Every production
    # config sets 4096, so assert the precondition rather than the folklore --
    # this fails loudly the day someone adds a batch size to the fixture.
    assert overall["slices"] == len(rest), (
        f"{overall['slices']} prune slices across {len(rest)} files, so the "
        "per-file fractions are not equally weighted and the mean below is not "
        "the run total; give this fixture one slice per file or drop the check")
    assert statistics.fmean(got) == pytest.approx(
        overall["live_fraction"], abs=1e-4), (got, overall)


def test_deltas_sum_to_the_run_totals(tmp_path, caplog, monkeypatch):
    """The per-file rows are a decomposition of the bf-bench line, not an
    independent measurement that could drift from it."""
    caplog.set_level("INFO", logger="nova_bf.compute")
    # Same injected delay as above. Warm, this fixture ran in ~0.1 s while the
    # tolerance below is 0.125 s, so a `read` field inflated 3x passed -- the
    # test only had teeth on a cold parquet path, i.e. by accident of ordering.
    _slow_reads(monkeypatch)
    run_compute(_cfg(tmp_path, 5))
    msgs = [r.getMessage() for r in caplog.records]
    rows = [m for m in (LINE.search(x) for x in msgs) if m]
    assert rows, "no progress rows matched, so nothing was checked"
    bench = next(x for x in msgs if "bf-bench io_workers" in x)
    got = dict(kv.split("=") for kv in bench.split() if "=" in kv)
    workers = int(got["io_workers"])
    # ABSOLUTE tolerance derived from the two print widths, not `rel`. The
    # bf-bench fields are %.1f and the per-file ones %.2f, so the two sides can
    # never agree exactly -- and with `rel`, a bf-bench value that rounds to 0.0
    # collapsed the tolerance to ~0, so this test failed whenever it ran first
    # in a process (it passed only because earlier tests warmed the read path).
    slack = 0.05 * workers + 0.005 * len(rows)
    assert sum(float(r["read"]) for r in rows) == pytest.approx(
        float(got["read_wall_s"]) * workers, abs=slack)
    # NOTE: this fixture configures no filter, so both sides of the filter
    # check are identically 0.0 -- it is a tautology here, kept only so the
    # relation is stated. It becomes a real decomposition only against a
    # filtered spec; `tests/test_prune_search_paths.py` has those fixtures.
    assert sum(float(r["filter"]) for r in rows) == pytest.approx(
        float(got["filter_wall_s"]) * workers, abs=slack)
    assert sum(float(r["dgpu"]) for r in rows) <= float(got["gpu_s"]) + 0.05
    assert sum(float(r["dio"]) for r in rows) == pytest.approx(
        float(got["io_wait_s"]), abs=0.05)


def test_the_twopass_line_reports_per_file_deltas(tmp_path, caplog, monkeypatch):
    """`twopass.stats()` is cumulative for the whole run; this line is not.

    It sits next to the `per-file` row precisely so a run's two regimes can be
    separated, and cumulative values cannot do that — the live-heavy first
    files would be smeared into every later row and the transition the line
    exists to show would be invisible. The check is exact rather than
    approximate: every two-pass slice belongs to exactly one file, so the
    per-file `slices=` numerators must sum to the manifest's `launches`,
    which a running total cannot do for more than one file.
    """
    import json

    from nova_bf import twopass

    # Fixture scale. None of these is a correctness parameter (see
    # `tests/test_twopass_pipeline.py`); the run has 6 query rows.
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 2)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 2)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)

    caplog.set_level("INFO", logger="nova_bf.compute")
    run_compute(_cfg(tmp_path, 5))

    rows = [m for m in (TP_LINE.search(r.getMessage()) for r in caplog.records)
            if m]
    assert len(rows) >= 2, (
        "fewer than two two-pass rows, so no delta was exercised and this "
        f"test proved nothing: {[r.group(0) for r in rows]}")

    doc = json.loads(
        next((tmp_path / "out").rglob("*manifest*.json")).read_text())
    tp = doc["params"]["kernels"]["twopass"]
    assert tp["launches"] > 0, tp

    # Against `slices_twopass`, NOT `launches`. The line's numerator is a
    # `slices_twopass` delta, while `launches` counts narrowed exact GEMMs —
    # and those differ by every slice whose live set came out empty, which
    # `_twopass_prepare` calls "not a degenerate case: at steady state a whole
    # slice can fail to beat any query's threshold". Measured on a corpus that
    # produces them: numerators summed to 199 against 95 launches. Comparing
    # to `launches` only passed because this fixture never fully prunes a
    # slice, so it pinned an identity that is false in general.
    assert sum(int(r["tp"]) for r in rows) == tp["slices_twopass"], (
        f"the per-file numerators {[r['tp'] for r in rows]} sum to "
        f"{sum(int(r['tp']) for r in rows)} against "
        f"{tp['slices_twopass']} two-pass slices in the manifest — the line "
        "is reporting a running total, not a delta")
    for r in rows:
        assert 0 < int(r["total"]) <= 2, (
            f"{r['total']} slices attributed to one file of a fixture that "
            "scores each file in a single slice — a cumulative count")
        assert int(r["tp"]) <= int(r["total"]), r.group(0)
        live, padded = float(r["live"]), float(r["padded"])
        assert 0.0 <= live <= 1.0, r.group(0)
        # the padding only ever adds rows to the exact GEMM
        assert live <= padded <= 1.0, r.group(0)
