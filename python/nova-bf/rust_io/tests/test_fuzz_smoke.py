"""Keeps tests/fuzz_differential.py runnable: every section at a tiny size
(well under 30 s). The real campaigns are run by hand, e.g.

    python tests/fuzz_differential.py --size local --workers 4 --out DIR
"""

import os
import sys

import pytest

pytest.importorskip("rust_fetch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuzz_differential as fd  # noqa: E402


def test_differential_harness_smoke(tmp_path):
    _, (cfg, sections) = fd.build_cfg([
        "--size", "smoke", "--workers", "2", "--out", str(tmp_path / "out"), "--seed", "1",
        "--mutants", "100", "--fetch", "12", "--fetch-mutants", "30", "--progress-s", "3600",
    ])
    lines = []
    rc, summary = fd.run(cfg, sections, log=lines.append)
    log = "\n".join(lines)
    assert rc == 0, log
    c = summary["counts"]
    assert summary["fails"] == 0 and summary["harness_errors"] == 0, log
    # every section ran, and both outcomes are actually exercised
    assert c["matrix.crate"].get("identical", 0) > 0 and c["matrix.store"].get("identical", 0) > 0, log
    assert c["matrix.crate"].get("fallback", 0) > 0, log            # brotli
    assert c["crafted.crate"].get("identical", 0) > 0 and c["crafted.crate"].get("fallback", 0) > 0, log
    assert sum(c["mutate.crate"].values()) == 100 and sum(c["mutate.store"].values()) == 100, log
    assert c["fetch.fetch"].get("identical", 0) > 0, log
    assert sum(c["fetch.fetch_mut"].values()) == 30, log
    assert os.path.exists(tmp_path / "out" / "crafted.tsv")
    assert log.splitlines()[-1].startswith("SUMMARY ")
