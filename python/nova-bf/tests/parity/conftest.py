"""Fixtures for the parity harness.

Everything expensive is session-scoped and built once: the synthetic corpus,
the Qdrant collection, the naive oracle's cached score matrices, and — the big
one — the nova-bf runs. All 119 matrix cases go through nova-bf in ONE run per
device, which is both far faster than 119 runs and a more honest test: it is
the shared-pass code path production actually uses, where one filter is
evaluated once per file for every search that names it and all searches of a
vector_type share one batch grid.

The Qdrant-backed tests skip themselves when no server is reachable
(`QDRANT_URL`, default http://localhost:6333), so the naive half of the suite
still runs anywhere — including in CI with no Qdrant.

That skip is a FOOTGUN, so it is made loud. A suite that skipped the Qdrant
oracle is not the suite it looks like: `naive.py` checks semantics, but only
`qdrant_ref.py` checks fidelity to the engine the ground truth actually grades,
and the two fail differently on purpose. So:

  * `pytest_terminal_summary` prints a banner naming how many cases skipped and
    why, AFTER the pass count, where it cannot be mistaken for a clean run.
  * `NOVA_BF_REQUIRE_QDRANT=1` turns the skip into a hard failure, for CI or
    any run whose result is going to be quoted.

Both exist because a run reporting "1811 passed" with the engine oracle silently
absent reads exactly like a run that checked everything.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from . import cases as cases_mod
from . import corpus as corpus_mod
from . import naive, runner
from .devices import parity_devices

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
def _require_qdrant() -> bool:
    """Whether an unreachable Qdrant is a FAILURE rather than a skip.

    Read at use time, not import time, so a test can monkeypatch the env. And
    `"0"`/`"false"` mean OFF: `bool("0")` is True in Python, so the obvious
    spelling turned the strictest behaviour ON for a CI job that set
    `NOVA_BF_REQUIRE_QDRANT=0` to mean "don't".
    """
    # Fail CLOSED: anything unrecognised ("none", "n", "disabled") turns the
    # strict behaviour ON. Better to over-enforce a testing flag than to have a
    # typo silently disable the check it was set to guarantee.
    return os.environ.get("NOVA_BF_REQUIRE_QDRANT", "").strip().lower() \
        not in ("", "0", "false", "no", "off")

# Filled in by the `client` fixture the first time it probes, and read by
# `pytest_terminal_summary`. Session-scoped state rather than a fixture so the
# summary can see it even when every dependent test skipped.
_QDRANT: dict = {"probed": False, "reachable": None, "graded": False,
                 "reason": None}


def pytest_configure(config):
    config.addinivalue_line("markers", "qdrant: needs a reachable Qdrant server")


@pytest.fixture(scope="session")
def ds(tmp_path_factory):
    return corpus_mod.build(tmp_path_factory.mktemp("bf_parity"))


# Enough queries that a per-query filter mask, which is bit-packed along the
# query axis, spans a DIFFERENT number of bytes at each of the heights the
# mask-height suite sets up. At 8 queries every height is one byte and reading
# one at the wrong height is undetectable — see `corpus.build`.
WIDE_QUERIES = 26


@pytest.fixture(scope="session")
def ds_wide(tmp_path_factory):
    """Same documents as `ds` (identical seed, corpus drawn first), more
    queries. Reuses the same Qdrant collection, since only the query side
    differs."""
    return corpus_mod.build(tmp_path_factory.mktemp("bf_parity_wide"),
                            n_queries=WIDE_QUERIES)


@pytest.fixture(scope="session")
def oracle_wide(ds_wide):
    return naive.Oracle(ds_wide.docs, ds_wide.queries, ds_wide.date_fields,
                        ds_wide.query_date_fields)


@pytest.fixture(scope="session")
def oracle(ds):
    return naive.Oracle(ds.docs, ds.queries, ds.date_fields, ds.query_date_fields)


@pytest.fixture(scope="session", params=parity_devices())
def device(request):
    """Every parity test runs once per device this machine can answer on —
    `["cpu"]` on a laptop, `["cpu", "cuda"]` on a GPU box. This is the whole
    of what "make it reusable on GPU" needs: no test knows which device it is
    on."""
    return request.param


@pytest.fixture(scope="session")
def matrix_run(ds, device):
    """Every case in `cases.CASES`, computed by nova-bf in one run on
    `device`. `{case_name: {query_index: [(row, score), …]}}`."""
    return runner.run(
        ds,
        [c.spec() for c in cases_mod.CASES],
        out_tag="matrix",
        device=device,
        # Small enough that the 4 corpus files each become several slices, so
        # the per-file batch loop, the top-K merge across slices and the
        # filter's row compaction are all exercised rather than short-circuited
        # by a single whole-file batch.
        params={"dense_batch_size": 37, "sparse_batch_size": 29,
                "multivector_batch_size": 23},
    )


@pytest.fixture(scope="session")
def client():
    try:
        import qdrant_client
    except ImportError as exc:
        _QDRANT.update(probed=True, reachable=False,
                       reason=f"qdrant_client is not installed ({exc})")
        _unavailable()
    try:
        c = qdrant_client.QdrantClient(url=QDRANT_URL, timeout=60)
        c.get_collections()
    except Exception as exc:  # pragma: no cover - environment gate
        _QDRANT.update(probed=True, reachable=False,
                       reason=f"no reachable Qdrant at {QDRANT_URL}: {exc}")
        _unavailable()
    # NOT `reachable=True` here: connecting proves only that a server answered.
    # A vector-config or version mismatch can still make collection creation
    # fail, and the banner would report a green "RAN" over an oracle that
    # graded nothing -- the same "looks like it checked everything" failure this
    # is meant to prevent, moved one step later. `graded` is set by the
    # `collection` fixture, after the data is actually in.
    _QDRANT.update(probed=True, reachable=True, graded=False, reason=None)
    return c


def _unavailable():
    """Skip, or fail outright when the caller demanded the oracle."""
    msg = (f"{_QDRANT['reason']} — the Qdrant oracle did NOT run. "
           f"Start one (docker run -d -p 6333:6333 qdrant/qdrant) or unset "
           f"NOVA_BF_REQUIRE_QDRANT.")
    if _require_qdrant():
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say plainly whether the engine-fidelity oracle ran.

    pytest emits its own `N passed` line LAST, so this necessarily prints just
    above it rather than after it (an earlier comment here claimed otherwise).
    Hence red + bold + a section separator: the count cannot be moved, so the
    banner has to be impossible to miss next to it.
    """
    # `_QDRANT` is a per-process global. Under pytest-xdist the fixtures run in
    # workers while this hook runs on the controller, so `probed` stays False
    # and the banner would silently vanish -- the one thing it exists to
    # prevent. xdist is not currently a dependency; if it is added, hand the
    # state over via `workeroutput`/`workerinput` before trusting this.
    tr = terminalreporter
    # `_require_qdrant()` used to be consulted ONLY inside the `client` fixture,
    # so any run that never instantiated it — `-k`, `--deselect`,
    # `-m "not qdrant"`, a renamed test, or a collection error in a parity
    # module — passed green with the flag set. That is exactly the "reads like a
    # run that checked everything" failure the flag exists to prevent, so the
    # check is session-level and lives here, where none of those route around
    # it.
    #
    # ONE route is still open and cannot be closed from this file: the
    # `importorskip("torch")` above. It skips the whole parity package during
    # conftest collection, so these hooks are never registered and the gate
    # vanishes with them. Closing it means moving this state and these hooks
    # into a parent conftest that does not import torch. Left open deliberately:
    # nova-bf cannot run at all without torch, so a torch-less box has no
    # passing suite to be misread in the first place.
    if _require_qdrant() and not _QDRANT.get("graded"):
        tr.write_sep("=", "Qdrant oracle: REQUIRED BUT DID NOT GRADE",
                     red=True, bold=True)
        tr.write_line("  NOVA_BF_REQUIRE_QDRANT is set, but no case was compared")
        tr.write_line("  against Qdrant in this session"
                      f"{'' if _QDRANT.get('probed') else ' (its fixtures never ran)'}.")
        if _QDRANT.get("reason"):
            tr.write_line(f"  reason: {_QDRANT['reason']}")
        # Flag it here; the exit status is set in `pytest_sessionfinish`, which
        # is the only hook late enough to own it and early enough to be
        # honoured (setting it from this hook is silently ignored — the run
        # still reported exit 0).
        config._nova_bf_qdrant_required_but_absent = True
    if _QDRANT.get("graded"):
        tr.write_sep("=", "Qdrant oracle: RAN", green=True)
        return
    if _QDRANT.get("reachable"):
        # Connected but never loaded a collection: the oracle did not grade.
        tr.write_sep("=", "Qdrant oracle: CONNECTED BUT DID NOT GRADE",
                     red=True, bold=True)
        tr.write_line("  A server answered, but no collection was loaded — so no")
        tr.write_line("  case was compared against the engine. Check for a")
        tr.write_line("  collection-creation or version error above.")
        return
    if not _QDRANT.get("probed"):
        return                      # no Qdrant-backed test was selected at all
    skipped = [r for r in tr.stats.get("skipped", [])
               if "Qdrant" in str(getattr(r, "longrepr", ""))]
    tr.write_sep("=", "Qdrant oracle: DID NOT RUN", red=True, bold=True)
    tr.write_line(f"  {len(skipped)} parity case(s) skipped: {_QDRANT['reason']}")
    tr.write_line("  These check nova-bf against the ENGINE the ground truth")
    tr.write_line("  grades, which the naive oracle cannot substitute for.")
    tr.write_line("  Set NOVA_BF_REQUIRE_QDRANT=1 to make this a failure.")


@pytest.fixture(scope="session")
def collection(client, ds):
    from . import qdrant_ref

    name = qdrant_ref.create_collection(client, ds)
    # Only now has the oracle got data to grade against; see `client`.
    _QDRANT["graded"] = True
    yield name
    client.delete_collection(name)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Turn a required-but-absent Qdrant oracle into a non-zero exit.

    The condition is recomputed here rather than read from a flag set in
    `pytest_terminal_summary`: that hook is invoked from INSIDE the terminal
    reporter's own `sessionfinish`, so it runs after this one and the flag was
    always still unset — the run printed the warning and exited 0, which is the
    exact failure `NOVA_BF_REQUIRE_QDRANT` exists to prevent. `trylast` so the
    status is not overwritten by another implementation.
    """
    if _require_qdrant() and not _QDRANT.get("graded"):
        session.exitstatus = 1
