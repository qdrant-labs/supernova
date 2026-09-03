"""The Qdrant availability gate, which had no tests at all.

A five-mutant sweep against the full live-Qdrant parity suite killed NOTHING:
reverting `_require_qdrant` to `bool(os.environ.get(...))` (the original bug,
where `"0"` meant ON), moving `graded=True` into `client` (green "RAN" before
any data is loaded), removing the `pytest.fail` so the flag only ever skipped,
never setting `graded`, and DELETING THE BANNER ENTIRELY all survived.

The gate exists so that a run which did not compare anything against the engine
cannot look like a run that did. Untested, it was decoration.
"""
from __future__ import annotations

import pytest

from _pytest.outcomes import Failed, Skipped

from . import conftest as gate


@pytest.mark.parametrize("value,required", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("FALSE", False), (" 0 ", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("yes", True), ("on", True),
    # fail CLOSED: an unrecognised value enables the strict behaviour rather
    # than silently disabling the check someone set to guarantee something
    ("none", True), ("disabled", True),
])
def test_require_qdrant_parses_the_flag(monkeypatch, value, required):
    """`bool("0")` is True in Python, so the obvious spelling turned the
    STRICTEST behaviour on for a CI job that set the flag to 0 to mean off."""
    if value is None:
        monkeypatch.delenv("NOVA_BF_REQUIRE_QDRANT", raising=False)
    else:
        monkeypatch.setenv("NOVA_BF_REQUIRE_QDRANT", value)
    assert gate._require_qdrant() is required


def test_require_qdrant_is_read_at_use_time(monkeypatch):
    """Captured at import, a test could never change it — and neither could a
    caller that sets the env after import."""
    monkeypatch.setenv("NOVA_BF_REQUIRE_QDRANT", "1")
    assert gate._require_qdrant() is True
    monkeypatch.setenv("NOVA_BF_REQUIRE_QDRANT", "0")
    assert gate._require_qdrant() is False


class _FakeTR:
    """Enough of pytest's terminal reporter to capture what the banner writes."""

    def __init__(self, skipped=()):
        self.lines: list[str] = []
        self.seps: list[tuple[str, dict]] = []
        self.stats = {"skipped": list(skipped)}

    def write_sep(self, _ch, title, **kw):
        self.seps.append((title, kw))

    def write_line(self, line):
        self.lines.append(line)


def _summary(monkeypatch, *, probed, reachable, graded, reason=None,
             required=False, skipped=()):
    monkeypatch.setitem(gate._QDRANT, "probed", probed)
    monkeypatch.setitem(gate._QDRANT, "reachable", reachable)
    monkeypatch.setitem(gate._QDRANT, "graded", graded)
    monkeypatch.setitem(gate._QDRANT, "reason", reason)
    if required:
        monkeypatch.setenv("NOVA_BF_REQUIRE_QDRANT", "1")
    else:
        monkeypatch.delenv("NOVA_BF_REQUIRE_QDRANT", raising=False)
    tr = _FakeTR(skipped)

    class _Cfg:
        pass

    gate.pytest_terminal_summary(tr, 0, _Cfg())
    return tr


def test_banner_says_ran_only_when_something_was_graded(monkeypatch):
    tr = _summary(monkeypatch, probed=True, reachable=True, graded=True)
    assert [t for t, _ in tr.seps] == ["Qdrant oracle: RAN"]


def test_banner_distinguishes_connected_from_graded(monkeypatch):
    """Connecting proves a server answered, NOT that any case was compared: a
    collection-creation or version failure would otherwise print green RAN over
    an oracle that graded nothing."""
    tr = _summary(monkeypatch, probed=True, reachable=True, graded=False)
    title = tr.seps[0][0]
    assert "DID NOT GRADE" in title, title
    assert tr.seps[0][1].get("red") is True


def test_banner_reports_unreachable_with_the_reason(monkeypatch):
    tr = _summary(monkeypatch, probed=True, reachable=False, graded=False,
                  reason="no reachable Qdrant at http://localhost:6399")
    assert "DID NOT RUN" in tr.seps[0][0]
    assert any("6399" in ln for ln in tr.lines), tr.lines


def test_banner_is_silent_when_no_qdrant_test_was_selected(monkeypatch):
    """Nothing probed means the Qdrant tests were not part of this run at all;
    a banner then would be a false alarm."""
    tr = _summary(monkeypatch, probed=False, reachable=None, graded=False)
    assert tr.seps == [] and tr.lines == []


def test_banner_fires_when_required_even_if_nothing_probed(monkeypatch):
    """The flag used to be consulted ONLY inside the `client` fixture, so any
    run that never instantiated it — `-k`, `--deselect`, a renamed test, or the
    `importorskip("torch")` at the top of conftest — passed green with the flag
    set. Verified: `NOVA_BF_REQUIRE_QDRANT=1 pytest tests/parity -k "not qdrant"`
    reported 370 passed, exit 0, no banner."""
    tr = _summary(monkeypatch, probed=False, reachable=None, graded=False,
                  required=True)
    assert any("REQUIRED BUT DID NOT GRADE" in t for t, _ in tr.seps), tr.seps


def test_session_exit_status_is_nonzero_when_required_but_absent(monkeypatch):
    """`pytest_terminal_summary` CANNOT change the exit status, and neither can
    a flag it sets — that hook runs from inside the terminal reporter's own
    `sessionfinish`, so it fires after ours. Both were tried; both exited 0."""
    monkeypatch.setenv("NOVA_BF_REQUIRE_QDRANT", "1")
    monkeypatch.setitem(gate._QDRANT, "graded", False)

    class _S:
        exitstatus = 0

        class config:
            pass

    s = _S()
    gate.pytest_sessionfinish(s, 0)
    assert s.exitstatus == 1

    monkeypatch.setitem(gate._QDRANT, "graded", True)
    s2 = _S()
    gate.pytest_sessionfinish(s2, 0)
    assert s2.exitstatus == 0, "a graded run must not be failed"


# --------------------------------------------------------------------------
# the fixtures themselves
#
# The tests above patch `_QDRANT` directly, so they cannot see WHERE `graded`
# gets set or whether `_unavailable` fails vs skips. A mutation sweep proved
# it: moving `graded=True` into `client`, never setting it, and deleting the
# `pytest.fail` all survived. These exercise the fixture bodies.
# --------------------------------------------------------------------------
def test_unavailable_skips_when_not_required(monkeypatch):
    monkeypatch.delenv("NOVA_BF_REQUIRE_QDRANT", raising=False)
    monkeypatch.setitem(gate._QDRANT, "reason", "simulated: server down")
    # `Skipped`/`Failed` derive from BaseException, so `raises(Exception)` does
    # NOT catch them — it lets the outcome propagate and the test quietly
    # becomes a skip instead of asserting anything.
    with pytest.raises(BaseException) as ei:
        gate._unavailable()
    assert isinstance(ei.value, Skipped), \
        f"expected a skip, got {type(ei.value).__name__}"


def test_unavailable_fails_hard_when_required(monkeypatch):
    """Without this, the flag only ever downgraded to a skip — which is
    indistinguishable from not setting it at all."""
    monkeypatch.setenv("NOVA_BF_REQUIRE_QDRANT", "1")
    monkeypatch.setitem(gate._QDRANT, "reason", "simulated: server down")
    with pytest.raises(BaseException) as ei:
        gate._unavailable()
    # NOT `raises(Failed)`: a `Skipped` is also a BaseException, so it would
    # escape and pytest would mark this test SKIPPED rather than failed —
    # exactly the silent pass the whole gate exists to prevent.
    assert isinstance(ei.value, Failed), \
        f"expected a hard failure, got {type(ei.value).__name__}"
    assert "did NOT run" in str(ei.value)


def test_client_connecting_does_not_by_itself_mean_graded(monkeypatch):
    """`client` succeeding proves a server ANSWERED. A vector-config or version
    mismatch can still make collection creation fail, and reporting green there
    is the exact failure this gate exists to prevent."""
    import sys
    import types

    fake = types.ModuleType("qdrant_client")

    class _C:
        def __init__(self, *a, **kw): pass
        def get_collections(self): return []

    fake.QdrantClient = _C
    monkeypatch.setitem(sys.modules, "qdrant_client", fake)
    monkeypatch.setitem(gate._QDRANT, "graded", False)
    monkeypatch.setitem(gate._QDRANT, "reachable", None)

    gate.client.__wrapped__()
    assert gate._QDRANT["reachable"] is True, "should record that it connected"
    assert gate._QDRANT["graded"] is False, \
        "connecting must NOT count as grading — that is the green-RAN hole"


def test_collection_is_what_marks_the_oracle_as_graded(monkeypatch):
    """The flip side of the test above: `graded` must be set SOMEWHERE, or the
    banner can never say RAN and the gate cries wolf on every healthy run."""
    from . import qdrant_ref

    monkeypatch.setitem(gate._QDRANT, "graded", False)
    monkeypatch.setattr(qdrant_ref, "create_collection", lambda client, ds: "coll")

    class _Cli:
        def delete_collection(self, name): pass

    # `collection` is a yield-fixture: calling it only BUILDS the generator, so
    # the body has to be advanced or nothing runs and the assert is vacuous.
    gen = gate.collection.__wrapped__(client=_Cli(), ds=object())
    assert next(gen) == "coll"
    assert gate._QDRANT["graded"] is True, \
        "the collection fixture must be what marks the oracle as graded"
    for _ in gen:  # run the teardown half too
        pass
