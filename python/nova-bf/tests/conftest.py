"""Shared fixtures for the nova-bf test suite."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slice_only: a two-pass test of slice-mode-only machinery (padded "
        "GEMM heights, shape verification); skipped in the tiles-mode run")

_MERGE_ENV = (
    "NOVA_BF_MERGE_FORCE", "NOVA_BF_MERGE_WINDOW",
    # Same property as the merge overrides: both are meant to be EXPORTED to
    # get a run through a bad day (throttle the `tiebreak='id'` startup pass,
    # or put it back on whole-file ranged reads), so they linger in the shell
    # afterwards. An exported NOVA_BF_ID_PASS_RANGED_GET is the quiet one --
    # the ids come out identical either way, so every id-pass test would go on
    # reporting green while exercising the whole-file path the change exists
    # to avoid.
    "NOVA_BF_ID_PASS_RANGED_GET", "NOVA_BF_ID_PASS_WORKERS",
)


@pytest.fixture(autouse=True)
def _tf32_flag_is_restored():
    """Every test leaves torch's process-global TF32 matmul flag as it found it.

    Tests set `torch.backends.cuda.matmul.allow_tf32` directly and rely on this
    to restore it BY VALUE. `monkeypatch.setattr` cannot be used for it: on
    torch >= 2.15 the flag is a property of `cuBLASModule`, monkeypatch's undo
    deletes the attribute, that raises at teardown, and TF32 stays ON for every
    later test -- silently changing GEMM bits under the two-pass suites.
    """
    try:
        import torch
        matmul = torch.backends.cuda.matmul
        before = matmul.allow_tf32
    except Exception:  # noqa: BLE001 - no torch / no such flag: nothing to guard
        yield
        return
    yield
    matmul.allow_tf32 = before


@pytest.fixture(autouse=True)
def _merge_env_is_hermetic(monkeypatch):
    """No test inherits the operator's run-rescue overrides.

    `NOVA_BF_MERGE_FORCE` is meant to be EXPORTED for a rescue run, so it will
    be set in the shell of whoever is working on this code -- exactly when
    these tests get run. Exported, it turns six of them red for the wrong
    reason and, more quietly, makes every other merge test in the repo
    exercise the forced path while still reporting green. Tests that want it
    set it themselves.
    """
    for name in _MERGE_ENV:
        monkeypatch.delenv(name, raising=False)
