"""The text scan's BATCHING, which no corpus in this repo is big enough to reach.

`filters._token_row_masks` splits its tokenisation into batches sized by
`_scan_batch_rows`, and those batches run CONCURRENTLY, writing into one shared
packed grid. The correctness of that split rests on a single property — every
batch but the last owns whole BYTES of the grid — because a boundary inside a
byte puts two threads in the same byte and the resulting mask corruption is
silent.

That property is currently argued, not exercised. `_scan_batch_rows` is floored
at 4096 rows; the parity corpus is 300 rows over four files, and the largest
corpus anywhere in the suite is 4000 rows. So the multi-batch path never runs:
`tests/test_filter_optimizations.py` checks the FORMULA and nothing checks the
scan it feeds. This is the same shape as the int32-offset guard that
`tests/test_multivector_int32_offsets.py` exists for — a predicate everyone
tests and a fallback nobody executes.

These force the split by shrinking the batch size rather than by growing the
corpus (a >4096-row parity corpus would slow every other file here for one
path), and then insist the answer is unchanged: against the single-batch run,
and against the naive oracle, so two nova-bf runs agreeing while both being
wrong still fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import cases as cases_mod
from . import compare
from .runner import run

# Text filters only — they are the ones that tokenise, so they are the only
# cases the scan batching can affect. Both the static and the per-query form,
# across all three modalities, because the mask is consumed differently by each.
TEXT_PROBES = [
    cases_mod.CASES_BY_NAME[n] for n in (
        "dedot_matchtext", "decos_pqtext",
        "spdot_matchtext", "spcos_pqtext",
        "mudot_matchtext", "mucos_pqtext",
    )
]
TEXT_IDS = [c.id for c in TEXT_PROBES]


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)


@pytest.fixture
def forced_batches(monkeypatch):
    """Make every text scan split, and record the sizes it was handed.

    Patches the SIZING helper rather than the corpus: `_token_row_masks` looks
    `_scan_batch_rows` up as a module global at call time, so this reaches the
    real scan with everything else — the thread pool, the packed grid, the
    concurrent writes — untouched.

    Returns the list of batch sizes actually used, so a test can assert the
    split really happened. Without that premise a green result would be
    indistinguishable from "the corpus was too small to split", which is
    exactly the state this file exists to escape.
    """
    from nova_bf import filters

    used: list[int] = []

    def small(col_nbytes, n_rows, width, n_tokens=1):
        # 8 rows: the smallest legal batch (one whole byte of the packed grid),
        # so a 97-row file becomes 13 batches rather than 1.
        used.append(8)
        return 8

    monkeypatch.setattr(filters, "_scan_batch_rows", small)
    return used


@pytest.mark.parametrize("case", TEXT_PROBES, ids=TEXT_IDS)
def test_a_split_text_scan_gives_the_same_answer(case, ds, oracle, device,
                                                 forced_batches):
    """The whole point: batching the tokenisation must be invisible.

    Compared against the ORACLE, not against another nova-bf run, so a split
    that corrupted the mask in a way both runs shared still fails.
    """
    got = run(ds, [case.spec()], out_tag=f"split_{case.name}", device=device)[case.name]
    assert forced_batches, "the scan never ran — this case does not tokenise"

    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] split-scan {case.id} q{qi}")


@pytest.mark.parametrize("case", TEXT_PROBES, ids=TEXT_IDS)
def test_a_split_scan_equals_the_unsplit_one_exactly(case, ds, device, monkeypatch):
    """Stronger than agreeing with the oracle: the two runs must produce the
    SAME ROWS, not merely two rankings the oracle tolerates.

    A filter mask is boolean — there is no rounding to hide behind — so any
    difference here is a real difference in which documents survived, and
    exact equality is the honest assertion.
    """
    from nova_bf import filters

    one = run(ds, [case.spec()], out_tag=f"unsplit_{case.name}", device=device)[case.name]

    used: list[int] = []

    def small(col_nbytes, n_rows, width, n_tokens=1):
        used.append(8)
        return 8

    monkeypatch.setattr(filters, "_scan_batch_rows", small)
    many = run(ds, [case.spec()], out_tag=f"split2_{case.name}", device=device)[case.name]
    assert used, "the scan never ran — this case does not tokenise"

    for qi in range(len(ds.queries)):
        assert [r for r, _ in one[qi]] == [r for r, _ in many[qi]], (
            f"[{device}] {case.id} q{qi}: splitting the text scan changed which "
            f"documents matched")


@pytest.mark.parametrize("batch_rows", [8, 16, 24, 40])
def test_every_batch_size_gives_one_mask(batch_rows, ds, device, monkeypatch):
    """Several splits, including ones that do not divide the file sizes.

    The parity files are 97/61/79/63 rows — deliberately none a multiple of any
    batch size — so each of these leaves a short final batch, which is the
    boundary the byte-alignment argument is really about.
    """
    from nova_bf import filters

    case = cases_mod.CASES_BY_NAME["dedot_matchtext"]
    monkeypatch.setattr(filters, "_scan_batch_rows",
                        lambda *a, **k: batch_rows)
    got = run(ds, [case.spec()], out_tag=f"bs{batch_rows}", device=device)[case.name]

    monkeypatch.undo()
    want = run(ds, [case.spec()], out_tag=f"bs{batch_rows}_ref", device=device)[case.name]
    for qi in range(len(ds.queries)):
        assert [r for r, _ in got[qi]] == [r for r, _ in want[qi]], (
            f"[{device}] batch_rows={batch_rows} q{qi} changed the matched set")


def test_a_misaligned_batch_size_is_re_imposed(ds, device, monkeypatch):
    """`_token_row_masks` re-applies `max(8, batch_rows & ~7)` to whatever the
    helper returned, rather than trusting it.

    That line is unreachable today — the helper always returns a multiple of 8
    — so nothing proves it works. Hand the scan a deliberately misaligned size
    and the answer must still be right: if the re-imposition were dropped, two
    concurrent batches would share a byte of the packed grid.
    """
    from nova_bf import filters

    case = cases_mod.CASES_BY_NAME["dedot_matchtext"]
    want = run(ds, [case.spec()], out_tag="align_ref", device=device)[case.name]

    for bad in (9, 13, 7, 1):
        monkeypatch.setattr(filters, "_scan_batch_rows", lambda *a, **k: bad)
        got = run(ds, [case.spec()], out_tag=f"align{bad}", device=device)[case.name]
        monkeypatch.undo()
        for qi in range(len(ds.queries)):
            assert [r for r, _ in got[qi]] == [r for r, _ in want[qi]], (
                f"[{device}] a misaligned batch size ({bad}) changed the matched "
                f"set at q{qi} — the `& ~7` re-imposition is not holding")


def test_the_sizer_never_returns_a_misaligned_batch():
    """The property the concurrency argument rests on, over the real helper.

    Every batch but the last must own whole bytes of the packed grid, so the
    size has to be a multiple of 8 for every input — including the ones where a
    different bound wins.
    """
    from nova_bf.filters import _scan_batch_rows

    rng = np.random.default_rng(0)
    for _ in range(2000):
        n_rows = int(rng.integers(1, 5_000_000))
        col_nbytes = int(rng.integers(1, 1 << 32))
        width = int(rng.integers(1, 129))
        n_tokens = int(rng.integers(1, 100_000))
        got = _scan_batch_rows(col_nbytes, n_rows, width, n_tokens)
        assert got % 8 == 0, (
            f"batch of {got} rows for n_rows={n_rows} col_nbytes={col_nbytes} "
            f"width={width} n_tokens={n_tokens} splits a byte of the packed grid")
        assert got >= 8, f"non-positive batch {got}"


@pytest.mark.parametrize("case", TEXT_PROBES[:3], ids=TEXT_IDS[:3])
def test_the_serial_and_concurrent_scans_agree(case, ds, device, monkeypatch):
    """Run the SAME split serially and concurrently; the masks must match.

    `_token_row_masks` picks between three executions —
    `workers = min(16, os.cpu_count(), len(offsets))`, then a shared pool, its
    own `ThreadPoolExecutor`, or a plain loop. Everything else in this file
    exercises a concurrent one, because a test box has cores and the split is
    forced. The serial loop is the odd branch out, and it is also the ONLY
    reference that cannot itself be racy.

    So this is the direct test for the data race the byte-alignment rule
    exists to prevent: if concurrent batches ever shared a byte of the packed
    grid, the concurrent answer would drift from the serial one. Comparing
    them is stronger than comparing either against the oracle, because a race
    that corrupted a row the filter excludes anyway would still show up here.
    """
    from nova_bf import filters

    monkeypatch.setattr(filters, "_scan_batch_rows", lambda *a, **k: 8)

    # Serial: one core, so `workers` collapses to 1 and the loop runs in-thread.
    monkeypatch.setattr(filters.os, "cpu_count", lambda: 1)
    serial = run(ds, [case.spec()], out_tag=f"serial_{case.name}",
                 device=device)[case.name]

    # Concurrent: many workers over the same 8-row batches.
    monkeypatch.setattr(filters.os, "cpu_count", lambda: 16)
    concurrent = run(ds, [case.spec()], out_tag=f"conc_{case.name}",
                     device=device)[case.name]

    for qi in range(len(ds.queries)):
        assert [r for r, _ in serial[qi]] == [r for r, _ in concurrent[qi]], (
            f"[{device}] {case.id} q{qi}: the concurrent scan disagreed with the "
            f"serial one — concurrent batches are corrupting each other's bytes")
