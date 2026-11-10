"""`NOVA_BF_NO_GPU_COMBO`: the lazy per-query text combine is invisible.

With the switch unset, per-query text filters reach scoring as a
`filters.ComboRowMask` combined on the device per slice; with it set, as the
materialised `PackedRowMask`. Both runs must agree hit-for-hit AND
score-for-score (the masks are bit-identical, so nothing about scoring may
move), and both must agree with the naive oracle — so a combine that broke
both arms the same way still fails. Every modality, three text shapes: the
plain per-query phrase, a should group mixing it with a static condition
(1-D `rest_or`), and a must_not phrase under a static must (1-D `keep`).
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare
from .runner import run, spec

TEXT_FILTERS = {
    "pqtext": cases_mod.FILTERS["pqtext"],
    "pqtext_should": {"should": [
        {"field": "title", "match_text_from_query": "q_phrase"},
        {"field": "category", "match": "news"}]},
    "pqtext_not": {
        "must": [{"field": "category", "match": ["news", "blog", "paper"]}],
        "must_not": [{"field": "title", "match_text_from_query": "q_phrase"}]},
}
CASES = [cases_mod.Case(vt, m, fname, fdict)
         for vt, m in cases_mod.MODALITIES
         for fname, fdict in TEXT_FILTERS.items()]



@pytest.fixture(autouse=True)
def _no_parts_guard(monkeypatch):
    """Small test query sets have more parts than queries; the memory guard
    would send them to the reference loop. Drive the lazy path instead (the
    guard itself is tested in test_combo_row_mask.py)."""
    from nova_bf import filters as _F
    monkeypatch.setattr(_F, "_COMBO_PARTS_GUARD", False)


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)


@pytest.fixture(scope="module", params=[None, 1], ids=["whole", "windowed"])
def both_arms(request, ds, device):
    """The lazy arm runs with `parts` held on the device whole and in
    one-column windows (reloaded as slices move)."""
    mp = pytest.MonkeyPatch()
    try:
        from nova_bf import compute as _C
        mp.setattr(_C, "_RESIDENT_WINDOW_BYTES", request.param)
        specs = [c.spec() for c in CASES]
        mp.setenv("NOVA_BF_NO_GPU_COMBO", "1")
        off = run(ds, specs, out_tag=f"combo_off_{request.param}", device=device)
        mp.delenv("NOVA_BF_NO_GPU_COMBO")
        on = run(ds, specs, out_tag=f"combo_on_{request.param}", device=device)
    finally:
        mp.undo()
    return on, off


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_gpu_combo_is_identical_and_oracle_exact(case, ds, oracle, device, both_arms):
    on, off = both_arms
    if device == "cuda" and case.vector_type == "multivector":
        # The multivector CUDA scoring is not bit-reproducible run to run
        # (measured on an A10G 2026-11-10: two runs with the switch OFF differ
        # by up to ~4e-6 in score, same ids and order), so exact equality
        # cannot be asked of on vs off there: use the boundary-aware parity
        # comparison (same hits outside the tie zone, scores within TOL). The
        # masks themselves are pinned bit-for-bit in test_combo_row_mask.py.
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(
                on[case.name][qi], off[case.name][qi], metric=case.metric,
                label=f"[{device}] combo on vs off {case.id} q{qi}")
    else:
        assert on[case.name] == off[case.name], (
            f"[{device}] {case.id}: NOVA_BF_NO_GPU_COMBO changed the output")
    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            on[case.name][qi], want[qi], metric=case.metric,
            label=f"[{device}] gpu combo {case.id} q{qi}")
