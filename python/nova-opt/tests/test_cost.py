import pytest

from nova_opt.cost import ArtifactCache, CostModel
from nova_opt.space import Candidate, Index, Layout, Quant, Search

PRIORS = {"layout": 600.0, "index": 300.0, "quant": 120.0, "search": 30.0}


def cand(ef_search=64) -> Candidate:
    return Candidate(
        layout=Layout(segments=8),
        index=Index(m=16, ef_construct=128),
        quant=Quant(variant="scalar_default"),
        search=Search(ef_search=ef_search),
    )


def test_marginal_cost_tiers():
    model = CostModel(priors=PRIORS)
    cache = ArtifactCache()
    c = cand()

    assert model.marginal_cost(c, cache) == pytest.approx(1050.0)  # everything
    cache.add(c, ("layout",))
    assert model.marginal_cost(c, cache) == pytest.approx(450.0)  # index+quant+search
    cache.add(c, ("index",))
    assert model.marginal_cost(c, cache) == pytest.approx(150.0)  # quant+search
    cache.add(c, ("quant",))
    assert model.marginal_cost(c, cache) == pytest.approx(30.0)  # search only


def test_search_sibling_reuses_artifacts():
    cache = ArtifactCache()
    cache.add(cand(), ("layout", "index", "quant"))
    sibling = cand(ef_search=256)
    assert cache.missing_levels(sibling) == ("search",)


def test_missing_prefix_invalidates_suffix():
    cache = ArtifactCache()
    c = cand()
    # a quant artifact recorded without its layout must NOT shortcut the cost
    cache.current_quant[c.layout_key] = c.quant_key
    assert cache.missing_levels(c) == ("layout", "index", "quant", "search")


def test_stale_index_on_same_layout_forces_rebuild():
    """A layout's collection is mutated in place: building index B on a
    layout that previously had index A built must NOT be treated as if A
    were still live — the physical collection now holds B."""
    cache = ArtifactCache()
    a = cand()
    b = index_cand(m=32)  # same layout, different index_key
    cache.add(a, ("layout", "index", "quant"))
    assert cache.missing_levels(b) == ("index", "quant", "search")
    cache.add(b, ("index", "quant"))
    # revisiting `a` must rebuild — the collection currently holds `b`
    assert cache.missing_levels(a) == ("index", "quant", "search")


def test_stale_quant_on_same_index_forces_rebuild():
    """Same idea one level down: two quant variants sharing an index_key
    must not both look 'live' at once."""
    cache = ArtifactCache()
    c = cand()  # quant_variant="scalar_default"
    other = Candidate(
        layout=c.layout, index=c.index, quant=Quant(variant="none"),
        search=c.search,
    )
    cache.add(c, ("layout", "index", "quant"))
    assert cache.missing_levels(other) == ("quant", "search")
    cache.add(other, ("quant",))
    assert cache.missing_levels(c) == ("quant", "search")


def test_ema_observation_moves_estimate():
    model = CostModel(priors=PRIORS, alpha=0.5)
    model.observe("index", 100.0)
    assert model.estimate("index") == pytest.approx(200.0)
    model.observe("index", 100.0)
    assert model.estimate("index") == pytest.approx(150.0)
    # non-positive observations are ignored
    model.observe("index", 0.0)
    assert model.estimate("index") == pytest.approx(150.0)


def test_missing_prior_rejected():
    with pytest.raises(ValueError, match="missing levels"):
        CostModel(priors={"layout": 1.0})


def test_non_positive_prior_rejected():
    bad = dict(PRIORS, quant=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        CostModel(priors=bad)


def index_cand(m, ef_construct=128) -> Candidate:
    return Candidate(
        layout=Layout(segments=8),
        index=Index(m=m, ef_construct=ef_construct),
        quant=Quant(variant="none"),
        search=Search(ef_search=64),
    )


def test_cost_regression_learns_config_scaling():
    """Index-build time scaling with m must show up in the estimates once a
    few configs have been observed — a flat constant would misprice
    candidates at the edges."""
    model = CostModel(priors=PRIORS, alpha=0.5)
    # build time proportional to m
    for m, secs in [(8, 80.0), (16, 160.0), (32, 320.0), (16, 160.0)]:
        model.observe("index", secs, index_cand(m))
    small = model.estimate("index", index_cand(8))
    big = model.estimate("index", index_cand(64))
    assert big > small
    assert small == pytest.approx(80.0, rel=0.35)


def test_cost_regression_clamped_to_ema_band():
    model = CostModel(priors=PRIORS, alpha=0.5)
    for m, secs in [(8, 80.0), (16, 160.0), (32, 320.0)]:
        model.observe("index", secs, index_cand(m))
    ema = model.estimate("index")
    # a wild extrapolation (m=1024) cannot leave the ema band
    extreme = model.estimate("index", index_cand(1024))
    assert extreme <= ema * 5.0 + 1e-9


def test_regression_needs_feature_variation():
    model = CostModel(priors=PRIORS, alpha=1.0)
    for _ in range(5):
        model.observe("index", 100.0, index_cand(16))
    # identical configs -> no regression; falls back to the EMA
    assert model.estimate("index", index_cand(64)) == pytest.approx(100.0)
