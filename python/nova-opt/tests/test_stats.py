import numpy as np
import pytest

from nova_opt.stats import (
    StatsParams,
    base_embedding_stats,
    load_matrix,
    query_stats,
    resolve_metric,
)

PARAMS = StatsParams(
    sample_size=50,
    pair_sample_size=2000,
    nn_query_sample_size=32,
    nn_reference_sample_size=200,
    knn_k=100,
    seed=0,
)


@pytest.fixture(scope="module")
def gaussian():
    rng = np.random.default_rng(42)
    return rng.normal(size=(500, 16))


def test_metric_aliases():
    assert resolve_metric("L2") == "euclidean"
    assert resolve_metric("COSINE") == "cosine"
    assert resolve_metric("IP") == "dot"
    with pytest.raises(ValueError, match="unknown distance metric"):
        resolve_metric("hamming")


def test_base_stats_exact_norms(gaussian):
    feats, meta = base_embedding_stats(
        gaussian, total_rows=500, full_pass=True, metric="euclidean", params=PARAMS
    )
    norms = np.linalg.norm(gaussian, axis=1)
    assert feats["number_of_embeddings"] == 500
    assert feats["dimensionality"] == 16
    assert feats["embedding_norm_mean"] == pytest.approx(norms.mean())
    assert feats["embedding_norm_min"] == pytest.approx(norms.min())
    assert feats["embedding_norm_max"] == pytest.approx(norms.max())
    assert meta["norm_stats"] == "exact"
    assert meta["pairwise"] == "sampled"
    assert meta["sampling"]["seed"] == 0


def test_base_stats_sampled_flag(gaussian):
    _, meta = base_embedding_stats(
        gaussian[:100], total_rows=500, full_pass=False, metric="euclidean",
        params=PARAMS,
    )
    assert meta["norm_stats"] == "sampled"
    assert meta["norm_rows_used"] == 100


def test_knn_percentiles_ordered(gaussian):
    feats, _ = base_embedding_stats(
        gaussian, total_rows=500, full_pass=True, metric="euclidean", params=PARAMS
    )
    # rank-k neighbor distance grows with k; p95 >= p50 within a rank
    assert feats["embedding_nn1_dist_p50"] <= feats["embedding_nn10_dist_p50"]
    assert feats["embedding_nn10_dist_p50"] <= feats["embedding_nn100_dist_p50"]
    assert feats["embedding_nn1_dist_p50"] <= feats["embedding_nn1_dist_p95"]
    assert feats["embedding_pairwise_distance_p05"] <= feats["embedding_pairwise_distance_p95"]
    assert feats["embedding_intrinsic_dimensionality"] > 0


def test_intrinsic_dim_tracks_true_dimension():
    rng = np.random.default_rng(0)
    low = rng.normal(size=(800, 2)) @ rng.normal(size=(2, 32))  # 2-d manifold in 32-d
    high = rng.normal(size=(800, 32))
    f_low, _ = base_embedding_stats(
        low, total_rows=800, full_pass=True, metric="euclidean", params=PARAMS
    )
    f_high, _ = base_embedding_stats(
        high, total_rows=800, full_pass=True, metric="euclidean", params=PARAMS
    )
    assert (
        f_low["embedding_intrinsic_dimensionality"]
        < f_high["embedding_intrinsic_dimensionality"]
    )


def test_query_stats_duplicates_and_pca(gaussian):
    q = np.vstack([gaussian[:100], gaussian[:50]])  # 50 duplicated rows
    feats, meta = query_stats(
        q, total_rows=len(q), full_pass=True, metric="cosine", params=PARAMS
    )
    assert feats["query_count"] == 150
    assert feats["query_duplicate_rate"] == pytest.approx(50 / 150)
    assert 0 < feats["query_pca_top1_var_ratio"] <= 1
    assert feats["query_pca_top1_var_ratio"] <= feats["query_pca_top10_var_ratio"] <= 1
    assert feats["query_intrinsic_dim_estimate"] > 0
    assert meta["pca"] == "sampled"


def test_dot_metric_distances_are_negated_similarity(gaussian):
    feats, meta = query_stats(
        gaussian, total_rows=500, full_pass=True, metric="dot", params=PARAMS
    )
    # dot "distance" = -similarity: p05 still <= p95 by construction
    assert feats["query_pairwise_distance_p05"] <= feats["query_pairwise_distance_p95"]
    # TwoNN needs a real non-negative metric distance -- dot's is neither,
    # so it must fall back rather than silently produce NaN (see below)
    assert not np.isnan(feats["query_intrinsic_dim_estimate"])
    assert meta["twonn_metric"] == "euclidean"


def test_twonn_not_nan_under_dot_metric_even_when_all_pairs_are_negative():
    """Regression: for embeddings with nonzero mean (typical of real
    corpora), a genuine nearest neighbor by dot/IP has a large positive
    inner product, so `-inner_product` (the "distance" used to keep
    "smaller is closer" uniform across metrics) is negative for essentially
    every row -- `_intrinsic_dim_twonn`'s `d1 > 0` validity check used to
    reject all of them, making this feature NaN for every dot-metric
    workload, silently."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 16)) + 5.0
    knn_params = StatsParams(
        sample_size=50, pair_sample_size=2000,
        nn_query_sample_size=64, nn_reference_sample_size=500,
        knn_k=20, seed=0,
    )
    feats, meta = base_embedding_stats(
        x, total_rows=1000, full_pass=True, metric="dot", params=knn_params
    )
    assert not np.isnan(feats["embedding_intrinsic_dimensionality"])
    assert feats["embedding_intrinsic_dimensionality"] > 0
    assert meta["twonn_metric"] == "euclidean"


def test_twonn_metric_passthrough_for_euclidean_and_cosine():
    from nova_opt.stats import _twonn_metric

    assert _twonn_metric("euclidean") == "euclidean"
    assert _twonn_metric("cosine") == "cosine"
    assert _twonn_metric("dot") == "euclidean"


def test_load_matrix_npy_full_and_sampled(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 8)).astype(np.float32)
    path = tmp_path / "x.npy"
    np.save(path, x)

    full, total, sampled = load_matrix(str(path), max_rows=200)
    assert total == 100 and not sampled and full.shape == (100, 8)

    sub, total, sampled = load_matrix(str(path), max_rows=40, seed=7)
    assert total == 100 and sampled and sub.shape == (40, 8)


def test_load_matrix_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 4))
    table = pa.table({"emb": [row.tolist() for row in x]})
    dest = tmp_path / "x.parquet"
    pq.write_table(table, dest)

    mat, total, sampled = load_matrix(str(dest), column="emb", max_rows=100)
    assert total == 60 and not sampled
    np.testing.assert_allclose(mat, x)
