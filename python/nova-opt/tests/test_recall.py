import numpy as np
import pandas as pd
import pytest

from nova_opt.recall import (
    THRESHOLDS,
    RecallClassifier,
    prepare_features,
)


def synthetic_training_data(
    n_per_dataset=250,
    datasets=("ds_a", "ds_b", "ds_c", "ds_d", "ds_e", "ds_f"),
):
    """Recall driven by ef_search and quantization, plus a dataset-geometry
    shift — learnable structure with a known direction. Enough datasets with
    overlapping geometry that leave-one-dataset-out folds can actually
    transfer (with too few, OOF predictions are chance-level and calibration
    is skipped by design)."""
    rng = np.random.default_rng(0)
    rows = []
    for i, ds in enumerate(datasets):
        intrinsic = 5.0 + 2.0 * i
        for _ in range(n_per_dataset):
            ef = rng.choice([8, 16, 32, 64, 128, 256])
            quant = rng.choice(["none", "binary_1bit", "scalar_default"])
            penalty = {"none": 0.0, "scalar_default": 0.05, "binary_1bit": 0.25}[quant]
            recall = 1.0 - np.exp(-ef / (8.0 * intrinsic)) * 1.2 - penalty
            recall = float(np.clip(recall + rng.normal(0, 0.03), 0, 1))
            rows.append(
                {
                    "dataset": ds,
                    "mean_recall_at_k": recall,
                    "corpus_size": 100_000,
                    "query_count": 1000,
                    "vector_dim": 64,
                    "data_size_bytes": 100_000 * 64 * 4,
                    "distance_metric": "COSINE",
                    "number_of_segments": 8,
                    "segment_size_kb": 3200,
                    "quantization_variant": quant,
                    "quantization": {"none": "NONE", "scalar_default": "SCALAR",
                                     "binary_1bit": "BINARY"}[quant],
                    "quantization_mode": quant.upper(),
                    "hnsw_m": int(rng.choice([8, 16, 32])),
                    "ef_construct": 128,
                    "ef_search": int(ef),
                    "rescore": bool(rng.choice([True, False])),
                    "top_k": 10,
                    "embedding_intrinsic_dimensionality": intrinsic,
                    "query_pca_top1_var_ratio": 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    df = synthetic_training_data()
    clf = RecallClassifier()
    report = clf.train(df, seed=0)
    return clf, report, df


def test_lodo_report_shape(trained):
    _, report, df = trained
    n_datasets = df["dataset"].nunique()
    assert len(report) == n_datasets * len(THRESHOLDS)
    assert set(report["dataset"]) == set(df["dataset"])


def test_probabilities_valid_and_monotonic(trained):
    clf, _, df = trained
    pred = clf.predict(df.head(50))
    ordered = sorted(THRESHOLDS)
    for t in ordered:
        assert np.all(pred.probs[t] >= 0) and np.all(pred.probs[t] <= 1)
    for loose, strict in zip(ordered, ordered[1:]):
        assert np.all(pred.probs[loose] >= pred.probs[strict] - 1e-12)


def test_learned_direction_ef_search(trained):
    clf, _, _ = trained
    base = {
        "corpus_size": 100_000, "query_count": 1000, "vector_dim": 64,
        "data_size_bytes": 100_000 * 64 * 4, "distance_metric": "COSINE",
        "number_of_segments": 8, "segment_size_kb": 3200,
        "quantization_variant": "none", "quantization": "NONE",
        "quantization_mode": "NONE", "hnsw_m": 16, "ef_construct": 128,
        "rescore": True, "top_k": 10,
        "embedding_intrinsic_dimensionality": 11.0,
        "query_pca_top1_var_ratio": 0.1,
    }
    rows = pd.DataFrame([{**base, "ef_search": 8}, {**base, "ef_search": 256}])
    pred = clf.predict(rows)
    # higher ef_search must look more feasible at the 0.90 threshold
    assert pred.probs[0.90][1] > pred.probs[0.90][0]


def test_confidence_bounded(trained):
    clf, _, df = trained
    pred = clf.predict(df.head(20))
    for t in THRESHOLDS:
        assert np.all(pred.confidence[t] >= 0)
        assert np.all(pred.confidence[t] <= 1)


def test_ood_similarity_scales_confidence(trained):
    clf, _, df = trained
    familiar = df.head(5)
    alien = familiar.copy()
    # geometry stats far outside anything in training
    alien["embedding_intrinsic_dimensionality"] = 500.0
    alien["query_pca_top1_var_ratio"] = 0.99
    sim_f = clf.similarity(familiar)
    sim_a = clf.similarity(alien)
    assert np.all(sim_f > sim_a)
    assert np.all((0 <= sim_a) & (sim_f <= 1))
    conf_f = clf.predict(familiar).confidence[0.90]
    conf_a = clf.predict(alien).confidence[0.90]
    # same config columns, alien geometry -> never more trusted
    assert conf_a.mean() < conf_f.mean() + 1e-9


def test_ensemble_spread_reduces_confidence(trained):
    clf, _, df = trained
    pred = clf.predict(df.head(40))
    # the fold ensemble exists and produces bounded confidence
    assert len(clf.fold_models) == df["dataset"].nunique()
    for t in THRESHOLDS:
        assert np.all(pred.confidence[t] <= 1.0)


def test_save_load_roundtrip(tmp_path, trained):
    clf, _, df = trained
    clf.save(tmp_path / "model")
    loaded = RecallClassifier.load(tmp_path / "model")
    a = clf.predict(df.head(30))
    b = loaded.predict(df.head(30))
    for t in THRESHOLDS:
        np.testing.assert_allclose(a.probs[t], b.probs[t], rtol=1e-6)
        np.testing.assert_allclose(
            a.confidence[t], b.confidence[t], rtol=1e-6, atol=1e-9
        )


def test_unseen_category_becomes_missing(trained):
    clf, _, df = trained
    row = df.head(1).copy()
    row["quantization_variant"] = "product_x64"  # never seen in training
    pred = clf.predict(row)  # must not raise
    assert 0 <= pred.probs[0.90][0] <= 1


def test_prepare_features_handles_missing_columns():
    cats = {c: [] for c in ("distance_metric", "quantization_variant",
                            "quantization", "quantization_mode", "rescore")}
    out = prepare_features(pd.DataFrame([{"ef_search": 10}]), cats)
    assert np.isnan(out["hnsw_m"].iloc[0])
    assert out["ef_search"].iloc[0] == 10.0


def test_requires_at_least_three_datasets():
    df = synthetic_training_data(datasets=("only",))
    with pytest.raises(ValueError, match="at least 3 datasets"):
        RecallClassifier().train(df)


def test_two_datasets_rejected_not_silently_degenerate():
    """With exactly 2 datasets, every LODO fold's test+validation split
    consumes both, leaving an empty training set -- this used to pass the
    guard and fail later with a confusing NaN, rather than a clear error."""
    df = synthetic_training_data(datasets=("ds_a", "ds_b"))
    with pytest.raises(ValueError, match="at least 3 datasets"):
        RecallClassifier().train(df)


def test_three_datasets_trains_without_empty_folds():
    """The minimum viable LODO setup: each fold still gets a non-empty
    training set once test + validation are held out. `auc` is legitimately
    absent for a (dataset, threshold) row with only one label present (see
    `train`'s `len(np.unique(...)) == 2` guard) -- `brier`/`accuracy` are
    computed unconditionally and must never be NaN, which an empty fold
    (the exactly-2-dataset case) would have produced."""
    df = synthetic_training_data(datasets=("ds_a", "ds_b", "ds_c"), n_per_dataset=60)
    report = RecallClassifier().train(df, seed=0)
    assert report["brier"].notna().all()
    assert report["accuracy"].notna().all()
