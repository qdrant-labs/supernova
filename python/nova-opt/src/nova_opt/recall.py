"""Recall feasibility classifier over the recall training data (data.csv
schema): one XGBoost binary classifier with the *threshold as an input
feature*, evaluated leave-one-dataset-out, calibrated with Venn–Abers.

Design (each piece replaces a weaker alternative deliberately):

- **Threshold-as-feature, not five independent heads.** Training rows are
  replicated once per threshold t in {0.25, 0.50, 0.80, 0.90, 0.95} with a
  `threshold` column and label 1[mean_recall_at_k >= t]. A single model
  shares statistical strength across thresholds (5x the rows per tree) and a
  monotone constraint on the threshold feature makes
  P(R >= 0.25) >= ... >= P(R >= 0.95) hold *by construction* instead of by
  post-hoc clamping. Domain monotone constraints on `ef_search` and
  `ef_construct` (recall never drops when either grows) keep extrapolation
  past the training grid sane — the optimizer probes exactly those edges.

- **The LODO fold models ARE the deployed model.** Each fold (test dataset
  held out, next dataset as early-stopping validation) trains one model;
  predictions are the fold-ensemble mean and the cross-fold spread is a real
  per-prediction epistemic uncertainty — "how much does this prediction
  depend on which datasets were in training", which is exactly the deployed
  condition (a new, unseen dataset).

- **Venn–Abers calibration** on the pooled out-of-fold scores per threshold
  gives a calibrated probability plus a validity-backed interval; the
  interval width is calibration uncertainty (see venn_abers.py).

- **OOD similarity**: confidence is scaled by how close the workload's
  geometry statistics sit to the training datasets (standardized distance to
  the nearest one, relative to typical inter-dataset spacing). A workload
  that looks like nothing in data.csv gets its probabilities used but
  trusted less.

Confidence per prediction = (1 - min(1, VA width + ensemble spread)) *
similarity. The acquisition turns low confidence into exploration, never
into discarding the candidate.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nova_opt.venn_abers import VennAbers

THRESHOLDS = (0.25, 0.50, 0.80, 0.90, 0.95)

TARGET_COLUMN = "mean_recall_at_k"
DATASET_COLUMN = "dataset"

CATEGORICAL_FEATURES = (
    "distance_metric",
    "quantization_variant",
    "quantization",
    "quantization_mode",
    "rescore",
)

NUMERIC_FEATURES = (
    # run/config
    "corpus_size",
    "query_count",
    "vector_dim",
    "data_size_bytes",
    "number_of_segments",
    "segment_size_kb",
    "hnsw_m",
    "ef_construct",
    "ef_search",
    "top_k",
    # base embedding matrix X
    "embedding_intrinsic_dimensionality",
    "embedding_nn1_dist_p50",
    "embedding_nn1_dist_p95",
    "embedding_nn10_dist_p50",
    "embedding_nn10_dist_p95",
    "embedding_nn100_dist_p50",
    "embedding_nn100_dist_p95",
    "embedding_norm_mean",
    "embedding_norm_std",
    "embedding_pairwise_distance_mean",
    "embedding_pairwise_distance_p05",
    "embedding_pairwise_distance_p50",
    "embedding_pairwise_distance_p95",
    "embedding_pairwise_distance_std",
    # query matrix Q
    "query_intrinsic_dim_estimate",
    "query_duplicate_rate",
    "query_norm_mean",
    "query_norm_std",
    "query_pairwise_distance_p05",
    "query_pairwise_distance_p50",
    "query_pairwise_distance_p95",
    "query_pca_top1_var_ratio",
    "query_pca_top10_var_ratio",
)

FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

# the threshold input column added by the row replication
THRESHOLD_FEATURE = "threshold"

# dataset-geometry columns used for the OOD similarity measure — deliberately
# only the stats features (a new workload legitimately differs in config
# columns; what matters is whether its *geometry* resembles training data)
OOD_FEATURES = tuple(
    f for f in NUMERIC_FEATURES if f.startswith(("embedding_", "query_"))
)

# monotone directions XGBoost enforces: feasibility can only fall as the
# threshold rises, and recall never drops when ef_search / ef_construct grow
_MONOTONE = {THRESHOLD_FEATURE: -1, "ef_search": 1, "ef_construct": 1}

_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "enable_categorical": True,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "n_jobs": -1,
}


def _threshold_key(t: float) -> str:
    return f"{t:.2f}"


class _ConstantModel:
    """Stand-in when a fold's labels are single-class — predicts that class's
    probability everywhere. Keeps degenerate custom datasets from crashing."""

    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.prob)
        return np.column_stack([1 - p, p])


def prepare_features(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Project onto the model's feature columns with stable dtypes: floats
    for numerics (absent columns become NaN = xgboost missing), fixed-vocab
    pandas categoricals for the categorical features (unseen values also
    become missing rather than crashing)."""
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        out[col] = (
            pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
        )
        out[col] = out[col].astype(np.float64)
    for col in CATEGORICAL_FEATURES:
        raw = df[col] if col in df else pd.Series(np.nan, index=df.index)
        # booleans (e.g. `rescore`) become their string form so the category
        # vocabulary is uniform across CSV round-trips
        as_str = raw.map(lambda v: str(v) if pd.notna(v) else np.nan)
        out[col] = pd.Categorical(as_str, categories=categories[col])
    return out


def _with_thresholds(x: pd.DataFrame) -> pd.DataFrame:
    """Replicate `x` once per threshold, appending the threshold column.
    Row order: all rows at 0.25, then all at 0.50, ... — `_slice(t)` below
    relies on this layout."""
    reps = []
    for t in THRESHOLDS:
        rep = x.copy()
        rep[THRESHOLD_FEATURE] = float(t)
        reps.append(rep)
    return pd.concat(reps, ignore_index=True)


def _monotone_tuple(columns: list[str]) -> str:
    """XGBoost's tuple-form monotone_constraints aligned with column order
    (dict form is finicky across versions; the tuple form always works).
    Categorical columns get 0 (no constraint)."""
    return "(" + ",".join(str(_MONOTONE.get(c, 0)) for c in columns) + ")"


def _fit_one(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    seed: int,
):
    if len(np.unique(y_train)) < 2:
        return _ConstantModel(float(y_train.mean()))
    from xgboost import XGBClassifier

    params = dict(
        _XGB_PARAMS,
        random_state=seed,
        monotone_constraints=_monotone_tuple(list(x_train.columns)),
    )
    if x_val is not None and len(np.unique(y_val)) == 2:
        model = XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, verbose=False)
    return model


@dataclass
class RecallPrediction:
    """Calibrated, monotonic threshold-probability profile for one or more
    candidates. `probs[t]` and `confidence[t]` are aligned float arrays."""

    probs: dict[float, np.ndarray]
    confidence: dict[float, np.ndarray]


class RecallClassifier:
    def __init__(self):
        self.fold_models: list[Any] = []
        self.fold_datasets: list[str] = []
        self.calibrators: dict[float, VennAbers] = {}
        self.categories: dict[str, list[str]] = {}
        self.lodo_metrics: dict[str, Any] = {}
        # OOD reference: per-training-dataset geometry vectors + spread
        self._ood_centers: np.ndarray | None = None  # (n_datasets, n_feats)
        self._ood_scale: np.ndarray | None = None  # feature-wise std
        self._ood_columns: list[str] = []
        self._ood_typical: float = 1.0  # median nearest-neighbor distance

    # -- training -----------------------------------------------------------

    def train(self, df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
        """LODO folds -> per-fold models (kept as the deployed ensemble) and
        pooled out-of-fold scores -> per-threshold Venn–Abers calibrators +
        metrics. Returns the LODO report (one row per dataset x threshold)."""
        from sklearn.metrics import brier_score_loss, roc_auc_score

        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        if DATASET_COLUMN not in df:
            raise ValueError(f"training data needs a '{DATASET_COLUMN}' column")
        self.categories = {
            col: sorted(
                {str(v) for v in df[col].dropna().unique()} if col in df else set()
            )
            for col in CATEGORICAL_FEATURES
        }
        x_base = prepare_features(df, self.categories)
        recall = df[TARGET_COLUMN].to_numpy(dtype=float)
        datasets = sorted(df[DATASET_COLUMN].unique())
        if len(datasets) < 3:
            # each fold holds out one dataset for test and the *next* one
            # for early-stopping validation (see class docstring) -- with
            # only 2 datasets total every fold's train set is empty
            raise ValueError(
                "leave-one-dataset-out needs at least 3 datasets (one fold "
                f"reserves a test set and a distinct validation set); got {datasets}"
            )

        n = len(df)
        x_all = _with_thresholds(x_base)  # n * len(THRESHOLDS) rows
        y_all = np.concatenate([(recall >= t).astype(int) for t in THRESHOLDS])
        ds_all = np.tile(df[DATASET_COLUMN].to_numpy(), len(THRESHOLDS))

        self.fold_models, self.fold_datasets = [], []
        oof = {t: np.full(n, np.nan) for t in THRESHOLDS}
        report_rows = []
        for i, test_ds in enumerate(datasets):
            val_ds = datasets[(i + 1) % len(datasets)]
            test_mask = ds_all == test_ds
            val_mask = ds_all == val_ds
            train_mask = ~test_mask & ~val_mask
            model = _fit_one(
                x_all[train_mask], y_all[train_mask],
                x_all[val_mask], y_all[val_mask], seed,
            )
            self.fold_models.append(model)
            self.fold_datasets.append(test_ds)

            p_test = model.predict_proba(x_all[test_mask])[:, 1]
            y_test = y_all[test_mask]
            # test rows appear once per threshold, in THRESHOLDS order
            per_t = p_test.reshape(len(THRESHOLDS), -1)
            y_per_t = y_test.reshape(len(THRESHOLDS), -1)
            base_mask = (df[DATASET_COLUMN] == test_ds).to_numpy()
            for j, t in enumerate(THRESHOLDS):
                oof[t][base_mask] = per_t[j]
                row = {"dataset": test_ds, "threshold": t, "n": int(base_mask.sum())}
                if len(np.unique(y_per_t[j])) == 2:
                    row["auc"] = float(roc_auc_score(y_per_t[j], per_t[j]))
                row["brier"] = float(brier_score_loss(y_per_t[j], per_t[j]))
                row["accuracy"] = float(((per_t[j] >= 0.5) == y_per_t[j]).mean())
                report_rows.append(row)
        report = pd.DataFrame(report_rows)

        # pooled-OOF Venn–Abers calibration + metrics per threshold
        for t in THRESHOLDS:
            y = (recall >= t).astype(int)
            p = oof[t]
            self.calibrators[t] = VennAbers.fit(p, y, seed=seed)
            pooled_auc = (
                float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5
            )
            self.lodo_metrics[_threshold_key(t)] = {
                "pooled_auc": pooled_auc,
                "pooled_brier": float(np.mean((p - y) ** 2)),
                "positive_rate": float(y.mean()),
            }

        self._fit_ood(df)
        return report

    def _fit_ood(self, df: pd.DataFrame) -> None:
        """Per-dataset geometry centroids + scale, for workload similarity."""
        cols = [c for c in OOD_FEATURES if c in df.columns]
        if not cols:
            self._ood_centers = None
            return
        centers = (
            df.groupby(DATASET_COLUMN)[list(cols)].mean().to_numpy(dtype=float)
        )
        scale = np.nanstd(centers, axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        z = np.nan_to_num(centers / scale)
        # typical spacing: each training dataset's distance to its nearest
        # other training dataset (median over datasets)
        if len(z) >= 2:
            d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            self._ood_typical = float(np.median(d.min(axis=1)))
        self._ood_centers = z
        self._ood_scale = scale
        self._ood_columns = list(cols)

    def similarity(self, rows: pd.DataFrame) -> np.ndarray:
        """Per-row workload familiarity in [0, 1]: 1 when the geometry stats
        sit within typical inter-dataset spacing of some training dataset,
        decaying as they drift out of distribution. Rows without stats
        columns get 1 (nothing to judge by — the VA width and ensemble
        spread still apply)."""
        n = len(rows)
        if self._ood_centers is None:
            return np.ones(n)
        cols = self._ood_columns
        present = [c for c in cols if c in rows.columns]
        if not present:
            return np.ones(n)
        x = np.full((n, len(cols)), np.nan)
        for j, c in enumerate(cols):
            if c in rows.columns:
                x[:, j] = pd.to_numeric(rows[c], errors="coerce").to_numpy()
        z = x / self._ood_scale
        out = np.ones(n)
        for i in range(n):
            valid = np.isfinite(z[i])
            if not valid.any():
                continue
            # compare on the shared dims, rescaled to full dimensionality so
            # partially-known rows aren't artificially "close"
            frac = np.sqrt(len(cols) / valid.sum())
            d = np.linalg.norm(
                self._ood_centers[:, valid] - z[i, valid], axis=1
            ).min() * frac
            out[i] = float(np.exp(-max(0.0, d / self._ood_typical - 1.0)))
        return out

    # -- prediction ----------------------------------------------------------

    def _raw_scores(self, x_base: pd.DataFrame) -> np.ndarray:
        """(n_folds, n_thresholds, n_rows) raw fold-model scores."""
        x5 = _with_thresholds(x_base)
        out = np.empty((len(self.fold_models), len(THRESHOLDS), len(x_base)))
        for k, model in enumerate(self.fold_models):
            out[k] = model.predict_proba(x5)[:, 1].reshape(
                len(THRESHOLDS), len(x_base)
            )
        return out

    def predict(self, rows: pd.DataFrame) -> RecallPrediction:
        if not self.fold_models:
            raise RuntimeError("classifier is not trained/loaded")
        x = prepare_features(rows, self.categories)
        scores = self._raw_scores(x)  # (folds, thresholds, rows)
        sim = self.similarity(rows)

        probs: dict[float, np.ndarray] = {}
        confidence: dict[float, np.ndarray] = {}
        for j, t in enumerate(THRESHOLDS):
            va = self.calibrators[t]
            p, width = va.predict(scores[:, j, :].mean(axis=0))
            # epistemic spread: how much the calibrated probability moves
            # across LODO folds (i.e. across "which datasets trained me")
            fold_p = np.stack([va.predict(scores[k, j, :])[0]
                               for k in range(scores.shape[0])])
            spread = fold_p.std(axis=0)
            probs[t] = np.clip(p, 0.0, 1.0)
            confidence[t] = (1.0 - np.minimum(1.0, width + spread)) * sim
        # the threshold monotone constraint makes raw scores monotone in t;
        # per-threshold calibrators can jitter that slightly — re-clamp
        ordered = sorted(THRESHOLDS, reverse=True)
        for prev, t in zip(ordered, ordered[1:]):
            probs[t] = np.maximum(probs[t], probs[prev])
        return RecallPrediction(probs=probs, confidence=confidence)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for k, model in enumerate(self.fold_models):
            if isinstance(model, _ConstantModel):
                (out / f"fold_{k}.const").write_text(str(model.prob))
            else:
                model.save_model(str(out / f"fold_{k}.ubj"))
        meta = {
            "thresholds": list(THRESHOLDS),
            "features": list(FEATURES),
            "categories": self.categories,
            "fold_datasets": self.fold_datasets,
            "lodo_metrics": self.lodo_metrics,
            "calibrators": {
                _threshold_key(t): va.to_dict() for t, va in self.calibrators.items()
            },
            "ood": (
                None
                if self._ood_centers is None
                else {
                    "columns": self._ood_columns,
                    "centers": self._ood_centers.tolist(),
                    "scale": self._ood_scale.tolist(),
                    "typical": self._ood_typical,
                }
            ),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str) -> "RecallClassifier":
        from xgboost import XGBClassifier

        src = Path(path)
        meta = json.loads((src / "meta.json").read_text())
        clf = cls()
        clf.categories = meta["categories"]
        clf.fold_datasets = meta["fold_datasets"]
        clf.lodo_metrics = meta.get("lodo_metrics", {})
        clf.calibrators = {
            float(k): VennAbers.from_dict(v)
            for k, v in meta["calibrators"].items()
        }
        for k in range(len(clf.fold_datasets)):
            const = src / f"fold_{k}.const"
            if const.exists():
                clf.fold_models.append(_ConstantModel(float(const.read_text())))
            else:
                model = XGBClassifier()
                model.load_model(str(src / f"fold_{k}.ubj"))
                clf.fold_models.append(model)
        ood = meta.get("ood")
        if ood:
            clf._ood_columns = ood["columns"]
            clf._ood_centers = np.array(ood["centers"])
            clf._ood_scale = np.array(ood["scale"])
            clf._ood_typical = ood["typical"]
        return clf


def load_or_train(
    *,
    data_csv: str,
    model_dir: str | None,
    seed: int = 0,
) -> tuple[RecallClassifier, pd.DataFrame | None]:
    """Load a saved classifier from `model_dir` when it exists; otherwise
    train from `data_csv` (and save to `model_dir` when given). Returns
    `(classifier, lodo_report_or_None)` — the report is None on a pure load."""
    if model_dir and (Path(model_dir) / "meta.json").exists():
        return RecallClassifier.load(model_dir), None
    df = pd.read_csv(data_csv)
    clf = RecallClassifier()
    report = clf.train(df, seed=seed)
    if model_dir:
        clf.save(model_dir)
    return clf, report
