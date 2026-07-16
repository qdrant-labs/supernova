"""Inductive Venn–Abers calibration, grid-precomputed.

For a raw classifier score s, a Venn–Abers predictor returns an *interval*
[p0(s), p1(s)]: isotonic calibration of the calibration set augmented with
the test point labeled 0 (resp. 1), evaluated at the test point. The
interval provably contains a perfectly calibrated probability, so its width
is a principled per-prediction measure of calibration uncertainty — exactly
the "how much should the tuner trust this probability" signal, replacing
any ad-hoc global reliability scalar.

Exact per-query IVAP costs a PAVA run per prediction; the tuner scores
thousands of candidates per iteration, so instead p0/p1 are precomputed on a
fixed score grid at fit time (one isotonic fit per grid point per label —
sklearn's PAVA is C-optimized, so this is seconds once) and queried by
monotone interpolation. Isotonic functions are piecewise-constant with
breakpoints only at calibration scores; a ~1e-2 grid keeps interpolation
error far below the interval widths that matter.

The single merged probability follows Vovk's minimax log-loss merger:
p = p1 / (1 - p0 + p1).
"""

from __future__ import annotations

import numpy as np


class VennAbers:
    def __init__(self, grid: np.ndarray, p0: np.ndarray, p1: np.ndarray):
        self.grid = np.asarray(grid, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        grid_size: int = 129,
        max_calibration: int = 10_000,
        seed: int = 0,
    ) -> "VennAbers":
        from sklearn.isotonic import IsotonicRegression

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        keep = np.isfinite(scores)
        scores, labels = scores[keep], labels[keep]
        if len(scores) > max_calibration:
            idx = np.random.default_rng(seed).choice(
                len(scores), size=max_calibration, replace=False
            )
            scores, labels = scores[idx], labels[idx]

        lo, hi = float(scores.min()), float(scores.max())
        grid = np.unique(
            np.concatenate([[0.0, 1.0], np.linspace(lo, hi, grid_size)])
        )
        p0 = np.empty(len(grid))
        p1 = np.empty(len(grid))
        for i, s in enumerate(grid):
            for label, out in ((0.0, p0), (1.0, p1)):
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
                ).fit(np.append(scores, s), np.append(labels, label))
                out[i] = iso.predict([s])[0]
        # p0 <= p1 holds pointwise in theory; enforce against fit jitter
        p0_c = np.minimum(p0, p1)
        return cls(grid, p0_c, np.maximum(p0, p1))

    def interval(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        return (
            np.interp(s, self.grid, self.p0),
            np.interp(s, self.grid, self.p1),
        )

    def predict(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(merged probability, interval width) per score."""
        p0, p1 = self.interval(scores)
        p = p1 / np.maximum(1.0 - p0 + p1, 1e-12)
        return p, p1 - p0

    # -- persistence (plain dict; stored inside the classifier's meta.json) --

    def to_dict(self) -> dict:
        return {
            "grid": self.grid.tolist(),
            "p0": self.p0.tolist(),
            "p1": self.p1.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VennAbers":
        return cls(np.array(d["grid"]), np.array(d["p0"]), np.array(d["p1"]))
