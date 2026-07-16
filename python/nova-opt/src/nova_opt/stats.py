"""Dataset/query statistics extractor for the recall feasibility classifier.

Reproduces the feature schema of the recall training data (data.csv): base
embedding matrix X features (`embedding_*` + counts) and query matrix Q
features (`query_*`), with the same default sampling sizes as the batch
script that produced the historical rows.

Two computation regimes, chosen per matrix:

- full pass: every row participates in norm/count stats (feasible when the
  matrix fits the configured row limit)
- sampled: a seeded random row sample stands in, and the feature is marked
  as a sampled estimate in the returned provenance

Pairwise-distance, kNN, and PCA features are *always* sampled (that's their
definition — random pairs / random query rows), so they carry their sampling
parameters in the provenance either way.

Q features are computed from Q alone — deliberately no query-to-base
retrieval features, which would leak retrieval difficulty from X into the
query representation.
"""

from __future__ import annotations

import hashlib

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# distance_metric vocabulary of the training data -> local metric names
METRIC_ALIASES = {
    "cosine": "cosine",
    "l2": "euclidean",
    "euclidean": "euclidean",
    "ip": "dot",
    "dot": "dot",
}

SUPPORTED_METRICS = ("cosine", "euclidean", "dot")


def resolve_metric(name: str) -> str:
    metric = METRIC_ALIASES.get(name.lower())
    if metric is None:
        raise ValueError(
            f"unknown distance metric '{name}'; expected one of "
            f"{sorted(set(METRIC_ALIASES))}"
        )
    return metric


@dataclass(frozen=True)
class StatsParams:
    """Sampling knobs, defaults matching the historical batch script."""

    sample_size: int = 1000  # query PCA
    pair_sample_size: int = 100_000  # random pair distances
    nn_query_sample_size: int = 256
    nn_reference_sample_size: int = 5000
    knn_k: int = 100
    seed: int = 0
    # Above this many rows, full-pass stats fall back to a random sample of
    # this many rows (marked "sampled" in the provenance).
    full_pass_row_limit: int = 2_000_000


# ---------------------------------------------------------------------------
# matrix loading


def load_matrix(
    path: str,
    *,
    column: str | None = None,
    max_rows: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, int, bool]:
    """Load an embedding matrix from `.npy` (memory-mapped) or parquet
    (file or directory; `column` selects the vector column). Returns
    `(matrix, total_rows, sampled)` — when the source holds more than
    `max_rows` rows, a seeded uniform row sample of `max_rows` is returned
    with `sampled=True`; the source is never required to fit in memory."""
    rng = np.random.default_rng(seed)
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p, mmap_mode="r")
        total = arr.shape[0]
        if max_rows is not None and total > max_rows:
            idx = np.sort(rng.choice(total, size=max_rows, replace=False))
            return np.asarray(arr[idx], dtype=np.float64), total, True
        return np.asarray(arr, dtype=np.float64), total, False

    import pyarrow.dataset as ds

    dataset = ds.dataset(str(p))
    if column is None:
        raise ValueError(f"a `column` is required to read vectors from parquet: {path}")
    total = dataset.count_rows()
    if max_rows is not None and total > max_rows:
        idx = np.sort(rng.choice(total, size=max_rows, replace=False))
        table = dataset.take(idx, columns=[column])
        sampled = True
    else:
        table = dataset.to_table(columns=[column])
        sampled = False
    col = table.column(column)
    mat = np.array(col.to_pylist(), dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"column '{column}' in {path} is not a fixed-width vector column"
        )
    return mat, total, sampled


# ---------------------------------------------------------------------------
# distances


def _pair_distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Row-wise distance between paired rows of `a` and `b`."""
    if metric == "euclidean":
        return np.linalg.norm(a - b, axis=1)
    if metric == "dot":
        # inner-product similarity flipped into a distance so "smaller is
        # closer" holds for every metric here
        return -np.einsum("ij,ij->i", a, b)
    if metric == "cosine":
        num = np.einsum("ij,ij->i", a, b)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return 1.0 - num / np.maximum(den, 1e-12)
    raise ValueError(f"unsupported metric '{metric}'")


def _cross_distances(q: np.ndarray, r: np.ndarray, metric: str) -> np.ndarray:
    """Dense (len(q), len(r)) distance matrix."""
    if metric == "euclidean":
        qq = np.sum(q * q, axis=1)[:, None]
        rr = np.sum(r * r, axis=1)[None, :]
        d2 = np.maximum(qq + rr - 2.0 * (q @ r.T), 0.0)
        return np.sqrt(d2)
    if metric == "dot":
        return -(q @ r.T)
    if metric == "cosine":
        qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
        return 1.0 - qn @ rn.T
    raise ValueError(f"unsupported metric '{metric}'")


def _sample_rows(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return np.arange(len(x))
    return rng.choice(len(x), size=n, replace=False)


def _pairwise_sample(
    x: np.ndarray, n_pairs: int, metric: str, rng: np.random.Generator
) -> np.ndarray:
    """Distances over `n_pairs` random unordered row pairs (i != j)."""
    n = len(x)
    if n < 2:
        return np.array([])
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    return _pair_distances(x[i[mask]], x[j[mask]], metric)


def _knn_distances(
    x: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    *,
    n_queries: int,
    n_refs: int,
    k: int,
) -> np.ndarray:
    """Sorted self-kNN distances: sampled query rows against sampled
    reference rows, self-matches excluded. Shape (n_queries, k'), columns
    are rank-1..k' neighbor distances."""
    qi = _sample_rows(x, n_queries, rng)
    ri = _sample_rows(x, n_refs, rng)
    dist = _cross_distances(x[qi], x[ri], metric)
    # exclude self: same underlying row index (both index into x)
    self_mask = qi[:, None] == ri[None, :]
    dist[self_mask] = np.inf
    dist.sort(axis=1)
    k_eff = min(k, dist.shape[1])
    out = dist[:, :k_eff]
    return out[np.isfinite(out[:, 0])]


def _twonn_metric(metric: str) -> str:
    """The metric TwoNN's kNN distances must be computed under.

    TwoNN's MLE (Facco et al. 2017) derives from how a d-ball's volume grows
    with radius — it requires genuine non-negative metric distances from a
    reference point. `euclidean`/`cosine` distances qualify; `dot`'s
    "distance" (negated inner product, so ranking is consistent — see
    `_pair_distances`) does not: it isn't translation-invariant, can be any
    sign, and doesn't correspond to ball-volume growth at all, so its ratio
    `d2/d1` isn't the quantity TwoNN's derivation is about. Nearest-neighbor
    *ranking* still legitimately uses the workload's own metric elsewhere
    (percentile features); only this estimator needs a real metric distance,
    so `dot` falls back to `euclidean` here specifically."""
    return "euclidean" if metric == "dot" else metric


def _intrinsic_dim_twonn(knn: np.ndarray) -> float:
    """TwoNN maximum-likelihood intrinsic dimensionality from the first two
    positive neighbor distances per query row. NaN when too few valid rows.
    `knn` must already be under a metric TwoNN's derivation holds for (see
    `_twonn_metric`) — never raw `dot`-metric distances, which are routinely
    negative for genuinely close points and would make every row invalid."""
    if knn.shape[0] == 0 or knn.shape[1] < 2:
        return float("nan")
    d1, d2 = knn[:, 0], knn[:, 1]
    valid = (d1 > 0) & (d2 > d1) & np.isfinite(d2)
    if valid.sum() < 2:
        return float("nan")
    mu = d2[valid] / d1[valid]
    return float(valid.sum() / np.sum(np.log(mu)))


def _rank_col(knn: np.ndarray, rank: int) -> np.ndarray:
    """Rank-`rank` neighbor distances (1-based), empty when k is too small."""
    if knn.shape[1] < rank:
        return np.array([])
    return knn[:, rank - 1]


def _pcts(values: np.ndarray, pcts: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return [float("nan")] * len(pcts)
    return [float(np.percentile(values, p)) for p in pcts]


# ---------------------------------------------------------------------------
# feature computation


def base_embedding_stats(
    x: np.ndarray,
    *,
    total_rows: int,
    full_pass: bool,
    metric: str,
    params: StatsParams,
) -> tuple[dict[str, float], dict[str, Any]]:
    """`embedding_*` features from the base matrix X. `x` is either the full
    matrix (`full_pass=True`) or an already-drawn row sample standing in for
    a matrix of `total_rows` rows."""
    rng = np.random.default_rng(params.seed)
    norms = np.linalg.norm(x, axis=1)
    features: dict[str, float] = {
        "number_of_embeddings": float(total_rows),
        "dimensionality": float(x.shape[1]),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "embedding_norm_p50": float(np.percentile(norms, 50)),
        "embedding_norm_p95": float(np.percentile(norms, 95)),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
    }

    pd_ = _pairwise_sample(x, params.pair_sample_size, metric, rng)
    p05, p50, p95 = _pcts(pd_, (5, 50, 95))
    features.update(
        embedding_pairwise_distance_p05=p05,
        embedding_pairwise_distance_p50=p50,
        embedding_pairwise_distance_p95=p95,
        embedding_pairwise_distance_mean=float(pd_.mean()) if pd_.size else float("nan"),
        embedding_pairwise_distance_std=float(pd_.std()) if pd_.size else float("nan"),
    )

    knn = _knn_distances(
        x, metric, rng,
        n_queries=params.nn_query_sample_size,
        n_refs=params.nn_reference_sample_size,
        k=params.knn_k,
    )
    for rank in (1, 10, 100):
        vals = _rank_col(knn, rank)
        r50, r95 = _pcts(vals, (50, 95))
        features[f"embedding_nn{rank}_dist_p50"] = r50
        features[f"embedding_nn{rank}_dist_p95"] = r95
    twonn_metric = _twonn_metric(metric)
    twonn_knn = knn if twonn_metric == metric else _knn_distances(
        x, twonn_metric, rng,
        n_queries=params.nn_query_sample_size,
        n_refs=params.nn_reference_sample_size,
        k=params.knn_k,
    )
    features["embedding_intrinsic_dimensionality"] = _intrinsic_dim_twonn(twonn_knn)

    norm_mode = "exact" if full_pass else "sampled"
    provenance = {
        "metric": metric,
        "norm_stats": norm_mode,
        "norm_rows_used": int(len(x)),
        "pairwise": "sampled",
        "knn": "sampled",
        "twonn_metric": twonn_metric,
        "sampling": asdict(params),
    }
    return features, provenance


def query_stats(
    q: np.ndarray,
    *,
    total_rows: int,
    full_pass: bool,
    metric: str,
    params: StatsParams,
) -> tuple[dict[str, float], dict[str, Any]]:
    """`query_*` features from the query matrix Q alone — no use of the base
    corpus, by design (see module docstring)."""
    rng = np.random.default_rng(params.seed)
    norms = np.linalg.norm(q, axis=1)

    hashes = {hashlib.md5(row.tobytes()).digest() for row in np.ascontiguousarray(q)}
    dup_rate = 1.0 - len(hashes) / len(q) if len(q) else float("nan")

    features: dict[str, float] = {
        "query_count": float(total_rows),
        "query_norm_mean": float(norms.mean()),
        "query_norm_std": float(norms.std()),
        "query_norm_p50": float(np.percentile(norms, 50)),
        "query_norm_p95": float(np.percentile(norms, 95)),
        "query_duplicate_rate": float(dup_rate),
    }

    # PCA on a seeded row sample
    from sklearn.decomposition import PCA

    pi = _sample_rows(q, params.sample_size, rng)
    sample = q[pi]
    n_comp = min(10, sample.shape[0] - 1, sample.shape[1])
    if n_comp >= 1:
        ratios = PCA(n_components=n_comp).fit(sample).explained_variance_ratio_
        features["query_pca_top1_var_ratio"] = float(ratios[0])
        features["query_pca_top10_var_ratio"] = float(ratios[:10].sum())
    else:
        features["query_pca_top1_var_ratio"] = float("nan")
        features["query_pca_top10_var_ratio"] = float("nan")

    pd_ = _pairwise_sample(q, params.pair_sample_size, metric, rng)
    p05, p50, p95 = _pcts(pd_, (5, 50, 95))
    features.update(
        query_pairwise_distance_p05=p05,
        query_pairwise_distance_p50=p50,
        query_pairwise_distance_p95=p95,
    )

    knn = _knn_distances(
        q, metric, rng,
        n_queries=params.nn_query_sample_size,
        n_refs=params.nn_reference_sample_size,
        k=params.knn_k,
    )
    twonn_metric = _twonn_metric(metric)
    twonn_knn = knn if twonn_metric == metric else _knn_distances(
        q, twonn_metric, rng,
        n_queries=params.nn_query_sample_size,
        n_refs=params.nn_reference_sample_size,
        k=params.knn_k,
    )
    features["query_intrinsic_dim_estimate"] = _intrinsic_dim_twonn(twonn_knn)

    provenance = {
        "metric": metric,
        "norm_stats": "exact" if full_pass else "sampled",
        "norm_rows_used": int(len(q)),
        "pca": "sampled",
        "pairwise": "sampled",
        "knn": "sampled",
        "twonn_metric": twonn_metric,
        "sampling": asdict(params),
    }
    return features, provenance


def compute_workload_stats(
    *,
    corpus_path: str,
    corpus_column: str | None,
    queries_path: str,
    queries_column: str | None,
    distance_metric: str,
    params: StatsParams = StatsParams(),
) -> tuple[dict[str, float], dict[str, Any]]:
    """One-call extractor for a workload: loads X and Q (falling back to
    row-sampled loads past `full_pass_row_limit`), computes both feature
    sets under the workload's own metric, and returns
    `(features, provenance)` ready to join with candidate config features."""
    metric = resolve_metric(distance_metric)
    x, x_total, x_sampled = load_matrix(
        corpus_path, column=corpus_column,
        max_rows=params.full_pass_row_limit, seed=params.seed,
    )
    q, q_total, q_sampled = load_matrix(
        queries_path, column=queries_column,
        max_rows=params.full_pass_row_limit, seed=params.seed,
    )
    base_feats, base_meta = base_embedding_stats(
        x, total_rows=x_total, full_pass=not x_sampled, metric=metric, params=params
    )
    q_feats, q_meta = query_stats(
        q, total_rows=q_total, full_pass=not q_sampled, metric=metric, params=params
    )
    features = {**base_feats, **q_feats}
    provenance = {"base": base_meta, "queries": q_meta}
    return features, provenance
