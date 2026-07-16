"""The hierarchical configuration space and its artifact-key decomposition.

A candidate is one point in Layout x Index x Quantization x Search. The four
levels are ordered by rebuild cost:

    layout_key = (segments, dtype, shard_count, payload_layout)      reinsert
    index_key  = (layout_key, m, ef_construct, on_disk, threshold)   index build
    quant_key  = (index_key, variant, always_ram)                    quant build
    search_key = (quant_key, ef_search, batch_size, top_k, conc, rescore)

Everything downstream (cost model, artifact cache, scheduler) keys off these
tuples — two candidates that share a prefix key share the artifact that prefix
names, which is what makes reuse-aware marginal costing possible.

Quantization variants use the exact vocabulary of the recall training data
(`quantization_variant` / `quantization` / `quantization_mode` columns in
data.csv) so candidate features join cleanly against the classifier, and each
variant knows its own nova-load `quantization:` block for live evaluation.
"""

from __future__ import annotations

import itertools

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantVariant:
    """One quantization option: its data.csv feature values plus the
    nova-load `quantization:` reindex block that materializes it."""

    variant: str  # data.csv `quantization_variant`
    quantization: str  # data.csv `quantization` (method family)
    mode: str  # data.csv `quantization_mode`
    load_block: dict[str, Any]  # nova-load QuantizationConfig fields


# The full vocabulary observed in data.csv, mapped onto nova-load's
# scalar/product/binary/turbo/none quantization schema
# (crates/nova-load/src/config.rs::QuantizationConfig).
QUANT_VARIANTS: dict[str, QuantVariant] = {
    v.variant: v
    for v in (
        QuantVariant("none", "NONE", "NONE", {"type": "none"}),
        QuantVariant(
            "scalar_default", "SCALAR", "SCALAR__scalar_default", {"type": "scalar"}
        ),
        QuantVariant(
            "binary_1bit", "BINARY", "BINARY__DEFAULT",
            {"type": "binary", "encoding": "one_bit"},
        ),
        QuantVariant(
            "binary_1_5bit", "BINARY", "BINARY__ONE_AND_HALF_BITS",
            {"type": "binary", "encoding": "one_and_half_bits"},
        ),
        QuantVariant(
            "binary_2bit", "BINARY", "BINARY__TWO_BITS",
            {"type": "binary", "encoding": "two_bits"},
        ),
        *(
            QuantVariant(
                f"product_x{ratio}", "PRODUCT", f"PRODUCT__X{ratio}",
                {"type": "product", "compression": f"x{ratio}"},
            )
            for ratio in (4, 8, 16, 32, 64)
        ),
        *(
            QuantVariant(
                f"turbo_bits{str(bits).replace('.', '_')}", "TURBO",
                f"TURBO__BITS{str(bits).replace('.', '_').upper()}",
                {"type": "turbo", "bits": bits},
            )
            for bits in (1, 1.5, 2, 4)
        ),
    )
}

_DTYPE_BYTES = {"float32": 4, "float16": 2, "uint8": 1}


@dataclass(frozen=True)
class Layout:
    """Data/layout level — changing any field means reinserting the corpus."""

    segments: int  # optimizers.default_segment_number
    dtype: str = "float32"  # vectors.dense.datatype
    shard_count: int = 1
    on_disk_payload: bool = False


@dataclass(frozen=True)
class Index:
    """Index level — changing any field means rebuilding the HNSW index."""

    m: int
    ef_construct: int
    on_disk: bool = False
    indexing_threshold: int | None = None


@dataclass(frozen=True)
class Quant:
    """Quantization level — cheaper than an index rebuild, dearer than search."""

    variant: str = "none"
    always_ram: bool = True

    def __post_init__(self) -> None:
        if self.variant not in QUANT_VARIANTS:
            raise ValueError(
                f"unknown quantization variant '{self.variant}'; "
                f"available: {sorted(QUANT_VARIANTS)}"
            )


@dataclass(frozen=True)
class Search:
    """Search level — a query-time operating point, no rebuild at all."""

    ef_search: int
    batch_size: int = 1
    top_k: int = 10
    concurrency: int = 1
    rescore: bool | None = None


@dataclass(frozen=True)
class Candidate:
    layout: Layout
    index: Index
    quant: Quant
    search: Search

    @property
    def layout_key(self) -> tuple:
        lo = self.layout
        return (lo.segments, lo.dtype, lo.shard_count, lo.on_disk_payload)

    @property
    def index_key(self) -> tuple:
        ix = self.index
        return (self.layout_key, ix.m, ix.ef_construct, ix.on_disk, ix.indexing_threshold)

    @property
    def quant_key(self) -> tuple:
        return (self.index_key, self.quant.variant, self.quant.always_ram)

    @property
    def search_key(self) -> tuple:
        s = self.search
        return (self.quant_key, s.ef_search, s.batch_size, s.top_k, s.concurrency, s.rescore)

    def with_search(self, search: Search) -> "Candidate":
        return replace(self, search=search)


def candidate_from_quant_key(quant_key: tuple, search: Search) -> Candidate:
    """Rebuild a Candidate from a cached quant-level artifact key plus a
    fresh search operating point — the inverse of `Candidate.quant_key`,
    used to propose cheap siblings of artifacts that already exist."""
    index_key, variant, always_ram = quant_key
    layout_key, m, ef_construct, on_disk, indexing_threshold = index_key
    segments, dtype, shard_count, on_disk_payload = layout_key
    return Candidate(
        layout=Layout(segments=segments, dtype=dtype, shard_count=shard_count,
                      on_disk_payload=on_disk_payload),
        index=Index(m=m, ef_construct=ef_construct, on_disk=on_disk,
                    indexing_threshold=indexing_threshold),
        quant=Quant(variant=variant, always_ram=always_ram),
        search=search,
    )


def config_features(
    cand: Candidate, workload: dict[str, Any]
) -> dict[str, Any]:
    """Join one candidate with workload metadata into the run/config feature
    columns of the recall training schema (data.csv). `workload` must carry
    `corpus_size`, `query_count`, `vector_dim`, and `distance_metric` (from
    the stats extractor / tuner config); `data_size_bytes` / `segment_size_kb`
    are derived the same way the batch pipeline derived them.
    """
    qv = QUANT_VARIANTS[cand.quant.variant]
    corpus_size = int(workload["corpus_size"])
    vector_dim = int(workload["vector_dim"])
    data_size_bytes = int(
        workload.get("data_size_bytes")
        or corpus_size * vector_dim * _DTYPE_BYTES.get(cand.layout.dtype, 4)
    )
    return {
        "corpus_size": corpus_size,
        "query_count": int(workload["query_count"]),
        "vector_dim": vector_dim,
        "data_size_bytes": data_size_bytes,
        "distance_metric": workload["distance_metric"],
        "number_of_segments": cand.layout.segments,
        # half-up, not banker's round() — reproduces the training pipeline's
        # values exactly (e.g. bigann-10M: 39062.5 -> 39063)
        "segment_size_kb": int(data_size_bytes / cand.layout.segments / 1024 + 0.5),
        "quantization_variant": qv.variant,
        "quantization": qv.quantization,
        "quantization_mode": qv.mode,
        "hnsw_m": cand.index.m,
        "ef_construct": cand.index.ef_construct,
        "ef_search": cand.search.ef_search,
        "rescore": cand.search.rescore,
        "top_k": cand.search.top_k,
    }


@dataclass(frozen=True)
class SpaceAxes:
    """The per-field value lists the space is the cartesian product of."""

    segments: tuple[int, ...] = (8,)
    dtype: tuple[str, ...] = ("float32",)
    shard_count: tuple[int, ...] = (1,)
    on_disk_payload: tuple[bool, ...] = (False,)

    m: tuple[int, ...] = (16, 32)
    ef_construct: tuple[int, ...] = (128,)
    index_on_disk: tuple[bool, ...] = (False,)
    indexing_threshold: tuple[int | None, ...] = (None,)

    quant_variant: tuple[str, ...] = ("none",)
    always_ram: tuple[bool, ...] = (True,)

    ef_search: tuple[int, ...] = (16, 32, 64, 128)
    batch_size: tuple[int, ...] = (1,)
    top_k: tuple[int, ...] = (10,)
    concurrency: tuple[int, ...] = (1,)
    rescore: tuple[bool | None, ...] = (None,)


class ConfigSpace:
    """Candidate generation over `SpaceAxes` — uniform random sampling without
    replacement (deduped against both itself and an exclusion set), plus the
    cheap search-level siblings used to amortize expensive builds."""

    def __init__(self, axes: SpaceAxes):
        self.axes = axes

    def size(self) -> int:
        a = self.axes
        n = 1
        for field in (
            a.segments, a.dtype, a.shard_count, a.on_disk_payload,
            a.m, a.ef_construct, a.index_on_disk, a.indexing_threshold,
            a.quant_variant, a.always_ram,
            a.ef_search, a.batch_size, a.top_k, a.concurrency, a.rescore,
        ):
            n *= len(field)
        return n

    def _make(self, choice: tuple) -> Candidate:
        (seg, dt, sh, odp, m, efc, iod, thr, qv, ram, efs, bs, k, conc, rs) = choice
        return Candidate(
            layout=Layout(segments=seg, dtype=dt, shard_count=sh, on_disk_payload=odp),
            index=Index(m=m, ef_construct=efc, on_disk=iod, indexing_threshold=thr),
            quant=Quant(variant=qv, always_ram=ram),
            search=Search(ef_search=efs, batch_size=bs, top_k=k, concurrency=conc, rescore=rs),
        )

    def _all_axes(self) -> tuple[tuple, ...]:
        a = self.axes
        return (
            a.segments, a.dtype, a.shard_count, a.on_disk_payload,
            a.m, a.ef_construct, a.index_on_disk, a.indexing_threshold,
            a.quant_variant, a.always_ram,
            a.ef_search, a.batch_size, a.top_k, a.concurrency, a.rescore,
        )

    def enumerate(self) -> list[Candidate]:
        return [self._make(c) for c in itertools.product(*self._all_axes())]

    def _random_search(self, rng: np.random.Generator) -> Search:
        a = self.axes
        return Search(
            ef_search=a.ef_search[rng.integers(len(a.ef_search))],
            batch_size=a.batch_size[rng.integers(len(a.batch_size))],
            top_k=a.top_k[rng.integers(len(a.top_k))],
            concurrency=a.concurrency[rng.integers(len(a.concurrency))],
            rescore=a.rescore[rng.integers(len(a.rescore))],
        )

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        exclude: set[tuple] | None = None,
        *,
        bias_quant_keys: tuple[tuple, ...] = (),
        bias_fraction: float = 0.5,
    ) -> list[Candidate]:
        """Up to `n` distinct candidates whose `search_key` is not in
        `exclude`. Small spaces are enumerated and subsampled exactly;
        large ones are rejection-sampled (bounded attempts, so a nearly
        exhausted space returns fewer than `n` rather than spinning).

        When `bias_quant_keys` (already-built artifacts) is non-empty,
        roughly `bias_fraction` of the draws are search-level variations of
        those existing artifacts — the space's dense cheap region — and the
        rest stay uniform over the whole space so fresh artifacts keep
        getting proposed."""
        exclude = exclude or set()
        total = self.size()
        if total <= 4 * n:
            pool = [c for c in self.enumerate() if c.search_key not in exclude]
            if len(pool) <= n:
                return pool
            picks = rng.choice(len(pool), size=n, replace=False)
            return [pool[i] for i in picks]

        axes = self._all_axes()
        out: list[Candidate] = []
        seen: set[tuple] = set()
        for _ in range(20 * n):
            if bias_quant_keys and rng.random() < bias_fraction:
                qk = bias_quant_keys[rng.integers(len(bias_quant_keys))]
                cand = candidate_from_quant_key(qk, self._random_search(rng))
            else:
                choice = tuple(ax[rng.integers(len(ax))] for ax in axes)
                cand = self._make(choice)
            key = cand.search_key
            if key in seen or key in exclude:
                continue
            seen.add(key)
            out.append(cand)
            if len(out) == n:
                break
        return out

    def children(
        self,
        cand: Candidate,
        *,
        ef_search: tuple[int, ...],
        batch_size: tuple[int, ...],
        max_children: int,
        exclude: set[tuple] | None = None,
    ) -> list[Candidate]:
        """Cheap search-level siblings of `cand` (same quant_key), used to
        amortize an expensive build: the ef_search x batch_size grid around
        the selected point, minus the point itself and `exclude`, capped at
        `max_children`. Ordered ef_search-fastest (the full ef_search sweep at
        the first batch size comes first), so the cap trims batch-size
        duplication before it trims ef_search coverage."""
        exclude = exclude or set()
        out = []
        for bs in batch_size:
            for efs in ef_search:
                child = cand.with_search(
                    replace(cand.search, ef_search=efs, batch_size=bs)
                )
                key = child.search_key
                if key == cand.search_key or key in exclude:
                    continue
                out.append(child)
                if len(out) == max_children:
                    return out
        return out
