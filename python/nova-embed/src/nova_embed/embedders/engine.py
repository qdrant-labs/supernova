"""
EmbeddingEngine — runs every configured embedder entry over a batch of rows.

Built from the config's `embedders:` list by [`build_engine`][build_engine],
which owns the three launch-time concerns:

* **validation before weights** — unknown (kind, type) pairs and unsupported
  modalities fail on the registry *class*, before any model download starts.
* **instance sharing** — two entries naming the same backend config (e.g. CLIP
  on the image column AND the caption column) share one loaded model.
* **hybrid fusion** — a dense + sparse entry pair pointing at the same
  sentence_transformer model and the same input column collapse into a single
  forward pass that feeds both output columns.

At run time the worker calls `engine.embed(rows)` with plain row dicts and gets
back `{entry_name: [embedding | None, ...]}`. The engine decodes each distinct
input column exactly once (through nova_embed.media, so backends only ever see
canonical objects), masks out empty inputs, and scatters results back so every
output list is row-aligned — None marks an empty input kept by
on_empty_input="null".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from nova_embed import media
from nova_embed.config import EmbedderEntry
from nova_embed.embedders.base import Embedder
from nova_embed.media import Modality
from nova_embed.models import Embedding, MultiVectorEmbedding, OutputKind
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)

POOLING_TYPES = {"mean", "max", "cls", "last"}


def pool_multivector(
    mv: MultiVectorEmbedding, pool_type: str, normalize: bool
) -> list[float]:
    arr = np.asarray(mv.vectors, dtype=np.float32)
    if pool_type == "mean":
        pooled = arr.mean(axis=0)
    elif pool_type == "max":
        pooled = arr.max(axis=0)
    elif pool_type == "cls":
        pooled = arr[0]
    elif pool_type == "last":
        pooled = arr[-1]
    else:
        raise ValueError(
            f"Unknown pooling type: {pool_type!r}. Choose from {sorted(POOLING_TYPES)}."
        )

    if normalize:
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

    return pooled.tolist()


@dataclass
class OutputSpec:
    """One output column the engine produces — what the writer and manifest need."""

    name: str  # entry name (keys the embed() result dict)
    column: str  # parquet column
    kind: OutputKind
    model_name: str
    dimensions: int | None
    max_tokens: int | None
    input_column: str
    modality: Modality


@dataclass
class _Unit:
    """One forward pass: an embedder bound to its input spec and output name(s)."""

    embedder: Any  # Embedder, or SentenceTransformerHybridEmbedder for fused units
    input_column: str
    modality: Modality
    max_length: int | None
    name: str | None = None  # plain unit: the entry name
    dense_name: str | None = None  # fused unit: dense entry name
    sparse_name: str | None = None  # fused unit: sparse entry name
    # pooling (multivector entries only): derived dense output
    pooled_name: str | None = None
    pooling_type: str | None = None
    pooling_normalize: bool = True


class EmbeddingEngine:
    def __init__(self, units: list[_Unit], output_specs: list[OutputSpec]):
        if not units:
            raise ValueError("EmbeddingEngine needs at least one embedder unit")
        self._units = units
        self._output_specs = output_specs

    @property
    def output_specs(self) -> list[OutputSpec]:
        return self._output_specs

    @property
    def input_specs(self) -> dict[str, Modality]:
        return {u.input_column: u.modality for u in self._units}

    @property
    def model_name(self) -> str:
        """Primary model name for logging."""
        return self._output_specs[0].model_name

    async def embed(self, rows: list[dict]) -> dict[str, list[Embedding | None]]:
        # Decode each distinct input column once: empties masked out, the rest
        # turned canonical. Backends never see raw transport forms.
        decoded: dict[str, tuple[list[Any], list[bool]]] = {}
        for col, modality in self.input_specs.items():
            raw = [row.get(col) for row in rows]
            mask = [media.is_empty(v, modality) for v in raw]
            values = [
                media.decode(v, modality) for v, empty in zip(raw, mask) if not empty
            ]
            decoded[col] = (values, mask)

        out: dict[str, list[Embedding | None]] = {}
        for unit in self._units:
            values, mask = decoded[unit.input_column]
            if unit.max_length is not None:
                values = [v[: unit.max_length] for v in values]

            if unit.dense_name is not None:  # fused dense+sparse forward pass
                if values:
                    dense, sparse = await unit.embedder.embed(values)
                else:
                    dense, sparse = [], []
                out[unit.dense_name] = _scatter(dense, mask)
                out[unit.sparse_name] = _scatter(sparse, mask)
                continue

            results = await unit.embedder.embed(values) if values else []
            out[unit.name] = _scatter(results, mask)

            if unit.pooled_name is not None:
                out[unit.pooled_name] = [
                    pool_multivector(mv, unit.pooling_type, unit.pooling_normalize)
                    if mv is not None
                    else None
                    for mv in out[unit.name]
                ]
        return out


def _scatter(results: list, mask: list[bool]) -> list:
    """Re-align a compacted result list with its empty-input mask (None = empty)."""
    expected = len(mask) - sum(mask)
    if len(results) != expected:
        raise RuntimeError(
            f"embedder returned {len(results)} results for {expected} inputs"
        )
    it = iter(results)
    return [None if empty else next(it) for empty in mask]


def _cache_key(kind: OutputKind, type_: str, kwargs: dict) -> str:
    return f"{kind.value}/{type_}::{sorted(kwargs.items())!r}"


def _can_fuse(dense: EmbedderEntry, sparse: EmbedderEntry) -> bool:
    """Same ST model reading the same input → one forward pass for both."""
    return (
        dense.type == sparse.type == "sentence_transformer"
        and dense.model is not None
        and dense.model == sparse.model
        and dense.input_column == sparse.input_column
        and dense.max_length == sparse.max_length
    )


def build_engine(entries: list[EmbedderEntry]) -> EmbeddingEngine:
    """Instantiate backends for every entry and assemble the engine.

    Order of operations matters: registry lookup and modality validation run on
    the CLASSES first, so a typo'd type or wrong modality kills the launch before
    any weights download.
    """
    # -- validate everything before loading anything -------------------------
    for e in entries:
        cls = EMBEDDERS.get(e.kind.value, e.type)
        if e.modality not in cls.supported_modalities:
            supported = sorted(m.value for m in cls.supported_modalities)
            raise ValueError(
                f"embedder {e.name!r}: backend {e.type!r} ({cls.__name__}) does not "
                f"support modality {e.modality.value!r}. Supported: {supported}"
            )

    # -- hybrid fusion: pair up dense+sparse entries on the same ST model ----
    fused: list[tuple[EmbedderEntry, EmbedderEntry]] = []
    taken: set[str] = set()
    dense_entries = [e for e in entries if e.kind == OutputKind.DENSE]
    sparse_entries = [e for e in entries if e.kind == OutputKind.SPARSE]
    for d in dense_entries:
        partner = next(
            (s for s in sparse_entries if s.name not in taken and _can_fuse(d, s)),
            None,
        )
        if partner is not None:
            fused.append((d, partner))
            taken.update((d.name, partner.name))

    units: list[_Unit] = []
    specs: list[OutputSpec] = []
    instances: dict[str, Embedder] = {}

    if fused:
        from nova_embed.embedders.backends.sentence_transformer import (
            SentenceTransformerHybridEmbedder,
        )

    for d, s in fused:
        logger.info(
            "Fusing dense %r + sparse %r into one forward pass (%s)",
            d.name,
            s.name,
            d.model,
        )
        hybrid = SentenceTransformerHybridEmbedder(**d.backend_kwargs())
        units.append(
            _Unit(
                embedder=hybrid,
                input_column=d.input_column,
                modality=d.modality,
                max_length=d.max_length,
                dense_name=d.name,
                sparse_name=s.name,
            )
        )
        for entry, kind in ((d, OutputKind.DENSE), (s, OutputKind.SPARSE)):
            specs.append(
                OutputSpec(
                    name=entry.name,
                    column=entry.column,
                    kind=kind,
                    model_name=hybrid.model_name,
                    dimensions=hybrid.dimensions if kind == OutputKind.DENSE else None,
                    max_tokens=hybrid.max_tokens,
                    input_column=entry.input_column,
                    modality=entry.modality,
                )
            )

    # -- plain units, sharing instances when the backend config matches ------
    for e in entries:
        if e.name in taken:
            continue
        key = _cache_key(e.kind, e.type, e.backend_kwargs())
        embedder = instances.get(key)
        if embedder is None:
            cls = EMBEDDERS.get(e.kind.value, e.type)
            embedder = cls(**e.backend_kwargs())
            instances[key] = embedder
        else:
            logger.info(
                "Entry %r shares the already-loaded %s instance", e.name, e.type
            )

        units.append(
            _Unit(
                embedder=embedder,
                input_column=e.input_column,
                modality=e.modality,
                max_length=e.max_length,
                name=e.name,
                pooled_name=e.pooled_column,
                pooling_type=e.pooling.type if e.pooling else None,
                pooling_normalize=e.pooling.normalize if e.pooling else True,
            )
        )
        specs.append(
            OutputSpec(
                name=e.name,
                column=e.column,
                kind=e.kind,
                model_name=embedder.model_name,
                dimensions=embedder.dimensions,
                max_tokens=embedder.max_tokens,
                input_column=e.input_column,
                modality=e.modality,
            )
        )
        if e.pooled_column:
            specs.append(
                OutputSpec(
                    name=e.pooled_column,
                    column=e.pooled_column,
                    kind=OutputKind.DENSE,
                    model_name=f"{embedder.model_name} ({e.pooling.type}-pooled)",
                    dimensions=embedder.dimensions,
                    max_tokens=embedder.max_tokens,
                    input_column=e.input_column,
                    modality=e.modality,
                )
            )

    return EmbeddingEngine(units, specs)
