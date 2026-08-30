"""
EmbeddingEngine — runs every configured embedder entry over a batch of rows.

Built from the config's `embedders:` list by [`build_engine`][build_engine],
which owns the three launch-time concerns:

* **validation before weights** — unknown (kind, type) pairs and unsupported
  modalities fail on the registry *class*, before any model download starts.
* **instance sharing** — two entries naming the same backend config (e.g. CLIP
  on the image column AND the caption column) share one loaded model.
* **fusion** — entries pointing at the same model and input column collapse
  into a single forward pass feeding all their output columns, when the
  backend registered a FusedEmbedder for the `type` (e.g. bge_m3's three
  heads). Detection is automatic; configs never opt in.

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
from nova_embed.manifest import hf_revision, redact
from nova_embed.media import Modality
from nova_embed.models import Embedding, MultiVectorEmbedding, OutputKind
from nova_embed.registry import EMBEDDERS, FUSED_EMBEDDERS

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
    # multimodal entries: a "modality=column,…" display string, not one column
    input_column: str
    modality: Modality
    instruction: str | None = None
    backend: str = ""
    backend_kwargs: dict | None = None
    model_revision: str | None = None
    max_length: int | None = None
    pooling: dict | None = None


@dataclass
class _Unit:
    """One forward pass: an embedder bound to its input spec and output name(s)."""

    embedder: Any  # Embedder, or FusedEmbedder for fused units
    input_column: str | None
    modality: Modality
    max_length: int | None
    name: str | None = None  # plain unit: the entry name
    # multimodal unit: part modality -> column (input_column is None then)
    parts: dict[Modality, str] | None = None
    # fused unit: output kind -> entry name (name is None then)
    fused_names: dict[OutputKind, str] | None = None
    # pooling (multivector entries only): derived dense output
    pooled_name: str | None = None
    pooling_type: str | None = None
    pooling_normalize: bool = True

    @property
    def input_cols(self) -> dict[str, Modality]:
        """column -> DECODE modality for this unit (part modalities, never
        'multimodal' — that value has no loader)."""
        if self.parts is not None:
            return {col: m for m, col in self.parts.items()}
        return {self.input_column: self.modality}


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
        specs: dict[str, Modality] = {}
        for u in self._units:
            specs.update(u.input_cols)
        return specs

    @property
    def input_groups(self) -> list[dict[str, Modality]]:
        """One column->modality group per unit — the unit of the empty policy
        (a group is empty only when ALL of its columns are)."""
        return [u.input_cols for u in self._units]

    @property
    def model_name(self) -> str:
        """Primary model name for logging."""
        return self._output_specs[0].model_name

    async def embed(self, rows: list[dict]) -> dict[str, list[Embedding | None]]:
        # Decode each distinct input column once, ROW-ALIGNED (None at empty
        # positions): multimodal units zip several columns back together per
        # row, which compacted lists can't support. Backends never see raw
        # transport forms.
        decoded: dict[str, tuple[list[Any], list[bool]]] = {}
        for col, modality in self.input_specs.items():
            raw = [row.get(col) for row in rows]
            mask = [media.is_empty(v, modality) for v in raw]
            aligned = [
                None if empty else media.decode(v, modality)
                for v, empty in zip(raw, mask)
            ]
            decoded[col] = (aligned, mask)

        out: dict[str, list[Embedding | None]] = {}
        for unit in self._units:
            if unit.parts is not None:
                out[unit.name] = await self._embed_multimodal(unit, decoded)
                continue

            aligned, mask = decoded[unit.input_column]
            values = [v for v, empty in zip(aligned, mask) if not empty]
            if unit.max_length is not None:
                values = [v[: unit.max_length] for v in values]

            if unit.fused_names is not None:  # one forward pass, several kinds
                fused = await unit.embedder.embed(values) if values else {}
                for kind, entry_name in unit.fused_names.items():
                    out[entry_name] = _scatter(fused.get(kind, []), mask)
                pooling_source = unit.fused_names.get(OutputKind.MULTIVECTOR)
            else:
                results = await unit.embedder.embed(values) if values else []
                out[unit.name] = _scatter(results, mask)
                pooling_source = unit.name

            if unit.pooled_name is not None and pooling_source is not None:
                out[unit.pooled_name] = [
                    pool_multivector(mv, unit.pooling_type, unit.pooling_normalize)
                    if mv is not None
                    else None
                    for mv in out[pooling_source]
                ]
        return out

    async def _embed_multimodal(
        self, unit: _Unit, decoded: dict[str, tuple[list[Any], list[bool]]]
    ) -> list[Embedding | None]:
        """One multimodal unit over the row-aligned decoded columns.

        A row is an input when AT LEAST ONE part is present; the batch item is
        a dict of the present parts ({"text": str, "image": PIL.Image}). Only
        all-parts-empty rows are masked to None.
        """
        part_masks = {col: decoded[col][1] for col in unit.input_cols}
        n_rows = len(next(iter(part_masks.values())))
        combined = [
            all(mask[i] for mask in part_masks.values()) for i in range(n_rows)
        ]

        values: list[dict[str, Any]] = []
        for i in range(n_rows):
            if combined[i]:
                continue
            item: dict[str, Any] = {}
            for modality, col in unit.parts.items():
                aligned, mask = decoded[col]
                if mask[i]:
                    continue
                v = aligned[i]
                if modality == Modality.TEXT and unit.max_length is not None:
                    v = v[: unit.max_length]
                item[modality.value] = v
            values.append(item)

        results = await unit.embedder.embed(values) if values else []
        return _scatter(results, combined)


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


def _fusion_groups(
    entries: list[EmbedderEntry],
) -> list[tuple[type, list[EmbedderEntry]]]:
    """Groups of entries that one fused forward pass can serve.

    Entries group when everything that changes the embeddings is identical:
    backend `type`, input column, truncation, and constructor kwargs.
    `batch_size` is exempt — it's a pure throughput knob, so differing values
    still fuse (on the min, warned about at instantiation). A group needs at
    least two entries with pairwise-distinct kinds; duplicate kinds (e.g. two
    dense entries on the same model) fall back to plain units.
    """
    candidates: dict[tuple, list[EmbedderEntry]] = {}
    for e in entries:
        if e.input_columns is not None:  # multimodal entries never fuse
            continue
        cls = FUSED_EMBEDDERS.get(e.type)
        if cls is None:
            continue
        if e.kind not in cls.fusable_kinds or e.modality not in cls.supported_modalities:
            continue
        kwargs = e.backend_kwargs()
        kwargs.pop("batch_size", None)
        key = (e.type, e.input_column, e.max_length, repr(sorted(kwargs.items())))
        candidates.setdefault(key, []).append(e)

    groups: list[tuple[type, list[EmbedderEntry]]] = []
    for group in candidates.values():
        if len(group) < 2:
            continue
        kinds = [e.kind for e in group]
        if len(set(kinds)) != len(kinds):
            logger.info(
                "Not fusing entries %s: duplicate output kinds on one model",
                [e.name for e in group],
            )
            continue
        groups.append((FUSED_EMBEDDERS.get(group[0].type), group))
    return groups


def _entry_provenance(e: EmbedderEntry, effective_kwargs: dict | None = None) -> dict:
    """The manifest-only half of an OutputSpec: what produced this column.
    Secrets are stripped (see `manifest.redact`): `backend_kwargs` is "every
    unknown key in the entry", and the manifest is published next to the
    embeddings.
    """
    kwargs = e.backend_kwargs() if effective_kwargs is None else dict(effective_kwargs)
    return {
        "backend": e.type,
        "backend_kwargs": redact(kwargs),
        # The pin, so the cache is asked for the snapshot this run actually
        # loaded rather than for `main`.
        "model_revision": hf_revision(e.model, kwargs.get("revision")),
        "max_length": e.max_length,
    }


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

    units: list[_Unit] = []
    specs: list[OutputSpec] = []
    instances: dict[str, Any] = {}
    taken: set[str] = set()

    # -- fusion: entry groups whose backend registered a FusedEmbedder -------
    for cls, group in _fusion_groups(entries):
        kinds = frozenset(e.kind for e in group)
        kwargs = group[0].backend_kwargs()
        kwargs.pop("batch_size", None)
        batch_sizes = sorted(
            {
                e.backend_kwargs()["batch_size"]
                for e in group
                if "batch_size" in e.backend_kwargs()
            }
        )
        if batch_sizes:
            if len(batch_sizes) > 1:
                logger.warning(
                    "Fused entries %s declare different batch_size values %s; "
                    "one forward pass serves all of them — using the min, %s",
                    [e.name for e in group],
                    batch_sizes,
                    batch_sizes[0],
                )
            kwargs["batch_size"] = batch_sizes[0]

        key = f"fused/{group[0].type}::{sorted(k.value for k in kinds)}::{sorted(kwargs.items())!r}"
        embedder = instances.get(key)
        if embedder is None:
            logger.info(
                "Fusing entries %s into one %r forward pass (%s)",
                [e.name for e in group],
                group[0].type,
                group[0].model,
            )
            embedder = cls(kinds=kinds, **kwargs)
            instances[key] = embedder
        else:
            logger.info(
                "Fused entries %s share the already-loaded %r instance",
                [e.name for e in group],
                group[0].type,
            )

        # ≤1 pooled entry per group: only multivector entries may pool, and
        # group kinds are pairwise distinct
        mv_pooled = next((e for e in group if e.pooling is not None), None)
        units.append(
            _Unit(
                embedder=embedder,
                input_column=group[0].input_column,
                modality=group[0].modality,
                max_length=group[0].max_length,
                fused_names={e.kind: e.name for e in group},
                pooled_name=mv_pooled.pooled_column if mv_pooled else None,
                pooling_type=mv_pooled.pooling.type if mv_pooled else None,
                pooling_normalize=mv_pooled.pooling.normalize if mv_pooled else True,
            )
        )
        taken.update(e.name for e in group)
        for e in group:
            specs.append(
                OutputSpec(
                    name=e.name,
                    column=e.column,
                    kind=e.kind,
                    model_name=embedder.model_name,
                    dimensions=embedder.dimensions_for(e.kind),
                    max_tokens=embedder.max_tokens,
                    input_column=e.input_column,
                    modality=e.modality,
                    instruction=embedder.instruction,
                    **_entry_provenance(e, kwargs),
                )
            )
            if e.pooled_column:
                specs.append(
                    OutputSpec(
                        name=e.pooled_column,
                        column=e.pooled_column,
                        kind=OutputKind.DENSE,
                        model_name=f"{embedder.model_name} ({e.pooling.type}-pooled)",
                        dimensions=embedder.dimensions_for(e.kind),
                        max_tokens=embedder.max_tokens,
                        input_column=e.input_column,
                        modality=e.modality,
                        instruction=embedder.instruction,
                        pooling=e.pooling.model_dump(mode="json"),
                        **_entry_provenance(e, kwargs),
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
                parts=e.input_columns,
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
                input_column=e.input_column or e.input_display,
                modality=e.modality,
                instruction=embedder.instruction,
                **_entry_provenance(e),
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
                    input_column=e.input_column or e.input_display,
                    modality=e.modality,
                    instruction=embedder.instruction,
                    pooling=e.pooling.model_dump(mode="json"),
                    **_entry_provenance(e),
                )
            )

    return EmbeddingEngine(units, specs)
