"""Typed config for the embedding pipeline (pydantic).

Mirrors the spirit of the Rust crates' serde config: validate the YAML up front,
fail with a clear message, and hand the rest of the code typed values instead of
``dict.get(...)`` chains.

Three kinds of section:

* **Typed knobs** — `pipeline` (and the structural shape) are fully modelled, so
  defaults live in one place and typos are caught (`extra="forbid"`).
* **Embedder entries** — `embedders:` is a list; each entry types the fields the
  pipeline itself consumes (name / kind / type / model / input_column / modality
  / output_column / max_length / pooling) and passes everything else through to
  the backend constructor, so adding a backend param never means touching this
  file.
* **Flexible sections** — `source` / `storage` / `chunking` each carry a `type`
  (or `strategy`) plus backend-specific kwargs, passed straight to the
  constructor via [`build_dict`][TypedSection.build_dict].

Cross-entry invariants (unique names/columns, chunking × multi-input, modality ×
chunking) are validated here so a bad manifest dies at launch, not mid-run.
"""

from __future__ import annotations

import os
import re
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nova_embed.media import Modality
from nova_embed.models import OutputKind

_ENV_RE = re.compile(r"\$\{([^}]+)\}")

# Chunker strategies that don't split (one row in, one row out). Multi-input and
# non-text configs are restricted to these.
NON_SPLITTING_STRATEGIES = {"passthrough"}


class TypedSection(BaseModel):
    """A section dispatched on `type`, with backend-specific kwargs allowed."""

    model_config = ConfigDict(extra="allow")
    type: str

    def build_dict(self) -> dict:
        """The original mapping (`type` + all kwargs) for the registry builders."""
        return {"type": self.type, **(self.__pydantic_extra__ or {})}


class ChunkingConfig(BaseModel):
    """Selected by `strategy`; remaining keys are constructor kwargs.

    The chunker always operates on THE input column — configs with more than one
    distinct input column may only use a non-splitting strategy (validated on
    EmbedConfig), so the target is never ambiguous and needs no `column:` key.
    """

    model_config = ConfigDict(extra="allow")
    strategy: str = "passthrough"

    def build_dict(self) -> dict:
        return {"strategy": self.strategy, **(self.__pydantic_extra__ or {})}

    @property
    def splits(self) -> bool:
        return self.strategy not in NON_SPLITTING_STRATEGIES


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "s3"

    def build_dict(self) -> dict:
        return {"type": self.type, **(self.__pydantic_extra__ or {})}


class PoolingConfig(BaseModel):
    """Derive a dense column from a multivector entry's output (mean/max/cls/last).

    Nested under a multivector embedder entry — pooling only makes sense there.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["mean", "max", "cls", "last"]
    normalize: bool = True
    # column for the derived dense output; defaults to "{entry.name}_pooled"
    output_column: str | None = None


class EmbedderEntry(BaseModel):
    """One embedder in the `embedders:` list.

    `kind` (output shape) and `type` (backend implementation) are separate axes:
    the same backend name can exist for several kinds (`sentence_transformer` is
    both a dense and a sparse backend). `modality` says how to decode
    `input_column`'s values — it is REQUIRED, no default: embedding the wrong
    modality for hours is far more expensive than typing `modality: text`.

    Unknown keys are backend constructor kwargs (batch_size, dtype, device, …).
    """

    model_config = ConfigDict(extra="allow")

    name: str
    kind: OutputKind
    type: str
    model: str | None = None
    input_column: str
    modality: Modality
    # parquet column for this entry's output; defaults to "{name}_embedding"
    output_column: str | None = None
    # truncate this entry's (text) input to N characters before embedding
    max_length: int | None = Field(default=None, gt=0)
    pooling: PoolingConfig | None = None

    @property
    def column(self) -> str:
        return self.output_column or f"{self.name}_embedding"

    @property
    def pooled_column(self) -> str | None:
        if self.pooling is None:
            return None
        return self.pooling.output_column or f"{self.name}_pooled"

    def backend_kwargs(self) -> dict:
        """Constructor kwargs for the backend: `model` + all unknown keys."""
        kwargs = dict(self.__pydantic_extra__ or {})
        if self.model is not None:
            kwargs["model"] = self.model
        return kwargs

    @model_validator(mode="after")
    def _check_entry(self) -> "EmbedderEntry":
        if self.pooling is not None and self.kind != OutputKind.MULTIVECTOR:
            raise ValueError(
                f"embedder {self.name!r}: pooling derives a dense column from "
                f"multivector output; it is not valid on kind={self.kind.value!r}"
            )
        if self.max_length is not None and self.modality != Modality.TEXT:
            raise ValueError(
                f"embedder {self.name!r}: max_length is character truncation and "
                f"only applies to modality: text (got {self.modality.value!r})"
            )
        return self


class PipelineConfig(BaseModel):
    """All the pipeline knobs, typed with defaults in one place."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = 10_000
    # A single model instance is serialized by an encode lock, so workers beyond
    # ~2 just contend on it. 2 is enough to keep the device fed (one prepping the
    # next chunk while one encodes). Bump only if embedding is genuinely parallel.
    num_workers: int = 2
    flush_threshold: int = 100_000
    row_group_size: int | None = None
    # What to do with a row where some embedder's input column is empty:
    #   skip  — drop the row (default; matches "no nulls in the output parquet").
    #           Skips are counted and reported in the manifest, never silent.
    #   null  — keep the row, write a null embedding for the empty input(s).
    #   error — abort the run on the first empty input (for "my data is clean,
    #           prove me wrong" runs).
    # A row where EVERY input is empty has nothing to embed: skipped under both
    # `skip` and `null`, and aborts under `error` like any other empty input.
    on_empty_input: Literal["skip", "null", "error"] = "skip"
    # Columns to drop from the OUTPUT after embedding (they never reach the
    # parquet, or the flush buffer's memory). This is how you embed a column
    # without carrying it through — e.g. a raw-bytes image column that would
    # bloat the output. Distinct from source.exclude_columns, which drops
    # columns BEFORE embedding and therefore can't touch an input_column.
    drop_columns: list[str] = Field(default_factory=list)
    # shard_by_rank=true  -> "rank00/batch_*.parquet" (subdir per rank)
    # shard_by_rank=false -> "rank00_batch_*.parquet" (flat)
    shard_by_rank: bool = False
    # Stamp each embedded row with where it came from: `source_file_name` (the
    # original parquet path in the source repo) and `source_row_number` (its row
    # index WITHIN that file). Lets you trace any embedding back to its origin —
    # the same (file, row) coordinate nova-load uses to derive point ids.
    include_source_provenance: bool = False


class EmbedConfig(BaseModel):
    """Top-level embedding config.

    `extra="allow"` so orchestration-only blocks (e.g. `resources:` for SkyPilot)
    are tolerated and ignored by the local pipeline.
    """

    model_config = ConfigDict(extra="allow")

    source: TypedSection
    embedders: list[EmbedderEntry] = Field(min_length=1)
    chunking: ChunkingConfig | None = None
    storage: StorageConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @property
    def input_specs(self) -> dict[str, Modality]:
        """input_column -> modality, across all entries (validated consistent)."""
        return {e.input_column: e.modality for e in self.embedders}

    @model_validator(mode="after")
    def _check_cross_entry(self) -> "EmbedConfig":
        # unique entry names
        names = [e.name for e in self.embedders]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate embedder names: {dupes}")

        # unique output columns (including pooled outputs)
        columns: list[str] = []
        for e in self.embedders:
            columns.append(e.column)
            if e.pooled_column:
                columns.append(e.pooled_column)
        dupes = sorted({c for c in columns if columns.count(c) > 1})
        if dupes:
            raise ValueError(
                f"output column collision: {dupes}. Give each embedder entry a "
                f"distinct output_column (and pooling.output_column)."
            )

        # one modality per input column — two entries reading the same column
        # through different decoders is a config mistake, not a feature
        by_column: dict[str, set[Modality]] = {}
        for e in self.embedders:
            by_column.setdefault(e.input_column, set()).add(e.modality)
        conflicts = {c: sorted(m.value for m in ms) for c, ms in by_column.items() if len(ms) > 1}
        if conflicts:
            raise ValueError(
                f"conflicting modalities for the same input column: {conflicts}"
            )

        # drop_columns operates on SOURCE columns; naming an embedding output
        # here is a config mistake (you'd configure the entry away instead)
        dropped_outputs = sorted(set(self.pipeline.drop_columns) & set(columns))
        if dropped_outputs:
            raise ValueError(
                f"pipeline.drop_columns lists embedding output column(s) "
                f"{dropped_outputs}. Remove the embedder entry (or its pooling) "
                f"instead of dropping its output."
            )

        chunking = self.chunking or ChunkingConfig()
        if chunking.splits:
            input_columns = sorted(by_column)
            if len(input_columns) > 1:
                raise ValueError(
                    f"You've specified more than one input_column "
                    f"({input_columns}) together with chunking strategy "
                    f"{chunking.strategy!r}. Splitting one column produces "
                    f"inconsistent row counts across embedders and is considered "
                    f"undefined behavior. Use strategy: passthrough, or split the "
                    f"work into separate pipeline runs."
                )
            non_text = [
                e.name for e in self.embedders if e.modality != Modality.TEXT
            ]
            if non_text:
                raise ValueError(
                    f"chunking strategy {chunking.strategy!r} splits text, but "
                    f"embedder(s) {non_text} declare a non-text modality. Use "
                    f"strategy: passthrough for non-text inputs."
                )
        return self


def load_config(path: str) -> EmbedConfig:
    """Read, env-expand, and validate an embedding config file.

    `${VAR}` and `${VAR:-default}` references are expanded from the environment
    first (matching the Rust crates); an unset reference with no default raises.
    """
    with open(path) as f:
        raw = f.read()
    data = yaml.safe_load(expand_env(raw)) or {}
    return EmbedConfig.model_validate(data)


def expand_env(text: str) -> str:
    """Expand `${VAR}` / `${VAR:-default}` against the environment."""

    def repl(m: re.Match) -> str:
        name, _, default = m.group(1).partition(":-")
        val = os.environ.get(name)
        if val:
            return val
        if "-" in m.group(1) or default != "":
            return default
        raise ValueError(
            f"environment variable '{name}' referenced in config is not set; "
            f"set it or supply a default with ${{{name}:-...}}"
        )

    return _ENV_RE.sub(repl, text)
