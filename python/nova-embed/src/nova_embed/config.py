"""Typed config for the embedding pipeline (pydantic).

Mirrors the spirit of the Rust crates' serde config: validate the YAML up front,
fail with a clear message, and hand the rest of the code typed values instead of
``dict.get(...)`` chains.

Two kinds of section:

* **Typed knobs** — `pipeline` (and the structural shape) are fully modelled, so
  defaults live in one place and typos are caught (`extra="forbid"`).
* **Flexible sections** — `source` / `*_embedder` / `storage` / `chunking` each
  carry a `type` (or `strategy`) plus backend-specific kwargs that vary per
  implementation. These allow extra fields and pass them straight to the
  constructor via [`build_dict`][TypedSection.build_dict], so adding an embedder
  param never means touching this file.
"""

from __future__ import annotations

import os
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


class TypedSection(BaseModel):
    """A section dispatched on `type`, with backend-specific kwargs allowed."""

    model_config = ConfigDict(extra="allow")
    type: str

    def build_dict(self) -> dict:
        """The original mapping (`type` + all kwargs) for the registry builders."""
        return {"type": self.type, **(self.__pydantic_extra__ or {})}


class ChunkingConfig(BaseModel):
    """Selected by `strategy`; remaining keys are constructor kwargs."""

    model_config = ConfigDict(extra="allow")
    strategy: str = "passthrough"

    def build_dict(self) -> dict:
        return {"strategy": self.strategy, **(self.__pydantic_extra__ or {})}


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "s3"

    def build_dict(self) -> dict:
        return {"type": self.type, **(self.__pydantic_extra__ or {})}


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
    max_text_length: int | None = None
    dense_embedding_column: str = "dense_embedding"
    sparse_embedding_column: str = "sparse_embedding"
    multivector_embedding_column: str = "multivector_embedding"
    rendered_text_column: str = "text"
    # shard_by_rank=true  -> "rank00/batch_*.parquet" (subdir per rank)
    # shard_by_rank=false -> "rank00_batch_*.parquet" (flat)
    shard_by_rank: bool = False


class EmbedConfig(BaseModel):
    """Top-level embedding config.

    `extra="allow"` so orchestration-only blocks (e.g. `resources:` for SkyPilot)
    are tolerated and ignored by the local pipeline.
    """

    model_config = ConfigDict(extra="allow")

    source: TypedSection
    dense_embedder: TypedSection | None = None
    sparse_embedder: TypedSection | None = None
    multivector_embedder: TypedSection | None = None
    chunking: ChunkingConfig | None = None
    storage: StorageConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


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
