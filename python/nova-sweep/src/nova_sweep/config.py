"""Typed sweep config (pydantic), with `${VAR}` env expansion — same two rules
as `nova-bf`'s `config.py` (`${VAR}`, `${VAR:-default}`; sweep configs don't
need a literal-`$` escape, so `$$` isn't implemented here).

    corpus:          the pre-embedded parquet corpus to load (nova-load's
                      datasource.path) plus which column holds the dense vector
    queries:         where query vectors + ground truth (`hit_ids`) live —
                      exactly nova-storm's own `query.source` shape, computed
                      separately (e.g. `nova bf compute`)
    target:          the instance under test, dispatched on `type` (see
                      `nova_sweep.backends`) plus how to handle a pre-existing
                      same-named collection. `type` defaults to `qdrant` — the
                      only backend implemented — so configs written before
                      this field existed still parse.
    data_layouts:    axis of structural nova-load params forcing a fresh load
    index_variants:  axis of nova-load `reindex`-patchable params (HNSW,
                      quantization, optimizers)
    searches:        axis of nova-storm query-time params
    output:          where the combined report parquet lands
"""

from __future__ import annotations

import os
import re

from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, field_validator

from nova_sweep.backends import TargetConfigBase, parse_target

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def expand_env(text: str) -> str:
    """Expand `${VAR}` / `${VAR:-default}` against the environment (on raw YAML)."""

    def repl(m: re.Match) -> str:
        name, _, default = m.group(1).partition(":-")
        val = os.environ.get(name)
        if val:
            return val
        if ":-" in m.group(1):
            return default
        raise ValueError(
            f"environment variable '{name}' referenced in config is not set; "
            f"set it or supply a default with ${{{name}:-...}}"
        )

    return _ENV_RE.sub(repl, text)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    dense_column: str = "dense_embedding"


class QueriesConfig(BaseModel):
    """Exactly nova-storm's own `query.source` shape — passed through
    unchanged into every generated storm config (this is what keeps
    ground-truth computation out of `nova-sweep`'s scope)."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    column: str
    ground_truth_column: str | None = None
    limit: int = 1000


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: CorpusConfig
    queries: QueriesConfig
    target: TargetConfigBase
    output: OutputConfig
    # Each axis is a flat dotted-path-key -> list-of-values grid.
    # Deliberately untyped (dict[str, list]) — the values' shape is nova-load's
    # / nova-storm's own config schema, not nova-sweep's to validate; nova-load
    # / nova-storm will reject an invalid value when the generated config is
    # actually run.
    data_layouts: dict[str, list] = {}
    index_variants: dict[str, list] = {}
    searches: dict[str, list] = {}

    @field_validator("target", mode="before")
    @classmethod
    def _dispatch_target(cls, value: object) -> TargetConfigBase:
        """Resolve `target.type` to its backend's own config model (default:
        `qdrant`, for configs predating this field) — see `nova_sweep.backends`."""
        if isinstance(value, TargetConfigBase):
            return value
        return parse_target(value)


def load_config(path: str) -> tuple[SweepConfig, str]:
    """Parse a sweep config. Returns `(config, sweep_name)` — `sweep_name`
    (used as the collection-name prefix, `{sweep_name}_{data_layout_name}`) is
    the config file's stem, matching `nova-dist`'s own pool-naming convention
    (`f"nova-{tool}-{Path(config).stem}"`) rather than a separate schema field.
    """
    with open(path) as f:
        raw = expand_env(f.read())
    cfg = SweepConfig.model_validate(yaml.safe_load(raw))
    return cfg, Path(path).stem
