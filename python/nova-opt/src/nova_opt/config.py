"""Typed tuner config (pydantic), with the same `${VAR}` / `${VAR:-default}`
env expansion convention as nova-sweep / nova-bf.

    data_csv:      the recall-classifier training data collected so far
    workload:      the dataset under tuning — corpus/queries locations plus
                   how to run stats extraction and storm evaluations
    target:        the live instance under test (qdrant only today); ignored
                   when `replay` is set
    replay:        optional path to a measured results table (parquet/CSV);
                   evaluates candidates offline instead of a live cluster
    space:         the hierarchical Layout x Index x Quantization x Search
                   axes candidates are drawn from
    optimizer:     acquisition and budget knobs (strategy selects baselines)
    scheduler:     cheap-child amortization after expensive builds
    cost_priors:   per-level seconds before any measurement exists
"""

from __future__ import annotations

import os
import re

from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nova_opt.acquisition import STRATEGIES
from nova_opt.cost import CostModel
from nova_opt.optimizer import OptSettings
from nova_opt.space import QUANT_VARIANTS, ConfigSpace, SpaceAxes
from nova_opt.stats import SUPPORTED_METRICS, StatsParams, resolve_metric

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
    """nova-storm's own `query.source` shape, passed through unchanged."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    column: str
    ground_truth_column: str | None = None
    limit: int = 1000


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: CorpusConfig
    queries: QueriesConfig
    # data.csv vocabulary: COSINE | L2 | IP (case-insensitive aliases accepted)
    distance_metric: str = "COSINE"
    # nova-storm run length per search-point measurement
    duration_s: float = 10.0

    @field_validator("distance_metric")
    @classmethod
    def _known_metric(cls, v: str) -> str:
        resolve_metric(v)  # raises with the full alias list on a typo
        return v


class StatsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_size: int = 1000
    pair_sample_size: int = 100_000
    nn_query_sample_size: int = 256
    nn_reference_sample_size: int = 5000
    knn_k: int = 100
    seed: int = 0
    full_pass_row_limit: int = 2_000_000

    def params(self) -> StatsParams:
        return StatsParams(**self.model_dump())


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "qdrant"
    url: str
    api_key: str | None = None

    @field_validator("type")
    @classmethod
    def _qdrant_only(cls, v: str) -> str:
        if v != "qdrant":
            raise ValueError(f"unknown target type '{v}'; only 'qdrant' is implemented")
        return v


class SpaceConfig(BaseModel):
    """Value lists per axis, grouped by artifact level. Defaults give a small
    but real space; every list must be non-empty."""

    model_config = ConfigDict(extra="forbid")

    # layout level
    segments: list[int] = [8]
    dtype: list[str] = ["float32"]
    shard_count: list[int] = [1]
    on_disk_payload: list[bool] = [False]
    # index level
    m: list[int] = [16, 32]
    ef_construct: list[int] = [128]
    index_on_disk: list[bool] = [False]
    indexing_threshold: list[int | None] = [None]
    # quantization level
    quant_variant: list[str] = ["none"]
    always_ram: list[bool] = [True]
    # search level
    ef_search: list[int] = [16, 32, 64, 128]
    batch_size: list[int] = [1]
    top_k: list[int] = [10]
    concurrency: list[int] = [1]
    rescore: list[bool | None] = [None]

    @field_validator("quant_variant")
    @classmethod
    def _known_variants(cls, v: list[str]) -> list[str]:
        unknown = [x for x in v if x not in QUANT_VARIANTS]
        if unknown:
            raise ValueError(
                f"unknown quantization variants {unknown}; "
                f"available: {sorted(QUANT_VARIANTS)}"
            )
        return v

    def axes(self) -> SpaceAxes:
        d = {k: tuple(v) for k, v in self.model_dump().items()}
        return SpaceAxes(**d)

    def space(self) -> ConfigSpace:
        return ConfigSpace(self.axes())


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_recall: float = 0.90
    strategy: str = "full"
    # both must be positive: budget_seconds=0 (or max_evaluations=0) makes
    # the optimizer loop's `while` guard false from the start, silently
    # producing zero trials instead of a clear config error
    budget_seconds: float = Field(3600.0, gt=0)
    max_evaluations: int = Field(200, gt=0)
    gamma: float = 0.5
    beta: float = 0.1
    prune_min_probability: float = 0.3
    n_candidates: int = Field(256, gt=0)
    seed: int = 0
    # anneal gamma toward 0 as the budget depletes (cost-cooling)
    cost_cooling: bool = True
    # fraction of proposals drawn as search-level variations of artifacts
    # that already exist (the rest stay uniform over the whole space)
    bias_fraction: float = 0.5
    # shrinkage prior of the online recall recalibration (higher = slower
    # to trust this run's measured recalls over the offline classifier)
    online_prior_strength: float = 4.0
    # measured recalls (with feature variation) needed before the run-local
    # recall GP is consulted at all
    recall_gp_min_observations: int = 5

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, v: str) -> str:
        if v not in STRATEGIES:
            raise ValueError(f"unknown strategy '{v}'; available: {STRATEGIES}")
        return v


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ef_search: list[int] = [16, 32, 64, 128, 256]
    batch_size: list[int] = [1, 8, 32, 128]
    max_children: int = 8


class CostPriorsConfig(BaseModel):
    """Seconds per level before anything has been measured — deliberately
    rough; the EMA takes over after the first real build."""

    model_config = ConfigDict(extra="forbid")

    # must be positive: the ridge-regression clamp band ([EMA/5, EMA*5], see
    # cost.py) and the EMA itself are meaningless (or sign-inverting) seeded
    # from a zero or negative prior
    layout_s: float = Field(600.0, gt=0)
    index_s: float = Field(300.0, gt=0)
    quant_s: float = Field(120.0, gt=0)
    search_s: float = Field(30.0, gt=0)
    ema_alpha: float = Field(0.3, gt=0, le=1)

    def model(self) -> CostModel:
        return CostModel(
            priors={
                "layout": self.layout_s,
                "index": self.index_s,
                "quant": self.quant_s,
                "search": self.search_s,
            },
            alpha=self.ema_alpha,
        )


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class OptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_csv: str = "data.csv"
    # trained-classifier directory: loaded when it exists, else trained from
    # data_csv and saved here (null = train fresh each run, save nothing)
    recall_model_dir: str | None = None
    workload: WorkloadConfig
    target: TargetConfig | None = None
    replay: str | None = None
    output: OutputConfig
    space: SpaceConfig = SpaceConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    stats: StatsConfig = StatsConfig()
    cost_priors: CostPriorsConfig = CostPriorsConfig()

    def settings(self) -> OptSettings:
        o, s = self.optimizer, self.scheduler
        return OptSettings(
            target_recall=o.target_recall,
            strategy=o.strategy,
            budget_seconds=o.budget_seconds,
            max_evaluations=o.max_evaluations,
            gamma=o.gamma,
            beta=o.beta,
            prune_min_probability=o.prune_min_probability,
            n_candidates=o.n_candidates,
            seed=o.seed,
            cost_cooling=o.cost_cooling,
            bias_fraction=o.bias_fraction,
            online_prior_strength=o.online_prior_strength,
            recall_gp_min_observations=o.recall_gp_min_observations,
            children_ef_search=tuple(s.ef_search),
            children_batch_size=tuple(s.batch_size),
            max_children=s.max_children,
        )


def load_config(path: str) -> tuple[OptConfig, str]:
    """Parse a tuner config. Returns `(config, run_name)` — `run_name` (used
    as the collection-name prefix on live runs) is the config file's stem,
    the same convention as nova-sweep/nova-dist."""
    with open(path) as f:
        raw = expand_env(f.read())
    cfg = OptConfig.model_validate(yaml.safe_load(raw))
    if cfg.replay is None and cfg.target is None:
        raise ValueError("config needs either a live `target:` or a `replay:` table")
    return cfg, Path(path).stem
