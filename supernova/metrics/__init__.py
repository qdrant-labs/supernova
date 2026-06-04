"""Supernova metrics — a swappable sink for measurements, set once per run and
reachable from anywhere through an ambient (context-local) backend.

The model mirrors ``wandb.init`` + ``wandb.log``, except the YAML *is* the
``init`` call::

    backend = build_metrics(cfg.get("metrics"))
    set_current(backend)
    backend.start(run_name, {"node_id": ..., "command": "storm", "config": cfg})
    try:
        ...                          # anywhere downstream: metrics.log("qps", v)
    finally:
        backend.finish()

Config selects the backend by ``type`` (matching every other block), remaining
keys passed as kwargs — exactly like ``build_vectorstore``::

    metrics:
      type: postgres
      dsn: ${SN_METRICS_DB_URL}

To add a backend, implement ``MetricsBackend`` and register it below. A future
``type: ./my_backend.py`` file-path branch slots into ``build_metrics`` without
touching anything else.
"""

import contextvars

from .base import MetricsBackend
from .naming import generate_run_name, make_run_id
from .null import NullBackend
from .stdout import StdoutBackend

# name -> callable(**cfg) -> backend. Classes work directly; postgres is wrapped
# so psycopg (optional dep) imports only when selected, not on every metrics import.
_REGISTRY = {
    "null": NullBackend,
    "stdout": StdoutBackend,
}


def _postgres_backend(**cfg):
    try:
        from .postgres import PostgresBackend
    except ImportError as e:
        raise ValueError(
            "metrics type 'postgres' needs psycopg — install the extra: "
            "pip install 'supernova[pg]'"
        ) from e
    return PostgresBackend(**cfg)


_REGISTRY["postgres"] = _postgres_backend

# Each backend's pip extra, so a worker installs a driver only when that backend
# is selected. The dispatcher composes this onto the command's extras.
_BACKEND_EXTRAS = {"postgres": "pg"}


def required_extra(cfg: dict | None) -> str | None:
    """The pip extra a worker needs for this metrics config ('pg' for postgres),
    or None for stdout / null / no metrics block."""
    if not cfg:
        return None
    return _BACKEND_EXTRAS.get(cfg.get("type", "stdout"))

# Ambient backend. A contextvar (not a plain module global) so concurrent async
# tasks share it cleanly and a future in-process multi-run case can't clobber.
# Default Null: calling metrics.log() before any bootstrap is a silent no-op.
_current: contextvars.ContextVar[MetricsBackend] = contextvars.ContextVar(
    "supernova_metrics_current", default=NullBackend()
)


def set_current(backend: MetricsBackend) -> None:
    """Register the ambient backend. Called once by the command bootstrap."""
    _current.set(backend)


def get_current() -> MetricsBackend:
    return _current.get()


def build_metrics(cfg: dict | None) -> MetricsBackend:
    """Construct a backend from a ``metrics:`` config block.

    Omitted block -> StdoutBackend (local-first: numbers with zero setup).
    """
    if not cfg:
        return StdoutBackend()
    cfg = dict(cfg)
    kind = cfg.pop("type", "stdout")
    factory = _REGISTRY.get(kind)
    if factory is None:
        raise ValueError(
            f"Unknown metrics type: {kind!r}. Available: {sorted(_REGISTRY)}"
        )
    return factory(**cfg)


# Module-level convenience so `metrics.log(...)` works from anywhere; each
# delegates to the ambient backend so call sites never hold a reference to it.
def log(name: str, value: float, **tags) -> None:
    get_current().log(name, value, **tags)


def observe(name: str, value: float, **tags) -> None:
    get_current().observe(name, value, **tags)


def event(message: str, **tags) -> None:
    get_current().event(message, **tags)


def summary(values: dict) -> None:
    get_current().summary(values)


def timed(name: str, **tags):
    return get_current().timed(name, **tags)


__all__ = [
    "MetricsBackend",
    "NullBackend",
    "StdoutBackend",
    "build_metrics",
    "set_current",
    "get_current",
    "generate_run_name",
    "make_run_id",
    "required_extra",
    "log",
    "observe",
    "event",
    "summary",
    "timed",
]