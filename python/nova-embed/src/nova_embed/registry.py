"""Self-registering component registries (SkyPilot-style).

Each pluggable component — sources, embedders, chunkers, storage — declares
its config name with a decorator next to the class:

    from nova_embed.registry import EMBEDDERS

    @EMBEDDERS.register("openai")
    class OpenAIEmbedder(Embedder):
        output_kind = OutputKind.DENSE
        ...

So adding a backend is a one-line decorator on the class, not an edit to a
central dict in the CLI. A class may register several names (aliases).

Embedders are keyed on (output_kind, type): the same backend name can appear
under several kinds (`sentence_transformer` is both a dense and a sparse
backend), so a config entry selects with both its `kind` and `type` fields.
The kind half of the key comes from the class's own `output_kind` declaration —
registration never restates it.

Registration is a decorator side-effect, so a class is only in the registry once
its module is imported. The subpackage ``__init__`` files import every concrete
module to populate the registries; importing e.g. ``nova_embed.embedders.dense``
registers all dense embedders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps config `type` strings to classes, built via [`build`][Registry.build]."""

    def __init__(self, kind: str, *, key: str = "type", default: str | None = None):
        self._kind = kind  # human label for error messages, e.g. "dense embedder"
        self._key = key  # the config field that selects the class (e.g. "strategy")
        self._default = default  # used when the config omits the selector key
        self._classes: dict[str, type[T]] = {}

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        """Decorator: register a class under one or more config names."""
        if not names:
            raise ValueError("register() needs at least one name")

        def decorate(cls: type[T]) -> type[T]:
            for name in names:
                existing = self._classes.get(name)
                if existing is not None and existing is not cls:
                    raise ValueError(
                        f"{self._kind} name {name!r} is already registered to "
                        f"{existing.__name__}"
                    )
                self._classes[name] = cls
            return cls

        return decorate

    def get(self, name: str) -> type[T]:
        cls = self._classes.get(name)
        if cls is None:
            raise ValueError(
                f"unknown {self._kind} {name!r}. available: {self.names()}"
            )
        return cls

    def build(self, cfg: dict | None) -> T:
        """Construct from a config mapping.

        Pops the selector key (`type` by default), then constructs. A class may
        own its config parsing via a `from_config(cfg)` classmethod (e.g. storage
        backends that alias keys or ignore staging-only fields); otherwise the
        remaining mapping is passed as keyword args.
        """
        cfg = dict(cfg or {})
        name = cfg.pop(self._key, self._default)
        if name is None:
            raise ValueError(
                f"{self._kind} config is missing required key {self._key!r}"
            )
        target = self.get(name)
        from_config = getattr(target, "from_config", None)
        if from_config is not None:
            return from_config(cfg)
        return target(**cfg)

    def names(self) -> list[str]:
        return sorted(self._classes)


class EmbedderRegistry:
    """Embedder classes, keyed on (output_kind, config `type` name).

    ``register()`` takes only the name(s) — the kind half of the key is read
    from the class's ``output_kind`` declaration at decoration time.
    """

    def __init__(self):
        self._classes: dict[tuple[str, str], type] = {}

    def register(self, *names: str):
        if not names:
            raise ValueError("register() needs at least one name")

        def decorate(cls):
            kind = getattr(cls, "output_kind", None)
            if kind is None:
                raise ValueError(
                    f"{cls.__name__} must declare `output_kind` before registration"
                )
            for name in names:
                key = (str(kind.value), name)
                existing = self._classes.get(key)
                if existing is not None and existing is not cls:
                    raise ValueError(
                        f"embedder ({kind.value}, {name!r}) is already registered to "
                        f"{existing.__name__}"
                    )
                self._classes[key] = cls
            return cls

        return decorate

    def get(self, kind: str, name: str) -> type:
        cls = self._classes.get((str(kind), name))
        if cls is None:
            available = self.names(kind)
            hint = f"available {kind} backends: {available}" if available else (
                f"no backends registered for kind {kind!r}"
            )
            other_kinds = sorted(k for k, n in self._classes if n == name and k != str(kind))
            if other_kinds:
                hint += f". (type {name!r} exists for kind(s): {other_kinds})"
            raise ValueError(f"unknown {kind} embedder type {name!r}. {hint}")
        return cls

    def names(self, kind: str | None = None) -> list[str]:
        if kind is None:
            return sorted({n for _, n in self._classes})
        return sorted(n for k, n in self._classes if k == str(kind))


if TYPE_CHECKING:
    from nova_embed.chunkers.base import Chunker
    from nova_embed.sources.base import DatasetSource
    from nova_embed.storage.base import StorageBackend

# The registries. One per pluggable component kind.
SOURCES: "Registry[DatasetSource]" = Registry("source")
EMBEDDERS = EmbedderRegistry()
# Chunkers select on `strategy:` and default to the no-op passthrough.
CHUNKERS: "Registry[Chunker]" = Registry("chunker", key="strategy", default="passthrough")
# Storage defaults to s3 when `type:` is omitted (matches prior behaviour).
STORAGE: "Registry[StorageBackend]" = Registry("storage backend", default="s3")
