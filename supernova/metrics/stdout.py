"""The local-dev backend: route measurements to the logger.

The default when a config has no ``metrics:`` block — local-first, you see your
numbers with zero setup. Per-sample ``observe`` goes to DEBUG so a high-rate
storm doesn't flood the terminal; everything else is INFO.
"""

import logging

from .base import MetricsBackend

logger = logging.getLogger("supernova.metrics")


def _fmt_tags(tags: dict) -> str:
    return " " + " ".join(f"{k}={v}" for k, v in tags.items()) if tags else ""


class StdoutBackend(MetricsBackend):
    def __init__(self):
        self._run_id: str | None = None
        self._node_id: str | None = None

    def _prefix(self) -> str:
        node = f"/{self._node_id}" if self._node_id is not None else ""
        return f"[{self._run_id}{node}]"

    def start(self, run_id: str, context: dict) -> None:
        self._run_id = run_id
        self._node_id = context.get("node_id")
        logger.info("%s run started (command=%s)", self._prefix(), context.get("command"))

    def log(self, name: str, value: float, **tags) -> None:
        logger.info("%s %s=%s%s", self._prefix(), name, value, _fmt_tags(tags))

    def observe(self, name: str, value: float, **tags) -> None:
        logger.debug("%s %s=%s%s", self._prefix(), name, value, _fmt_tags(tags))

    def event(self, message: str, **tags) -> None:
        logger.info("%s · %s%s", self._prefix(), message, _fmt_tags(tags))

    def summary(self, values: dict) -> None:
        logger.info("%s summary %s", self._prefix(), values)

    def finish(self, status: str = "ok") -> None:
        logger.info("%s run finished (%s)", self._prefix(), status)