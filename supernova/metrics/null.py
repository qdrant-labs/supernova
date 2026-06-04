"""The do-nothing backend.

The default ambient backend — so ``metrics.log(...)`` is a silent no-op when
supernova is imported as a library with no command bootstrap — and the right
choice in tests. It is literally ``MetricsBackend`` with no overrides.
"""

from .base import MetricsBackend


class NullBackend(MetricsBackend):
    pass