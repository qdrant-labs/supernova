"""Which devices this machine can answer on — the whole GPU-reuse mechanism.

`parity_devices()` returns `["cpu"]` on a laptop and `["cpu", "cuda"]` on a
box with a GPU, and every parity test is parametrized over it. Nothing in the
suite is device-specific, so running it on a GPU machine is not a port: the
same file simply produces twice the cases, each nova-bf run pinned via
`NOVA_BF_DEVICE`, and `test_parity_devices.py` additionally asserts the two
devices agree with each other on the same input.

`NOVA_BF_PARITY_DEVICES` overrides the list (comma-separated) for a run that
wants only one — e.g. `NOVA_BF_PARITY_DEVICES=cuda` on a CI box where the CPU
pass would just be slow duplication.
"""

from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=1)
def parity_devices() -> tuple[str, ...]:
    override = os.environ.get("NOVA_BF_PARITY_DEVICES", "").strip()
    if override:
        return tuple(d.strip() for d in override.split(",") if d.strip())
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except ImportError:
        pass
    return tuple(devices)


def has_cuda() -> bool:
    return "cuda" in parity_devices()
