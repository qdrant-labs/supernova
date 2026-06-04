"""Friendly run-name generation (``frosty-mango`` style).

CRITICAL for distributed runs: generate the name ONCE on the controller and
forward it to every worker (in the config or an env var). If each worker
generated its own, N workers would log N unrelated runs instead of one run with
N nodes. ``node_id`` (SKYPILOT_JOB_RANK) is what distinguishes workers *within*
a run.
"""

import random
from datetime import datetime

_ADJECTIVES = (
    "frosty", "amber", "brisk", "cobalt", "dapper", "eager", "fuzzy", "gilded",
    "hazy", "ivory", "jolly", "keen", "lucid", "mellow", "nimble", "opal",
    "plucky", "quiet", "rustic", "snowy", "teal", "umber", "vivid", "wispy",
)
_NOUNS = (
    "mango", "otter", "comet", "delta", "ember", "falcon", "grove", "harbor",
    "ibis", "juniper", "kestrel", "lagoon", "marten", "nebula", "orchard",
    "pine", "quartz", "raven", "summit", "tundra", "vortex", "willow", "zephyr",
)


def generate_run_name() -> str:
    """A readable base name when a config doesn't set dispatch.run_name."""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


def make_run_id(name: str) -> str:
    """Unique id for ONE execution: base name + timestamp. The runs table keys on
    this, so rerunning with the same dispatch.run_name no longer collides.
    Distributed workers must share one id — the controller mints it and forwards
    NOVA_RUN_ID rather than each worker calling this."""
    return f"{name}-{datetime.now():%Y%m%d-%H%M%S}"