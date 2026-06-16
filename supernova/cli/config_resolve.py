"""``${VAR}`` resolution for orchestrator configs.

The ``nova dist *`` orchestrators (and ``nova experiment``) read a config
in-process to build their dispatch plan — resolving ``${VAR}`` references
against the environment so the plan shows real values. Workers get the *raw*
(unresolved) config plus forwarded env vars and resolve it themselves at run
time, so no secret is ever written to a worker's disk.

(The local Rust/Python tools do their own resolution; this is only for the
controller-side read.)
"""

import os
import re


def resolve_env_vars(value: str) -> str:
    """Replace ``${VAR_NAME}`` references in a string with env var values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return val

    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", _replace, value)
    return value


def resolve_config(obj):
    """Recursively resolve ``${VAR}`` references throughout a parsed config."""
    if isinstance(obj, str):
        return resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_config(v) for v in obj]
    return obj
