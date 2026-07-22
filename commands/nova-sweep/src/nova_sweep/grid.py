"""Cartesian-grid axis expansion (`expand_grid`) and rebuild-cost ordering
(`order_by_rebuild_cost`) — both pure functions, no I/O.
"""

from __future__ import annotations

import itertools

from typing import Any


# full HNSW build is the most expensive axis of nova-load's `reindex`
_EXPENSIVE_KEYS = ("hnsw",)


def _set_path(d: dict, dotted_key: str, value: Any) -> None:
    """Assign `value` at `dotted_key` (dot-separated) inside `d`, creating
    intermediate dicts as needed.

    A YAML `null` (Python `None`) prunes the whole path instead of setting
    it — deliberately `None`-only, not string-`"none"`-too: nova-load's own
    quantization vocabulary already has a real `none` *value*
    (`quantization.type: none` explicitly clears quantization on `reindex`,
    distinct from omitting the `quantization:` block entirely — see
    `crates/nova-load/src/stores/qdrant.rs`'s `parse_quantization`), so the
    string "none" must pass through unchanged like any other value here, not
    get treated as a pruning sentinel.
    """
    if value is None:
        return
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _name_segment(dotted_key: str, value: Any) -> str | None:
    if value is None:
        return None
    return f"{dotted_key.rsplit('.', 1)[-1]}{value}"


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian-product every key's value list into one nested dict per
    combination, each carrying a deterministic auto-generated `_name`.

    A `null` leaf value omits that key entirely from the resulting dict (and
    its name segment) instead of setting it to `None` — see `_set_path`. An
    empty grid produces a single `{"_name": "default"}` combination (a grid
    axis with nothing declared is just "run once with nothing overridden").
    """
    if not grid:
        return [{"_name": "default"}]

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    combinations = []
    for values in itertools.product(*value_lists):
        combo: dict[str, Any] = {}
        name_segments = []
        for key, value in zip(keys, values):
            _set_path(combo, key, value)
            segment = _name_segment(key, value)
            if segment is not None:
                name_segments.append(segment)
        combo["_name"] = "_".join(name_segments) if name_segments else "default"
        combinations.append(combo)
    return combinations


def _sort_key_repr(value: Any) -> str:
    """A sortable, hashable representation of a (possibly nested-dict) field
    value, for use inside a sort key — plain dicts aren't orderable."""
    if isinstance(value, dict):
        return repr(sorted((k, _sort_key_repr(v)) for k, v in value.items()))
    return repr(value)


def order_by_rebuild_cost(combinations: list[dict]) -> list[dict]:
    """Stable-sort expanded `index_variants` combinations so EXPENSIVE fields
    (see `_EXPENSIVE_KEYS`) change as infrequently as possible — effectively
    the outer loop — regardless of the order they were declared in the YAML.

    A pure re-sort: same combinations, same count, just reordered. A no-op
    (preserves `expand_grid`'s original cartesian order) when no EXPENSIVE
    keys are present in any combination — `sorted` is stable, and every
    combination gets an identical (empty) sort key in that case.
    """

    def cost_key(combo: dict) -> tuple:
        return tuple(_sort_key_repr(combo.get(k)) for k in _EXPENSIVE_KEYS)

    return sorted(combinations, key=cost_key)
