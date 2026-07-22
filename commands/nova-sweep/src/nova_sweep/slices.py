"""Work-slice model: one collection per expanded
`data_layouts` entry, carrying every expanded `index_variants` x `searches`
combination under it. Building the slice list is a pure function of the
parsed config — no I/O, no bin-packing.
"""

from __future__ import annotations

from dataclasses import dataclass

from nova_sweep.config import SweepConfig
from nova_sweep.grid import expand_grid, order_by_rebuild_cost


@dataclass(frozen=True)
class Slice:
    data_layout: dict[str, object]
    data_layout_name: str
    collection_name: str
    # Already ordered by `order_by_rebuild_cost` — walk as-is.
    index_variants: list[dict]
    searches: list[dict]


def build_slices(cfg: SweepConfig) -> list[Slice]:
    """Every `data_layouts` entry gets its own `Slice`, but all slices share
    the same `index_variants`/`searches` lists (the full grid applies
    uniformly across every layout) — building these once and referencing them
    from every slice avoids re-expanding identical grids per layout."""
    data_layouts = expand_grid(cfg.data_layouts)
    index_variants = order_by_rebuild_cost(expand_grid(cfg.index_variants))
    searches = expand_grid(cfg.searches)

    return [
        Slice(
            data_layout=layout,
            data_layout_name=layout["_name"],
            collection_name=(
                cfg.collection_name
                if layout["_name"] == "default"
                else f"{cfg.collection_name}_{layout['_name']}"
            ),
            index_variants=index_variants,
            searches=searches,
        )
        for layout in data_layouts
    ]
