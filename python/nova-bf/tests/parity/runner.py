"""Running nova-bf itself, on a chosen device, and reading its answer back.

Configs are built as plain dicts and validated through the SAME prepass
`load_config` uses (`_normalize_static_date_bounds` then `model_validate`),
not by constructing `BruteForceConfig` directly. That matters for one reason:
a static `range` bound on a declared date field is written as an RFC-3339
string in real YAML and converted to epoch microseconds by that prepass. A
harness that hand-constructed the model would have to pre-convert its own
bounds to µs, which would both be unlike every real config and quietly skip
the conversion step from coverage.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pyarrow.parquet as pq

from nova_bf.compute import run_compute
from nova_bf.config import BruteForceConfig, _normalize_static_date_bounds


def spec(
    name: str,
    *,
    vector_type: str = "dense",
    metric: str = "cosine",
    k: int = 25,
    filter: dict | None = None,
    rows: dict | None = None,
) -> dict:
    """One search, in the dict form a YAML config would carry."""
    out = {"name": name, "vector_type": vector_type, "metric": metric, "k": k}
    if filter is not None:
        out["filter"] = filter
    if rows is not None:
        out["rows"] = rows
    return out


def build_config(ds, specs: list[dict], *, out_tag: str, params: dict | None = None):
    data = {
        "corpus": {
            "path": ds.corpus_dir,
            "id_column": "id",
            "date_fields": ds.date_fields,
        },
        "queries": {
            "path": ds.queries_path,
            "id_column": "qid",
            # Declared so a `range_from_query` can draw a datetime bound from
            # `q_after`; the config rejects mixing a date corpus field with a
            # non-date query column, so this has to be declared to be usable.
            "date_fields": ds.query_date_fields,
        },
        "output": {"path": f"{ds.tmp}/out_{out_tag}"},
        "params": {"io_workers": 2, **(params or {})},
        "searches": specs,
    }
    return BruteForceConfig.model_validate(_normalize_static_date_bounds(data))


@contextmanager
def env(**overrides: str | None):
    """Set (or, with `None`, unset) environment variables for the block.

    The kernel switches nova-bf exposes to operators are environment variables
    — `NOVA_BF_NO_TOPK_KERNEL`, `NOVA_BF_NO_FOLD_KERNEL` — so an A/B between
    the fused and portable paths is an env A/B, and it has to restore whatever
    was there before or one test's setting leaks into the next.
    """
    previous = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def pinned_device(device: str | None):
    """Pin `run_compute` to `device` for the duration of the block.

    nova-bf auto-selects CUDA when it sees one, which is right for production
    and useless for a parity harness: on a GPU box there would otherwise be no
    way to ask for the CPU answer and compare the two. `None` restores
    auto-selection.
    """
    previous = os.environ.get("NOVA_BF_DEVICE")
    if device is None:
        os.environ.pop("NOVA_BF_DEVICE", None)
    else:
        os.environ["NOVA_BF_DEVICE"] = device
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NOVA_BF_DEVICE", None)
        else:
            os.environ["NOVA_BF_DEVICE"] = previous


def read_results(paths: dict[str, str]) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """`{search_name: {query_index: [(corpus_row, score), …]}}`, hits left in
    the rank order nova-bf wrote them (they are already sorted best-first —
    keeping the order is what lets a caller check ranking, not just set
    membership)."""
    out = {}
    for name, path in paths.items():
        t = pq.read_table(path).to_pydict()
        out[name] = {
            int(qid): [(int(i), float(s)) for i, s in zip(ids, scores)]
            for qid, ids, scores in zip(t["query_id"], t["hit_ids"], t["hit_scores"])
        }
    return out


def run(ds, specs: list[dict], *, out_tag: str, device: str | None = None,
        params: dict | None = None):
    """Run one nova-bf config on `device` and return its hits per search."""
    cfg = build_config(ds, specs, out_tag=f"{out_tag}_{device or 'auto'}", params=params)
    with pinned_device(device):
        paths = run_compute(cfg)
    return read_results(paths)
