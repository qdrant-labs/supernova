#!/usr/bin/env python3
"""`nova storm` — load-test a vector store from this machine.

This is the single-machine smoke test. For trustworthy p99 numbers run
`nova storm-dist` in-region: a generator on your laptop measures
laptop->cluster latency across the public internet, not the cluster's own.
"""

import asyncio
import logging
import os

import click
import yaml

from supernova.metrics import build_metrics, generate_run_name, make_run_id, set_current
from supernova.storm.base import BaseLoadTester, LoadProfile
from supernova.storm.qdrant import QdrantLoadTester
from supernova.storm.runner import run_storm

logger = logging.getLogger(__name__)

# Vendor-agnostic dispatch: the load tester is chosen by the config's
# `target.type`, mirroring DATASOURCE_REGISTRY / VECTORSTORE_REGISTRY in
# run_loader. Add a backend (elastic, pinecone, ...) by implementing
# BaseLoadTester and registering it here.
LOAD_TESTER_REGISTRY: dict[str, type[BaseLoadTester]] = {
    "qdrant": QdrantLoadTester,
}


def build_tester(target: dict, query: dict) -> BaseLoadTester:
    """Construct the load tester named by ``target['type']``."""
    target = dict(target)
    kind = target.pop("type", "qdrant")
    cls = LOAD_TESTER_REGISTRY.get(kind)
    if cls is None:
        raise click.UsageError(
            f"Unknown target type: {kind!r}. Available: {list(LOAD_TESTER_REGISTRY)}"
        )
    # Remaining target fields (url, api_key, collection_name, ...) + query knobs
    # are passed through to the backend.
    return cls(
        vector_name=query.get("vector_name"),
        top_k=query.get("top_k", 10),
        **target,
    )


def _load_query_vectors(source: dict, limit: int) -> list[list[float]]:
    """Pull query vectors from a parquet (local path or ``s3://``) via DuckDB.

    TODO: unify with the artifacts `nova generate-queries` produces, and support
    hf:// sources. For ``s3://`` you may need AWS creds + region in the env.
    """
    import duckdb

    uri = source["uri"]
    column = source["column"]
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(
        f"SELECT {column} FROM read_parquet('{uri}') LIMIT {int(limit)}"
    ).fetchall()
    return [list(r[0]) for r in rows]


@click.command(name="storm", help="Load-test a vector store (single machine).")
@click.argument("config")
@click.option("--duration", type=float, default=None, help="Override load.duration_s.")
@click.option("--concurrency", type=int, default=None, help="Override load.concurrency.")
def storm(config, duration, concurrency):
    """Run a load test against the configured vector store."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    from supernova.cli.run_loader import resolve_config

    with open(config) as f:
        cfg = resolve_config(yaml.safe_load(f))

    query = cfg["query"]
    load = cfg.get("load", {})

    tester = build_tester(cfg["target"], query)

    src = query["source"]
    vectors = _load_query_vectors(src, src.get("limit", 5000))
    if not vectors:
        raise click.UsageError(f"No query vectors loaded from {src.get('uri')!r}")

    profile = LoadProfile(
        concurrency=concurrency or load.get("concurrency", 32),
        duration_s=duration or load.get("duration_s", 60),
        ramp_s=load.get("ramp_s", 0),
    )

    logger.info(
        "storm: %d query vectors, concurrency=%d, duration=%.0fs",
        len(vectors),
        profile.concurrency,
        profile.duration_s,
    )

    node_id = os.environ.get("SKYPILOT_JOB_RANK", "local")
    base_name = cfg.get("dispatch", {}).get("run_name") or generate_run_name()
    # Unique per execution so reruns don't collide on the runs PK. Distributed
    # workers share one id via NOVA_RUN_ID (the controller mints + forwards it).
    run_id = os.environ.get("NOVA_RUN_ID") or make_run_id(base_name)
    metrics_backend = build_metrics(cfg.get("metrics"))
    set_current(metrics_backend)
    metrics_backend.init()
    metrics_backend.start(run_id, {"command": "storm", "node_id": node_id, "config": cfg})
    logger.info("storm run: %s (node %s)", run_id, node_id)

    status = "ok"
    try:
        results = asyncio.run(run_storm(tester, vectors, profile))
        summary = results.summary()
        metrics_backend.summary(summary)
        click.echo("=" * 50)
        for k, v in summary.items():
            click.echo(f"  {k:>16}: {v}")
        click.echo("=" * 50)
    except Exception:
        status = "error"
        raise
    finally:
        metrics_backend.finish(status)


if __name__ == "__main__":
    storm()