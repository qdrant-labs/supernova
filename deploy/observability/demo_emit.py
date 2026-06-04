"""Synthetic emit -> watch samples stream into the local Grafana, no Qdrant needed.

    docker compose -f deploy/observability/docker-compose.yml up -d
    pip install 'psycopg[binary]'
    SN_METRICS_DB_URL='postgresql://postgres:supernova@localhost:5432/supernova?sslmode=disable' \
        python deploy/observability/demo_emit.py

Then open http://localhost:3000 and chart the `samples` table (see README).
This drives the PostgresBackend directly — it's sync, which is exactly why it
works fine from this plain script and from storm's async loop alike.
"""

import logging
import os
import random
import time

from supernova import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

DSN = os.environ.get(
    "SN_METRICS_DB_URL",
    "postgresql://postgres:supernova@localhost:5432/supernova?sslmode=disable",
)
DURATION_S = float(os.environ.get("DEMO_DURATION_S", "60"))


def main():
    backend = metrics.build_metrics({"type": "postgres", "dsn": DSN, "flush_interval_s": 1.0})
    metrics.set_current(backend)
    backend.init()

    run_id = metrics.make_run_id("demo")
    backend.start(run_id, {"command": "demo", "node_id": "local"})
    backend.event("synthetic workload started")
    logging.getLogger("demo").info("emitting to run %s for %.0fs", run_id, DURATION_S)

    n = 0
    status = "ok"
    end = time.time() + DURATION_S
    try:
        while time.time() < end:
            # fake query latency (ms): lognormal body, occasional tail spike.
            base = random.lognormvariate(1.4, 0.45)
            lat = base if random.random() > 0.02 else base * random.uniform(5, 15)
            metrics.observe("latency_ms", lat, ok=random.random() > 0.005)
            n += 1
            time.sleep(0.002)  # ~500 samples/s
    except Exception:
        status = "error"
        raise
    finally:
        backend.summary({"requests": n, "note": "synthetic"})
        backend.finish(status)
        logging.getLogger("demo").info("done: emitted %d samples to run %s", n, run_id)


if __name__ == "__main__":
    main()