"""Load testing for vector stores — the `nova storm` workload.

A *storm* throws sustained query load at a target cluster and records latency
(and optionally recall). Unlike embed/load, work is **replicated, not
partitioned**: every worker runs the same load profile, so total offered load
is roughly ``num_workers × per-worker concurrency``.
"""