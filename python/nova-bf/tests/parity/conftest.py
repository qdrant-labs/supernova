"""Fixtures for the parity harness.

Everything expensive is session-scoped and built once: the synthetic corpus,
the Qdrant collection, the naive oracle's cached score matrices, and — the big
one — the nova-bf runs. All 119 matrix cases go through nova-bf in ONE run per
device, which is both far faster than 119 runs and a more honest test: it is
the shared-pass code path production actually uses, where one filter is
evaluated once per file for every search that names it and all searches of a
vector_type share one batch grid.

The Qdrant-backed tests skip themselves when no server is reachable
(`QDRANT_URL`, default http://localhost:6333), so the naive half of the suite
still runs anywhere — including in CI with no Qdrant.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from . import cases as cases_mod
from . import corpus as corpus_mod
from . import naive, runner
from .devices import parity_devices

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def pytest_configure(config):
    config.addinivalue_line("markers", "qdrant: needs a reachable Qdrant server")


@pytest.fixture(scope="session")
def ds(tmp_path_factory):
    return corpus_mod.build(tmp_path_factory.mktemp("bf_parity"))


# Enough queries that a per-query filter mask, which is bit-packed along the
# query axis, spans a DIFFERENT number of bytes at each of the heights the
# mask-height suite sets up. At 8 queries every height is one byte and reading
# one at the wrong height is undetectable — see `corpus.build`.
WIDE_QUERIES = 26


@pytest.fixture(scope="session")
def ds_wide(tmp_path_factory):
    """Same documents as `ds` (identical seed, corpus drawn first), more
    queries. Reuses the same Qdrant collection, since only the query side
    differs."""
    return corpus_mod.build(tmp_path_factory.mktemp("bf_parity_wide"),
                            n_queries=WIDE_QUERIES)


@pytest.fixture(scope="session")
def oracle_wide(ds_wide):
    return naive.Oracle(ds_wide.docs, ds_wide.queries, ds_wide.date_fields,
                        ds_wide.query_date_fields)


@pytest.fixture(scope="session")
def oracle(ds):
    return naive.Oracle(ds.docs, ds.queries, ds.date_fields, ds.query_date_fields)


@pytest.fixture(scope="session", params=parity_devices())
def device(request):
    """Every parity test runs once per device this machine can answer on —
    `["cpu"]` on a laptop, `["cpu", "cuda"]` on a GPU box. This is the whole
    of what "make it reusable on GPU" needs: no test knows which device it is
    on."""
    return request.param


@pytest.fixture(scope="session")
def matrix_run(ds, device):
    """Every case in `cases.CASES`, computed by nova-bf in one run on
    `device`. `{case_name: {query_index: [(row, score), …]}}`."""
    return runner.run(
        ds,
        [c.spec() for c in cases_mod.CASES],
        out_tag="matrix",
        device=device,
        # Small enough that the 4 corpus files each become several slices, so
        # the per-file batch loop, the top-K merge across slices and the
        # filter's row compaction are all exercised rather than short-circuited
        # by a single whole-file batch.
        params={"dense_batch_size": 37, "sparse_batch_size": 29,
                "multivector_batch_size": 23},
    )


@pytest.fixture(scope="session")
def client():
    qdrant_client = pytest.importorskip("qdrant_client")
    try:
        c = qdrant_client.QdrantClient(url=QDRANT_URL, timeout=60)
        c.get_collections()
    except Exception as exc:  # pragma: no cover - environment gate
        pytest.skip(f"no reachable Qdrant at {QDRANT_URL}: {exc}")
    return c


@pytest.fixture(scope="session")
def collection(client, ds):
    from . import qdrant_ref

    name = qdrant_ref.create_collection(client, ds)
    yield name
    client.delete_collection(name)
