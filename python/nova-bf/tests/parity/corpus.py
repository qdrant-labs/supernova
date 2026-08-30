"""The one synthetic dataset every parity test shares.

A single corpus carries ALL THREE modalities (dense, sparse, multivector) on
the same points, alongside a payload that exercises every filter kind nova-bf
supports. That is what makes a filter × modality cross-product affordable:
one generation, one set of parquet files, one Qdrant upsert, and any test can
then ask for "cosine multivector under a datetime range" without building its
own fixture.

Everything is derived from a fixed seed, so the corpus is identical on the
laptop and on the GPU box — a device comparison is only meaningful if both
devices saw the same bytes.

Shape choices that are load-bearing, not arbitrary:

  * corpus rows are split across files of UNEVEN, non-round sizes, so file
    boundaries never coincide with batch boundaries and the global row number
    of a hit depends on the file-then-row order being right;
  * a few sparse rows repeat a token id, which is ONE dimension summed, not
    two (`naive.sparse_from_pairs`);
  * a few sparse rows carry a token no query ever uses, which must be dropped
    silently by the query-vocab truncation rather than error or shift a score;
  * a few points have a null or empty multivector, making them non-candidates
    for a multivector search but perfectly good candidates for dense/sparse —
    the same point behaving differently per modality is exactly the kind of
    bookkeeping a shared corpus pass can get wrong;
  * `tier` is null on some rows, so "a null payload value never matches" is
    tested on a field a filter actually reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import naive

SEED = 20260829

DIM = 16          # dense width
MV_DIM = 8        # multivector token width
VOCAB = 40        # sparse token ids a query may use
FILE_SIZES = (97, 61, 79, 63)   # 300 corpus rows, none a multiple of any batch size
N_QUERIES = 8

LANGUAGES = ("eng", "fra", "deu", "spa")
CATEGORIES = ("news", "blog", "paper", "forum")
TIERS = ("gold", "silver", None)
# ASCII only, and every word distinct enough that a phrase match is decidable
# by eye — see `naive.tokens` for why non-ASCII is deliberately out of scope.
TITLE_WORDS = (
    "vector", "search", "engine", "recall", "latency", "index", "quantized",
    "graph", "payload", "filter", "sparse", "dense", "token", "cluster",
)


@dataclass
class Dataset:
    """Paths + the in-memory truth the oracles read.

    `docs` is in GLOBAL ROW ORDER (file order, then row within file), which is
    the order nova-bf assigns ordinals in and the order every id in this
    harness refers to.
    """

    tmp: str
    corpus_dir: str
    queries_path: str
    docs: list[naive.Doc]
    queries: list[dict]
    # Declared datetime columns, corpus-side and queries-side. They are two
    # separate declarations in a real config (`corpus.date_fields` /
    # `queries.date_fields`) and both normalize to epoch µs at load, so the
    # oracles need both to compare the same instants nova-bf does.
    date_fields: dict[str, str]
    query_date_fields: dict[str, str]

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    def doc(self, row: int) -> naive.Doc:
        return self.docs[row]


def _sparse_row(rng, *, duplicate: bool, oov: bool) -> tuple[list[int], list[float]]:
    nnz = int(rng.integers(5, 11))
    idx = rng.choice(VOCAB, size=nnz, replace=False).tolist()
    val = rng.standard_normal(nnz).astype(np.float32).tolist()
    if duplicate:
        # Same token id twice: one dimension whose value is the sum.
        idx.append(idx[0])
        val.append(float(rng.standard_normal()))
    if oov:
        idx.append(VOCAB + 7)  # outside every query's support
        val.append(3.5)
    return [int(i) for i in idx], [float(v) for v in val]


def _mv_array(docs: list[np.ndarray | None]) -> pa.Array:
    """`list<list<float32>>`, with a genuine Arrow NULL for a point that has no
    multivector at all (as distinct from one with zero tokens — both are
    non-candidates, and both occur in this corpus)."""
    offsets, flat, mask = [0], [], []
    for d in docs:
        mask.append(d is None)
        if d is None:
            offsets.append(offsets[-1])
            continue
        offsets.append(offsets[-1] + len(d))
        for tok in d:
            flat.extend(float(x) for x in tok)
    values = pa.array(flat, type=pa.float32())
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, len(flat) + 1, MV_DIM, dtype=np.int32)), values
    )
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        inner,
        mask=pa.array(mask, type=pa.bool_()),
    )


def _sparse_array(rows) -> pa.Array:
    return pa.array(
        [{"indices": idx, "values": val} for idx, val in rows],
        type=pa.struct([
            pa.field("indices", pa.list_(pa.uint32())),
            pa.field("values", pa.list_(pa.float32())),
        ]),
    )


def build(tmp_path, n_queries: int = N_QUERIES) -> Dataset:
    """Generate the corpus + queries under `tmp_path` and return the truth.

    `n_queries` varies ONLY the query side: the corpus is drawn first, from the
    same seeded generator, so every dataset built here has bit-identical
    documents regardless of how many queries accompany them. That is what lets
    the mask-height suite ask for a taller query file and still reuse the one
    Qdrant collection everything else was loaded into.

    Why it needs one: a per-query filter mask is bit-PACKED along the query
    axis (`compute._pack_query_axis`), so with 8 queries every mask — whatever
    its height — is a single byte, and reading one at the wrong height is
    invisible. Heights only become distinguishable once they span different
    numbers of bytes.
    """
    rng = np.random.default_rng(SEED)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    docs: list[naive.Doc] = []
    row = 0
    for fi, n in enumerate(FILE_SIZES):
        dense, sparse_rows, mvs = [], [], []
        payloads = []
        for r in range(n):
            g = row + r
            d = rng.standard_normal(DIM).astype(np.float32)
            dense.append(d)

            idx, val = _sparse_row(rng, duplicate=(g % 37 == 0), oov=(g % 53 == 0))
            sparse_rows.append((idx, val))

            # Every 29th point has no multivector, every 41st an empty one:
            # both are non-candidates for MaxSim while remaining ordinary
            # dense/sparse points.
            if g % 29 == 0:
                mv = None
            elif g % 41 == 0:
                mv = np.zeros((0, MV_DIM), dtype=np.float32)
            else:
                mv = rng.standard_normal(
                    (int(rng.integers(2, 7)), MV_DIM)
                ).astype(np.float32)
            mvs.append(mv)

            n_words = int(rng.integers(5, 10))
            title = " ".join(
                TITLE_WORDS[int(i)] for i in rng.choice(len(TITLE_WORDS), size=n_words)
            )
            payloads.append({
                "id": str(g),
                "language": LANGUAGES[g % len(LANGUAGES)],
                "category": CATEGORIES[(g // 3) % len(CATEGORIES)],
                "tier": TIERS[g % len(TIERS)],
                "views": int(rng.integers(0, 10_000)),
                "rating": float(rng.uniform(0.0, 5.0)),
                "title": title,
                # Spread over ~3 years at day granularity so a datetime range
                # lands on a real boundary rather than slicing an empty gap.
                "published_at": _rfc3339(2013, int(rng.integers(0, 1095))),
            })

            docs.append(naive.Doc(
                row=g,
                payload={k: v for k, v in payloads[-1].items() if k != "id"},
                dense=[float(x) for x in d],
                sparse=naive.sparse_from_pairs(idx, val),
                multivector=None if mv is None else [[float(x) for x in t] for t in mv],
            ))

        table = pa.table({
            "dense_embedding": pa.array(
                [v.tolist() for v in dense], type=pa.list_(pa.float32())
            ),
            "sparse_embedding": _sparse_array(sparse_rows),
            "multivector_embedding": _mv_array(mvs),
            "id": pa.array([p["id"] for p in payloads]),
            "language": pa.array([p["language"] for p in payloads]),
            "category": pa.array([p["category"] for p in payloads]),
            "tier": pa.array([p["tier"] for p in payloads]),
            "views": pa.array([p["views"] for p in payloads], type=pa.int64()),
            "rating": pa.array([p["rating"] for p in payloads], type=pa.float64()),
            "title": pa.array([p["title"] for p in payloads]),
            "published_at": pa.array([p["published_at"] for p in payloads]),
        })
        pq.write_table(table, str(corpus_dir / f"part-{fi:03d}.parquet"))
        row += n

    queries = _build_queries(rng, tmp_path, n_queries)
    return Dataset(
        tmp=str(tmp_path),
        corpus_dir=str(corpus_dir),
        queries_path=str(tmp_path / "queries.parquet"),
        docs=docs,
        queries=queries,
        date_fields={"published_at": "rfc3339"},
        query_date_fields={"q_after": "rfc3339"},
    )


def _rfc3339(base_year: int, day_offset: int) -> str:
    from datetime import datetime, timedelta, timezone

    dt = datetime(base_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_offset)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_queries(rng, tmp_path, n_queries: int = N_QUERIES) -> list[dict]:
    """Queries carry all three modalities plus the columns the per-query
    filter conditions read (`q_*`). The per-query columns are chosen so that
    no two queries get the same corpus subset — a per-query filter that was
    silently collapsed to one shared mask would still pass a test where every
    query wanted the same thing."""
    rows = []
    for qi in range(n_queries):
        dense = rng.standard_normal(DIM).astype(np.float32)
        idx, val = _sparse_row(rng, duplicate=False, oov=False)
        mv = rng.standard_normal((int(rng.integers(2, 6)), MV_DIM)).astype(np.float32)
        rows.append({
            "qid": str(qi),
            "dense": dense,
            "sparse": (idx, val),
            "multivector": mv,
            # per-query filter inputs
            "q_language": LANGUAGES[qi % len(LANGUAGES)],
            "q_languages": [LANGUAGES[qi % 4], LANGUAGES[(qi + 1) % 4]],
            "q_min_views": int(qi * 900),
            "q_max_views": int(3000 + qi * 800),
            "q_phrase": TITLE_WORDS[qi % len(TITLE_WORDS)],
            "q_after": _rfc3339(2013, 120 + qi * 90),  # never day 0: a per-query
            # bound that let every row through would make that query vacuous
            # query-row selectors (see SearchSpec.rows). Two independent
            # groupings, because the per-FILTER mask height is the union of
            # the `rows` of the specs sharing that filter — with only one
            # grouping every filter's union is either half the file or all of
            # it, and the heights that must stay distinct (file / vector_type
            # union / filter union) collapse onto each other.
            "query_set": "even" if qi % 2 == 0 else "odd",
            "query_third": f"g{qi % 3}",
        })

    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            [r["dense"].tolist() for r in rows], type=pa.list_(pa.float32())
        ),
        "sparse_embedding": _sparse_array([r["sparse"] for r in rows]),
        "multivector_embedding": _mv_array([r["multivector"] for r in rows]),
        "qid": pa.array([r["qid"] for r in rows]),
        "q_language": pa.array([r["q_language"] for r in rows]),
        "q_languages": pa.array([r["q_languages"] for r in rows]),
        "q_min_views": pa.array([r["q_min_views"] for r in rows], type=pa.int64()),
        "q_max_views": pa.array([r["q_max_views"] for r in rows], type=pa.int64()),
        "q_phrase": pa.array([r["q_phrase"] for r in rows]),
        "q_after": pa.array([r["q_after"] for r in rows]),
        "query_set": pa.array([r["query_set"] for r in rows]),
        "query_third": pa.array([r["query_third"] for r in rows]),
    }), str(tmp_path / "queries.parquet"))

    # The oracles want the query vectors in plain-Python form, and the
    # per-query filter values keyed by the column names the conditions name.
    out = []
    for r in rows:
        idx, val = r["sparse"]
        out.append({
            "qid": r["qid"],
            "dense": [float(x) for x in r["dense"]],
            "sparse": naive.sparse_from_pairs(idx, val),
            "multivector": [[float(x) for x in t] for t in r["multivector"]],
            "payload": {
                k: r[k] for k in (
                    "q_language", "q_languages", "q_min_views", "q_max_views",
                    "q_phrase", "q_after", "query_set", "query_third",
                )
            },
        })
    return out


def query_vector(query: dict, vector_type: str):
    """The query side of `vector_type`, in the form both oracles want."""
    return query[{"dense": "dense", "sparse": "sparse",
                  "multivector": "multivector"}[vector_type]]


def l2_normalized_sparse(vec: dict[int, float]) -> dict[int, float]:
    """A sparse vector scaled to unit norm, so a plain dot product over it IS
    cosine similarity. Qdrant has no cosine distance for sparse vectors, so
    this is how the harness gets a live-engine check on sparse cosine at all
    (see `qdrant_ref`) rather than leaving that metric naive-only."""
    n = math.sqrt(sum(w * w for w in vec.values()))
    return dict(vec) if n == 0.0 else {t: w / n for t, w in vec.items()}
