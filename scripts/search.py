"""
Search REPL — run with `python -i scripts/search.py` or paste into IPython.

Edit the config block below, then call:
  hybrid("my query")          # prefetch dense+sparse k=1000, fuse with RRF
  search("my query")          # dense + sparse side by side (no fusion)
  dense("my query")           # dense only
  sparse("my query")          # sparse/BM25 only
"""

import os
import time

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed.sparse import SparseTextEmbedding

# ── config ────────────────────────────────────────────────────────────────────
QDRANT_URL     = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION     = "fineweb-bge-large-bm25"
DENSE_MODEL    = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL   = "Qdrant/bm25"
DENSE_VECTOR   = "dense"
SPARSE_VECTOR  = "sparse"
TEXT_FIELD     = "text"
LIMIT          = 5
PREFETCH_LIMIT = 100   # candidates pulled per vector before RRF fusion
# ─────────────────────────────────────────────────────────────────────────────

print("Loading models...")
_dense  = SentenceTransformer(DENSE_MODEL)
_sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
print(f"Ready — collection: {COLLECTION}\n")


def _show(hits, label, encode_ms=None, query_ms=None):
    parts = []
    if encode_ms is not None:
        parts.append(f"encode {encode_ms:.0f}ms")
    if query_ms is not None:
        parts.append(f"query {query_ms:.0f}ms")
    timing = f"  ({', '.join(parts)})" if parts else ""
    print(f"\n── {label}{timing} ──")
    for i, h in enumerate(hits, 1):
        p = h.payload or {}
        title = p.get("title") or p.get("url") or "(no title)"
        url   = p.get("url", "")
        text  = (p.get(TEXT_FIELD) or "")[:300].replace("\n", " ")
        date = p.get("date", "")
        if date:
            title = f"{date}  {title}"
        print(f"  {i}. [{h.score:.4f}]  {title}")
        if url and url != title:
            print(f"       {url}")
        print(f"       {text}…")


def dense(query, limit=LIMIT):
    t0 = time.perf_counter()
    vec = _dense.encode(query, normalize_embeddings=True).tolist()
    encode_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    hits = _client.query_points(
        collection_name=COLLECTION,
        query=vec,
        using=DENSE_VECTOR,
        limit=limit,
        with_payload=True,
        timeout=10*1000, # 10s for dense queries since they can be more expensive, especially if not pre-normalized
        search_params=models.SearchParams(exact=True)
    ).points
    query_ms = (time.perf_counter() - t0) * 1000
    _show(hits, f"dense  '{query}'", encode_ms, query_ms)


def sparse(query, limit=LIMIT):
    t0 = time.perf_counter()
    emb = next(_sparse.embed([query]))
    vec = models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
    encode_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    hits = _client.query_points(
        collection_name=COLLECTION,
        query=vec,
        using=SPARSE_VECTOR,
        limit=limit,
        with_payload=True,
    ).points
    query_ms = (time.perf_counter() - t0) * 1000
    _show(hits, f"sparse '{query}'", encode_ms, query_ms)


def hybrid(query, limit=LIMIT, prefetch=PREFETCH_LIMIT):
    t0 = time.perf_counter()
    dense_vec = _dense.encode(query, normalize_embeddings=True).tolist()
    emb = next(_sparse.embed([query]))
    sparse_vec = models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
    encode_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    hits = _client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec,  using=DENSE_VECTOR,  limit=prefetch),
            models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR, limit=prefetch),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points
    query_ms = (time.perf_counter() - t0) * 1000
    _show(hits, f"hybrid/RRF  '{query}'  (prefetch={prefetch})", encode_ms, query_ms)

def search(query, limit=LIMIT):
    dense(query, limit)
    sparse(query, limit)