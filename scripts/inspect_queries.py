import bisect
import hashlib
import os
import random
import uuid

import polars as pl
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from qdrant_client import QdrantClient
from qdrant_client import models

BUCKET = "qdrant--vectorforge"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = "mteb_tweets_all-MiniLM-L6-v2"
DENSE_VECTOR = "dense"
DENSE_COLUMN = "dense_embedding"
K = 1000
ENRICH_COLUMNS = ["text"]

queries = pl.read_parquet("/Users/nathanleroy/Downloads/queries_1000.parquet")
bf_ground_truth = pl.read_parquet(
    "/Users/nathanleroy/Downloads/brute_force_queries_1000_k1000.parquet"
)

queries = queries.with_columns(
    (pl.col("__source_file__") + ":" + pl.col("__source_row__").cast(pl.String))
    .map_elements(
        lambda s: str(uuid.UUID(hashlib.md5(s.encode()).hexdigest())),
        return_dtype=pl.String,
    )
    .alias("point_id")
)

bf_ground_truth = bf_ground_truth.join(
    queries.select(pl.col("point_id").alias("query_id")),
    on="query_id",
    how="inner",
)

fs = pafs.S3FileSystem()
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def enrich_one(row: dict) -> dict:
    key, lo = row["__source_file__"], row["__source_row__"]
    with fs.open_input_file(f"{BUCKET}/{key}") as f:
        pf = pq.ParquetFile(f)
        rg_starts, cum = [], 0
        for rg_i in range(pf.metadata.num_row_groups):
            rg_starts.append(cum)
            cum += pf.metadata.row_group(rg_i).num_rows
        rg_idx = bisect.bisect_right(rg_starts, lo) - 1
        table = pf.read_row_group(rg_idx, columns=ENRICH_COLUMNS)
        return {
            col: table.column(col)[lo - rg_starts[rg_idx]].as_py()
            for col in ENRICH_COLUMNS
        }


def query_qdrant(row: dict, ef_search: int = 128) -> list[str]:
    result = client.query_points(
        collection_name=COLLECTION,
        query=row[DENSE_COLUMN],
        limit=K,
        with_payload=True,
        with_vectors=False,
        using=DENSE_VECTOR,
        search_params=models.SearchParams(hnsw_ef=ef_search),
    )
    return [str(p.id) for p in result.points]


def recall_one(
    row: dict, qdrant_hit_ids: list[str], negative_control: bool = False
) -> float:
    qid = row["point_id"]
    if negative_control:
        # select random bf_row as negative control to sanity check recall calculation (should be near 0)
        bf_row = bf_ground_truth.sample(1)
    else:
        bf_row = bf_ground_truth.filter(pl.col("query_id") == qid)
    if bf_row.is_empty():
        return None
    bf_set = set(bf_row["hit_ids"][0][:K])
    return len(bf_set & set(qdrant_hit_ids)) / K


row = queries.row(random.randrange(len(queries)), named=True)

text = enrich_one(row)
hits = query_qdrant(row, ef_search=256)
recall = recall_one(row, hits)

print(f"point_id:  {row['point_id']}")
print(f"source:    {row['__source_file__']}  row={row['__source_row__']}")
print(f"text:      {text['text'][:300]}")
print(f"recall@{K}: {recall:.4f}")

result = client.query_points(
    collection_name=COLLECTION,
    query=row[DENSE_COLUMN],
    limit=K,
    with_payload=True,
    with_vectors=False,
    using=DENSE_VECTOR,
)
