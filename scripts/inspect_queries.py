import bisect
import hashlib
import os
import random
import uuid

import polars as pl
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from qdrant_client import QdrantClient

BUCKET = "qdrant--vectorforge"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = "fineweb-bge-large-bm25"
DENSE_VECTOR = "dense"
DENSE_COLUMN = "dense_embedding"
K = 1000
ENRICH_COLUMNS = ["text"]

queries = pl.read_parquet("/Users/nathanleroy/Downloads/queries_1000.parquet")
bf_ground_truth = pl.read_parquet("/Users/nathanleroy/Downloads/brute_force_queries_1000_k1000.parquet")

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
        return {col: table.column(col)[lo - rg_starts[rg_idx]].as_py() for col in ENRICH_COLUMNS}


def query_qdrant(row: dict) -> list[str]:
    result = client.query_points(
        collection_name=COLLECTION,
        query=row[DENSE_COLUMN],
        limit=K,
        with_payload=True,
        with_vectors=False,
        using=DENSE_VECTOR,
    )
    return [str(p.id) for p in result.points]


def recall_one(row: dict, qdrant_hit_ids: list[str]) -> float:
    qid = row["point_id"]
    bf_row = bf_ground_truth.filter(pl.col("query_id") == qid)
    if bf_row.is_empty():
        return None
    bf_set = set(bf_row["hit_ids"][0][:K])
    return len(bf_set & set(qdrant_hit_ids)) / K


row = queries.row(random.randrange(len(queries)), named=True)

text = enrich_one(row)
hits = query_qdrant(row)
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

# #  full recall sweep
# all_recalls = []
# for row in queries.iter_rows(named=True):
#     hits = query_qdrant(row)
#     all_recalls.append(recall_one(row, hits))

# all_recalls = [r for r in all_recalls if r is not None]
# print(f"\nRecall@{K}: {sum(all_recalls)/len(all_recalls):.4f}  ({len(all_recalls)} queries)")

# recall_df = pl.DataFrame({"query_id": queries["point_id"].to_list(), "recall": all_recalls})
# recall_df.sort("recall").head(10)   # worst queries


import hashlib, uuid

key = row['__source_file__']
target = result.points[0].id

# brute-force the actual DuckDB row number — file has at most ~200k rows                                                                                                                                                                                                                
for i in range(200_000):
    if str(uuid.UUID(hashlib.md5(f"{key}:{i}".encode()).hexdigest())) == target:
        print(f"loader used row {i}, pyarrow stored __source_row__ {row['__source_row__']}")
        break
    else:
        print("no match — filename must still be wrong")

import boto3
import duckdb
import polars as pl

# ── download the specific file ─────────────────────────────────────────────
local = '/tmp/test_batch.parquet'
boto3.client('s3').download_file(BUCKET, key, local)

# ── polars: reads in file order (row group 0, 1, 2...) ────────────────────
df_pl = pl.read_parquet(local, columns=['rendered_text'])

# ── duckdb 2 threads (what the loader used) ───────────────────────────────
conn2 = duckdb.connect()
conn2.execute("SET threads = 2")
rn_to_text_2t = dict(conn2.execute(
    "SELECT ROW_NUMBER() OVER () - 1, rendered_text FROM read_parquet(?)", [local]
).fetchall())

# ── duckdb 1 thread (sequential) ──────────────────────────────────────────
conn1 = duckdb.connect()
conn1.execute("SET threads = 1")
rn_to_text_1t = dict(conn1.execute(
    "SELECT ROW_NUMBER() OVER () - 1, rendered_text FROM read_parquet(?)", [local]
).fetchall())

pyarrow_row = row['__source_row__']   # 36531
duckdb_row  = 18099                    # what the loader actually assigned

# these should print True / False / True
print("polars[36531] == duckdb_2t[18099]:", df_pl['rendered_text'][pyarrow_row][:80] == rn_to_text_2t[duckdb_row][:80])
print("polars[36531] == duckdb_2t[36531]:", df_pl['rendered_text'][pyarrow_row][:80] == rn_to_text_2t[pyarrow_row][:80])
print("polars[36531] == duckdb_1t[36531]:", df_pl['rendered_text'][pyarrow_row][:80] == rn_to_text_1t[pyarrow_row][:80])

# iterate over rows in zip(df_pl.rows(), and rn_to_text_1t.items()) to find the first row where polars and duckdb 1t DONT match
for i, (pl_row, (duck_i, duck_text)) in enumerate(zip(df_pl.rows(), rn_to_text_1t.items())):
    if pl_row[0] != duck_text:
        print(f"first mismatch at row {i}:")
        print(f"  polars: {pl_row[0][:80]}")
        print(f"  duckdb: {duck_text[:80]}")
        break

# iterate over rows in zip(df_pl.rows(), and rn_to_text_2t.items()) to find the first row where polars and duckdb 2t DONT match
for i, (pl_row, (duck_i, duck_text)) in enumerate(zip(df_pl.rows(), rn_to_text_2t.items())):
    if pl_row[0] != duck_text:
        print(f"first mismatch at row {i}:")
        print(f"  polars: {pl_row[0][:80]}")
        print(f"  duckdb: {duck_text[:80]}")
        break
