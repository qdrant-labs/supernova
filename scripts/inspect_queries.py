import hashlib
import uuid

import polars as pl

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

# join bf_ground_truth on query_id = point_id to get the point_ids for the ground truth neighbors
bf_ground_truth = bf_ground_truth.join(
    queries.select(pl.col("point_id").alias("query_id")),
    left_on="query_id",
    right_on="query_id",
    how="inner",
)