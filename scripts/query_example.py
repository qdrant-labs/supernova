import duckdb

# Point this at your S3 path or local output dir
S3_PATH = "s3://qdrant--vectorforge/mteb--tweet-sentiment/openai-3-small/*.parquet"
LOCAL_PATH = "/tmp/vectorforge/*.parquet"

# Use local path by default — swap to S3_PATH if querying from S3
PATH = LOCAL_PATH
PATH = S3_PATH

con = duckdb.connect()

con.sql(
f"""
    SELECT
        count(*) AS total_records,
        count(DISTINCT source_row_id) AS unique_source_rows,
        count(DISTINCT chunk_id) AS chunks,
        min(row_id) AS min_row_id,
        max(row_id) AS max_row_id
    FROM '{PATH}'
""")
con.sql(f"DESCRIBE SELECT * FROM '{PATH}'")
con.sql(
f"""
    SELECT row_id, source_row_id, chunk_index, source, text[:80] AS text_preview, model, embedding
    FROM '{PATH}'
    LIMIT 10
""")

