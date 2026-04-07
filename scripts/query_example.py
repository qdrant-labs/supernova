import os

import duckdb

# Point this at your S3 path or local output dir
S3_PATH = "s3://qdrant--vectorforge/cohere--wikipedia/embed-multilingual-v3/**/*.parquet"
LOCAL_PATH = "/tmp/vectorforge/*.parquet"

# Use local path by default — swap to S3_PATH if querying from S3
PATH = LOCAL_PATH
PATH = S3_PATH

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
con.execute(f"SET s3_region = '{region}';")

key = os.environ.get("AWS_ACCESS_KEY_ID", "")
secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
if key and secret:
    con.execute(f"SET s3_access_key_id = '{key}';")
    con.execute(f"SET s3_secret_access_key = '{secret}';")

con.sql(
f"""
    SELECT
        count(*) AS total_records,
    FROM '{PATH}'
""")
con.sql(f"DESCRIBE SELECT * FROM '{PATH}'")
con.sql(
f"""
    SELECT row_id, source_row_id, chunk_index, source, text[:80] AS text_preview, model, embedding
    FROM '{PATH}'
    LIMIT 10
""")

