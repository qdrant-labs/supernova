import duckdb

S3_PATH = "s3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/**/*.parquet"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

con.sql(
f"""
    SELECT
        count(*) AS total_records,
    FROM '{S3_PATH}'
""")
con.sql(f"DESCRIBE SELECT * FROM '{S3_PATH}'")

# query that counts length of 'abstract' text field
con.sql(
f"""
    SELECT
        length(text) AS text_length,
        count(*) AS count
    FROM '{S3_PATH}'
    GROUP BY text_length
    ORDER BY text_length DESC
    LIMIT 10
""")

# select fisrt 5 records with text
con.sql(
f"""
    SELECT
        text
    FROM '{S3_PATH}'
    WHERE text IS NOT NULL
    LIMIT 5
""").df()