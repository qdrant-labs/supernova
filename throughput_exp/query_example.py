import duckdb

S3_PATH = "s3://qdrant--vectorforge/finewiki/embed-gte-multilingual-base/**/*.parquet"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

con.sql(f"SELECT count(*) AS total_records FROM '{S3_PATH}'")
con.sql(f"DESCRIBE SELECT * FROM '{S3_PATH}'")

# select first 5 rows (text, dense_embedding, sparse_embedding
con.sql(f"""
    SELECT title, text, dense_embedding, sparse_embedding as bm25, in_language
    FROM '{S3_PATH}'
    WHERE in_language = 'en'
    LIMIT 5
    OFFSET 1000000
""")

# count total records and number of unique files
con.sql(f"""
  select count(*) as total, count(distinct filename) as n_files
  from read_parquet(
    '{S3_PATH}',
    filename=true
  )
""")