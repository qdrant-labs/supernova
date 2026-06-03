import duckdb

S3_PATH = "s3://qdrant--vectorforge/finewiki/embed-gte-multilingual-base/**/*.parquet"
S3_PATH = "s3://qdrant--vectorforge/fineweb/embedder-bge-large-en-v1.5/cc-main-2025-26/**/*.parquet"
S3_PATH = "s3://poshmark-benchmark/listings_90_days_may_4_with_lattice_embeddings_spark_method/*.parquet"
S3_PATH = "s3://ziprecruiter-benchmark/jobs_index_data_monetized/*.parquet"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

con.sql(f"SELECT count(*) AS total_records FROM '{S3_PATH}'")
con.sql(f"DESCRIBE SELECT * FROM '{S3_PATH}'")

# select first 5 rows (text, dense_embedding, sparse_embedding
con.sql(f"""
    SELECT text, dense_embedding, sparse_embedding as bm25
    FROM '{S3_PATH}'
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
