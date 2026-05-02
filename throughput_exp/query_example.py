import duckdb

S3_PATH = "s3://qdrant--vectorforge/finewiki/embed-gte-multilingual-base/en/**/*.parquet"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

con.sql(f"SELECT count(*) AS total_records FROM '{S3_PATH}'")
con.sql(f"DESCRIBE SELECT * FROM '{S3_PATH}'")

con.sql("""
  WITH t AS (                                               
    SELECT text,                                                                                         
           count(DISTINCT regexp_extract(filename, 'rank(\d+)', 1)) AS n_ranks
    FROM read_parquet(                                                                                   
      's3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/2016/**/*.parquet',         
      filename=true                                                                                      
    )                                                                                                    
    GROUP BY text                                                                                        
  )                                                         
  SELECT n_ranks, count(*) AS n_texts
  FROM t
  GROUP BY n_ranks
  ORDER BY n_ranks;
""")

# query that counts length of 'abstract' text field
con.sql(
"""
DESCRIBE SELECT * FROM read_parquet('s3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/2016/rank00_batch_00000000.parquet');
""")
con.sql("""
  -- (b) is (filename, row_id) the only true global key? must be -- row_id is a within-file counter      
  SELECT count(*) AS total, count(DISTINCT (filename, row_id)) AS unique_file_rid
  FROM read_parquet(
    's3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/2016/**/*.parquet',
    filename=true
  );
""")

con.sql("""
select distinct filename
from read_parquet(
  's3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/2016/**/*.parquet',
  filename=true
)
""")


S3_PATH="s3://qdrant--vectorforge/huggingface-tb--dclm-edu/embed-gte-multilingual-base/rank25/batch_00000000.parquet"
# select fisrt 5 records with text
# convert to polars
con.sql(
f"""
    SELECT text, language, language_score
    FROM '{S3_PATH}'
    WHERE text IS NOT NULL
    LIMIT 5
""")