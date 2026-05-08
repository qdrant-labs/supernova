import polars as pl
import numpy as np

from tqdm import tqdm

queries = pl.read_parquet("/Users/nathanleroy/Downloads/queries_1000.parquet")
sample_embeddings = pl.read_parquet(
    "/Users/nathanleroy/Downloads/batch_00000000.parquet"
)

# grab first query as numpy array
for row in tqdm(queries.iter_rows(), total=queries.shape[0]):
    query_embedding = row[0]  # assuming the embedding is in the first column
    query_embedding = np.array(query_embedding)  # convert to numpy array
    # grab all sample embeddings as numpy array
    sample_embeddings_array = np.stack(sample_embeddings["dense_embedding"].to_numpy())
    # compute cosine similarity between query and sample embeddings, attach to dataframe
    similarity_scores = np.dot(sample_embeddings_array, query_embedding) / (
        np.linalg.norm(sample_embeddings_array, axis=1)
        * np.linalg.norm(query_embedding)
        + 1e-10
    )
    sample_embeddings = sample_embeddings.with_columns(
        pl.Series("similarity", similarity_scores)
    )
