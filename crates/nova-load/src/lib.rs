pub mod config;
pub mod engine;
pub mod sources;
pub mod stores;

use config::LoadConfig;
use sources::DataSource;
use stores::CollectionSchema;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Source(#[from] sources::SourceError),
    #[error(transparent)]
    Store(#[from] stores::StoreError),
    #[error(transparent)]
    Engine(#[from] engine::EngineError),
    #[error("background read task panicked: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Run the load: list files, then for each file download → read with DuckDB →
/// upsert in batches. The collection is created (and indexing deferred) lazily
/// from the first file's data, so dimensions are inferred from the parquet.
///
/// Sequential for now: file-level partitioning for distributed runs and
/// intra-worker concurrency are TODOs.
pub async fn run_loader(config: LoadConfig) -> Result<(), LoadError> {
    let LoadConfig { datasource, vectorstore, vectors, loader } = config;
    let batch_size = loader.batch_size.max(1);
    let id_expression = datasource.reader().id_expression.clone();
    let payload = datasource.reader().payload_fields.clone();

    let store = vectorstore.connect().await?;
    let files = datasource.list_files().await?;
    if files.is_empty() {
        tracing::warn!("no files to load");
        return Ok(());
    }
    tracing::info!("loading {} file(s) into {store}", files.len());

    let mut total = 0u64;
    for (i, file) in files.iter().enumerate() {
        let local = datasource.fetch(file).await?;

        let read_job = engine::ReadJob {
            path: local.path().to_path_buf(),
            filename: local.source.key.clone(),
            vectors: vectors.clone(),
            payload: payload.clone(),
            id_expression: id_expression.clone(),
        };
        let points = tokio::task::spawn_blocking(move || read_job.run()).await??;
        drop(local); // read is done; delete the temp download

        // Create the collection from the first file's inferred dimensions, then
        // turn off the indexing optimizer for the bulk load.
        if i == 0 {
            let dims = engine::infer_dims(&points, &vectors);
            let schema = CollectionSchema { vectors: vectors.clone(), dims };
            store.ensure_collection(&schema).await?;
            store.defer_indexing().await?;
        }

        for chunk in points.chunks(batch_size) {
            store.upsert_batch(chunk.to_vec()).await?;
        }
        total += points.len() as u64;
        tracing::info!("{} ({}/{}) → {} points", file.key, i + 1, files.len(), points.len());
    }

    // Re-enable indexing and wait for it to settle before declaring done.
    store.enable_indexing().await?;
    store.wait_for_indexing().await?;
    store.close().await?;
    tracing::info!("done: {total} points loaded");
    Ok(())
}
