pub mod config;
pub mod engine;
pub mod plan;
pub mod sources;
pub mod stores;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::{StreamExt, TryStreamExt};
use indicatif::{ProgressBar, ProgressStyle};

use config::{LoadConfig, LoaderConfig, VectorSpec};
use plan::Partition;
use sources::{DataSource, DataSourceConfig, FileRef};
use stores::{CollectionSchema, Point, StoreError, VectorStore};

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
    #[error("cannot prepare collection: the datasource has no files to infer dimensions from")]
    NoFilesToPrepare,
}

// ---------------------------------------------------------------------------
// Public phases. A single-node run is `run` (prepare + load + finalize in one
// process). A distributed fleet splits them: a master runs `prepare`, every
// worker runs `load` with its own `--job-rank`, then the master runs `finalize`.
// ---------------------------------------------------------------------------

/// Single-node: create the collection, load every file, then finalize indexing.
pub async fn run(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    let all_files = config.datasource.list_files().await?;

    let dims = resolve_dims(&config.datasource, &config.vectors, &all_files).await?;
    create_collection(store.as_ref(), &config.vectors, dims).await?;
    store.defer_indexing().await?;

    let n = load_files(
        store.as_ref(),
        &config.datasource,
        &config.vectors,
        &config.loader,
        all_files,
        Partition::single(),
    )
    .await?;

    finish_indexing(store.as_ref()).await?;
    tracing::info!("done: {n} points loaded");
    Ok(())
}

/// Master step: create the collection (inferring dimensions from the data) and
/// defer indexing for the bulk load. Idempotent unless `params.recreate` drops
/// an existing collection.
pub async fn prepare(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    let all_files = config.datasource.list_files().await?;

    let dims = resolve_dims(&config.datasource, &config.vectors, &all_files).await?;
    create_collection(store.as_ref(), &config.vectors, dims).await?;
    store.defer_indexing().await?;
    tracing::info!("prepared collection on {store}");
    Ok(())
}

/// Worker step: load this worker's slice of the files. Assumes the collection
/// already exists (run `prepare` first) and does NOT manage indexing.
pub async fn load(config: LoadConfig, partition: Partition) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    let all_files = config.datasource.list_files().await?;
    let n = load_files(
        store.as_ref(),
        &config.datasource,
        &config.vectors,
        &config.loader,
        all_files,
        partition,
    )
    .await?;
    tracing::info!("worker {}/{} done: {n} points", partition.rank, partition.num_jobs);
    Ok(())
}

/// Master step: re-enable indexing and wait for it to settle. Run once, after
/// every worker's `load` has completed.
pub async fn finalize(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    finish_indexing(store.as_ref()).await?;
    tracing::info!("finalized {store}");
    Ok(())
}

/// Inspect what a load *would* do without connecting to the store or
/// downloading anything: dump the resolved config (secrets redacted) and the
/// file list this worker would handle. Listing does hit the source (an S3
/// `ListObjects`, say), since enumerating files is the whole point.
pub async fn inspect(config: LoadConfig, partition: Partition) -> Result<(), LoadError> {
    println!("== nova-load inspect ==\n");
    println!("partition:   job_rank={} num_jobs={}", partition.rank, partition.num_jobs);
    println!("loader:      {:?}", config.loader);
    println!("vectorstore: {:?}", config.vectorstore);
    println!("datasource:  {:?}", config.datasource);
    println!("vectors:     {:?}", config.vectors);

    let all_files = config.datasource.list_files().await?;
    let mine = plan::partition(&all_files, partition);
    let total: u64 = mine.iter().filter_map(|f| f.size).sum();
    println!(
        "\nfiles: this worker loads {} of {} found ({} of its slice)",
        mine.len(),
        all_files.len(),
        human_size(total),
    );
    for f in &mine {
        let size = f.size.map(human_size).unwrap_or_else(|| "?".to_string());
        println!("  {} ({size})", f.key);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Resolve each vector's dimensions. Explicit `size:` in config wins; otherwise
/// sample one row of the first file and measure the vectors. Sparse vectors have
/// no fixed size and are simply absent from the map.
async fn resolve_dims(
    datasource: &DataSourceConfig,
    vectors: &HashMap<String, VectorSpec>,
    files: &[FileRef],
) -> Result<HashMap<String, u64>, LoadError> {
    // If every dense/multivector spec has an explicit size, no read is needed.
    let needs_inference = vectors
        .values()
        .any(|s| s.size.is_none() && !matches!(s.kind, config::VectorKind::Sparse));
    if !needs_inference {
        return Ok(vectors.iter().filter_map(|(k, s)| s.size.map(|d| (k.clone(), d))).collect());
    }

    let first = files.first().ok_or(LoadError::NoFilesToPrepare)?;
    let local = datasource.fetch(first).await?;
    let read_job = engine::ReadJob {
        path: local.path().to_path_buf(),
        filename: local.source.key.clone(),
        vectors: vectors.clone(),
        payload: HashMap::new(), // dims don't need payload
        id_expression: datasource.reader().id_expression.clone(),
        limit: Some(1),
    };
    let sample = tokio::task::spawn_blocking(move || read_job.run()).await??;
    Ok(engine::infer_dims(&sample, vectors))
}

async fn create_collection(
    store: &dyn VectorStore,
    vectors: &HashMap<String, VectorSpec>,
    dims: HashMap<String, u64>,
) -> Result<(), LoadError> {
    let schema = CollectionSchema { vectors: vectors.clone(), dims };
    store.ensure_collection(&schema).await?;
    Ok(())
}

async fn finish_indexing(store: &dyn VectorStore) -> Result<(), LoadError> {
    tracing::info!("re-enabling indexing…");
    store.enable_indexing().await?;
    store.wait_for_indexing().await?;
    store.close().await?;
    Ok(())
}

/// The core load loop: partition the files, then download → DuckDB-read →
/// upsert each, with file prefetch and concurrent batch upserts. Returns the
/// number of points loaded. Does NOT touch indexing or collection creation.
async fn load_files(
    store: &dyn VectorStore,
    datasource: &DataSourceConfig,
    vectors: &HashMap<String, VectorSpec>,
    loader: &LoaderConfig,
    all_files: Vec<FileRef>,
    partition: Partition,
) -> Result<u64, LoadError> {
    let batch_size = loader.batch_size.max(1);
    let concurrency = loader.concurrency.max(1);
    let look_ahead = loader.file_look_ahead.max(1);
    let id_expression = datasource.reader().id_expression.clone();
    let payload = datasource.reader().payload_fields.clone();

    let total_files = all_files.len();
    let files = plan::partition(&all_files, partition);
    if files.is_empty() {
        tracing::warn!(
            "worker {}/{} has no files of {total_files} to load (more workers than files?)",
            partition.rank,
            partition.num_jobs,
        );
        return Ok(0);
    }
    tracing::info!(
        "worker {}/{} loading {} of {total_files} file(s) into {store}",
        partition.rank,
        partition.num_jobs,
        files.len(),
    );

    // Files progress + a live aggregate upsert rate (points/sec across all the
    // concurrent in-flight upserts). The bar draws to stderr, clear of the
    // stdout logs.
    let progress = ProgressBar::new(files.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} files · {msg}",
        )
        .expect("valid template")
        .progress_chars("=>-"),
    );
    progress.enable_steady_tick(Duration::from_millis(120));

    // `points_done` is the running total; `rate_window` holds the last sample
    // (time, count) for an instantaneous rate, vs. the cumulative average at end.
    let points_done = AtomicU64::new(0);
    let rate_window = std::sync::Mutex::new((Instant::now(), 0u64));
    let started = Instant::now();
    let mut total = 0u64;

    // Prefetch pipeline: download + DuckDB-read up to `look_ahead` files
    // concurrently while the current file's batches are still uploading, so the
    // store connection isn't stalled on S3/parse time. `buffered` keeps file
    // order and bounds how many reads are in flight (and how much sits in RAM).
    let reads = futures::stream::iter(files.iter())
        .map(|file| {
            let datasource = &datasource;
            let vectors = &vectors;
            let payload = &payload;
            let id_expression = &id_expression;
            async move {
                let local = datasource.fetch(file).await?;
                let read_job = engine::ReadJob {
                    path: local.path().to_path_buf(),
                    filename: local.source.key.clone(),
                    vectors: (*vectors).clone(),
                    payload: payload.clone(),
                    id_expression: id_expression.clone(),
                    limit: None,
                };
                let points = tokio::task::spawn_blocking(move || read_job.run()).await??;
                drop(local); // read done; delete the temp download
                Ok::<Vec<Point>, LoadError>(points)
            }
        })
        .buffered(look_ahead);
    tokio::pin!(reads);

    let progress = &progress;
    let points_done = &points_done;
    let rate_window = &rate_window;

    while let Some(points) = reads.next().await {
        let points = points?;

        // Upsert this file's batches with up to `concurrency` requests in flight.
        // `buffer_unordered` is the idiomatic bounded-concurrency primitive — a
        // semaphore over a stream of upsert futures — and lets each future borrow
        // `store` without `Arc`/spawning. Each landed batch refreshes the rate.
        futures::stream::iter(points.chunks(batch_size))
            .map(|chunk| async move {
                let n = chunk.len() as u64;
                store.upsert_batch(chunk.to_vec()).await?;
                let done = points_done.fetch_add(n, Ordering::Relaxed) + n;

                // Recompute the rate over the most recent window (~4×/sec).
                let mut w = rate_window.lock().expect("rate window");
                let dt = w.0.elapsed().as_secs_f64();
                if dt >= 0.25 {
                    let rate = (done - w.1) as f64 / dt;
                    *w = (Instant::now(), done);
                    drop(w);
                    progress.set_message(format!("{done} pts · {rate:.0} pts/s"));
                }
                Ok::<(), StoreError>(())
            })
            .buffer_unordered(concurrency)
            .try_collect::<Vec<()>>()
            .await?;

        total += points.len() as u64;
        progress.inc(1);
    }

    let avg = total as f64 / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
    progress.finish_with_message(format!("{total} pts · {avg:.0} pts/s avg"));
    Ok(total)
}

/// Human-readable byte count (e.g. `2.4 MB`).
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}
