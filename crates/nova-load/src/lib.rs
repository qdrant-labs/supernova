pub mod config;
pub mod engine;
pub mod plan;
pub mod sources;
pub mod stores;

use std::collections::HashMap;
use std::io::IsTerminal;
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
    #[error("aborting: {skipped} file(s) failed after retries, exceeding loader.max_failed_files={max}")]
    TooManyFailedFiles { skipped: u64, max: usize },
}

// ---------------------------------------------------------------------------
// Public phases. A single-node run is `run` (prepare + load + finalize in one
// process). A distributed fleet splits them: a master runs `prepare`, every
// worker runs `load` with its own `--job-rank`, then the master runs `finalize`.
// ---------------------------------------------------------------------------

/// Single-node: create the collection, load every file, then finalize indexing.
pub async fn run(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;

    let dims = resolve_dims(&config.datasource, &config.vectors).await?;
    let schema = CollectionSchema { vectors: config.vectors.clone(), dims };
    store.ensure_collection(&schema).await?;
    store.defer_indexing().await?;

    // `load_files` lists + partitions internally.
    let all_files = config.datasource.list_files().await?;
    let n = load_files(
        store.as_ref(),
        &config.datasource,
        &config.vectors,
        &config.loader,
        all_files,
        Partition::single(),
    )
    .await?;

    finish_indexing(store.as_ref(), &schema).await?;
    tracing::info!("done: {n} points loaded");
    Ok(())
}

/// Master step: create the collection (inferring dimensions from the data) and
/// defer indexing for the bulk load. Idempotent unless `params.recreate` drops
/// an existing collection.
pub async fn prepare(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    // No file listing — dims come from sampling one file (or explicit config),
    // so prepare doesn't pay to enumerate a huge corpus.
    let dims = resolve_dims(&config.datasource, &config.vectors).await?;
    let schema = CollectionSchema { vectors: config.vectors.clone(), dims };
    store.ensure_collection(&schema).await?;
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
    // This process never called `ensure_collection`, so re-derive the schema
    // that `enable_indexing` needs (per-vector metric etc.). No dims needed here —
    // the collection already exists, so no backend creates fields in this phase —
    // which also means `finalize` doesn't have to reach the datasource to sample.
    let schema = CollectionSchema { vectors: config.vectors.clone(), dims: HashMap::new() };
    finish_indexing(store.as_ref(), &schema).await?;
    tracing::info!("finalized {store}");
    Ok(())
}

/// Re-apply index settings from config to an *already-existing* collection, then
/// wait for the change to reconverge — no data is touched. Uses the same
/// `LoadConfig` shape as every other phase. Backends read what they need from it:
/// Qdrant patches `vectorstore.params.{hnsw,quantization,optimizers}` in place;
/// Milvus drops+rebuilds each index with the configured `index_type`/`index_params`
/// and the per-vector `distance`.
pub async fn reindex(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    // Build the schema so backends that rebuild (Milvus/Elastic) can read
    // per-vector settings (metric). No dims needed — reindex touches an existing
    // collection, so nothing here creates fields; this also keeps reindex from
    // having to reach the datasource just to sample a dimension.
    let schema = CollectionSchema { vectors: config.vectors.clone(), dims: HashMap::new() };
    let started = Instant::now();
    store.reindex(&schema).await?;
    let converged_at = store.wait_for_indexing().await?;
    let effective_elapsed = converged_at.duration_since(started);
    tracing::info!("reindex timing: index_seconds={:.3}", effective_elapsed.as_secs_f64());
    tracing::info!("reindexed {store}");
    Ok(())
}

/// Delete the collection if it exists. A no-op if it doesn't.
pub async fn delete(config: LoadConfig) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    store.delete_collection().await?;
    tracing::info!("deleted {store}");
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

/// Resolve each vector's dimensions. Explicit `size:` in config wins (and needs
/// no file access at all); otherwise sample one row of *a* file and measure the
/// vectors. Sparse vectors have no fixed size and are simply absent from the map.
///
/// Uses `first_file()` rather than a full listing, so `prepare` on a 500k-object
/// corpus doesn't hang enumerating everything just to peek at one schema.
async fn resolve_dims(
    datasource: &DataSourceConfig,
    vectors: &HashMap<String, VectorSpec>,
) -> Result<HashMap<String, u64>, LoadError> {
    // If every dense/multivector spec has an explicit size, no read is needed.
    let needs_inference = vectors
        .values()
        .any(|s| s.size.is_none() && !matches!(s.kind, config::VectorKind::Sparse));
    if !needs_inference {
        return Ok(vectors.iter().filter_map(|(k, s)| s.size.map(|d| (k.clone(), d))).collect());
    }

    let first = datasource.first_file().await?.ok_or(LoadError::NoFilesToPrepare)?;
    let local = datasource.fetch(&first).await?;
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

async fn finish_indexing(
    store: &dyn VectorStore,
    schema: &CollectionSchema,
) -> Result<(), LoadError> {
    tracing::info!("re-enabling indexing…");
    // Time index building the same way `reindex` does: from kicking indexing off
    // (`enable_indexing`) to the instant the backend first reached its converged
    // state — `wait_for_indexing` returns that instant, so this EXCLUDES the
    // stability hold each backend adds on top (e.g. Qdrant's 5s green-hold,
    // Milvus's 30s pending-zero hold). It's the effective build time, not wall
    // clock. Backend-agnostic: reports for whichever store this is.
    let started = Instant::now();
    store.enable_indexing(schema).await?;
    let converged_at = store.wait_for_indexing().await?;
    // Each backend reports its own timing: build time (Qdrant/Milvus) vs ES's
    // inline index_time (Elasticsearch) — see `report_index_time`.
    store.report_index_time(converged_at.duration_since(started)).await;
    store.close().await?;
    Ok(())
}

/// Download one file and DuckDB-read it into points — the unit retried per file.
async fn fetch_and_read(
    datasource: &DataSourceConfig,
    file: &FileRef,
    vectors: &HashMap<String, VectorSpec>,
    payload: &HashMap<String, String>,
    id_expression: &str,
) -> Result<Vec<Point>, LoadError> {
    let local = datasource.fetch(file).await?;
    let read_job = engine::ReadJob {
        path: local.path().to_path_buf(),
        filename: local.source.key.clone(),
        vectors: vectors.clone(),
        payload: payload.clone(),
        id_expression: id_expression.to_string(),
        limit: None,
    };
    let points = tokio::task::spawn_blocking(move || read_job.run()).await??;
    drop(local); // read done; delete the temp download
    Ok(points)
}

/// The core load loop: partition the files, then download → DuckDB-read →
/// upsert each, with file prefetch and concurrent batch upserts. Returns the
/// number of points loaded. Does NOT touch indexing or collection creation.
///
/// Resilience: each file's download+read is retried (`loader.file_retries`) with
/// exponential backoff; a file that still fails is logged and **skipped** so one
/// bad object can't abort the whole load. `loader.max_failed_files` caps how many
/// skips are tolerated before aborting (a guard against silently skipping a whole
/// broken corpus).
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
    let file_retries = loader.file_retries;
    let upsert_retries = loader.upsert_retries;
    let max_failed_files = loader.max_failed_files;
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

    // Progress + a live aggregate upsert rate (points/sec across all the
    // concurrent in-flight upserts). On a TTY this is a pretty bar; on a headless
    // worker (fleet logs captured to a file, not a terminal) `indicatif` hides the
    // bar, so we instead emit the rate through `tracing` on a coarse interval —
    // see the rate block in the loop below.
    let tty = std::io::stderr().is_terminal();
    // Refresh the live bar ~4×/sec; on the fleet, log the rate every 10s so tens
    // of thousands of files don't flood the log.
    let rate_interval = if tty { 0.25 } else { 10.0 };
    let progress = if tty {
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:30.cyan/blue}] {pos}/{len} files · {msg}",
            )
            .expect("valid template")
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(Duration::from_millis(120));
        pb
    } else {
        ProgressBar::hidden()
    };

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
                // Retry the whole download+read with exponential backoff; a
                // re-fetch also recovers from a truncated/corrupt prior download.
                let mut attempt = 0u32;
                loop {
                    match fetch_and_read(datasource, file, vectors, payload, id_expression).await {
                        Ok(points) => return Ok::<Vec<Point>, (String, LoadError)>(points),
                        Err(err) => {
                            attempt += 1;
                            if attempt > file_retries as u32 {
                                // Exhausted retries: hand the key + error to the
                                // consumer to log and skip — don't abort the run.
                                return Err((file.key.clone(), err));
                            }
                            let backoff =
                                Duration::from_millis((250u64 << (attempt - 1)).min(30_000));
                            tracing::warn!(
                                "file `{}` failed (attempt {}/{}): {err}; retrying in {:?}",
                                file.key,
                                attempt,
                                file_retries + 1,
                                backoff,
                            );
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
            }
        })
        .buffered(look_ahead);
    tokio::pin!(reads);

    let progress = &progress;
    let points_done = &points_done;
    let rate_window = &rate_window;

    let mut skipped = 0u64;
    while let Some(outcome) = reads.next().await {
        let points = match outcome {
            Ok(points) => points,
            Err((key, err)) => {
                tracing::warn!(
                    "skipping file `{key}` after {} attempt(s): {err}",
                    file_retries + 1,
                );
                skipped += 1;
                if let Some(max) = max_failed_files
                    && skipped > max as u64
                {
                    return Err(LoadError::TooManyFailedFiles { skipped, max });
                }
                progress.inc(1);
                continue;
            }
        };

        // Upsert this file's batches with up to `concurrency` requests in flight.
        // `buffer_unordered` is the idiomatic bounded-concurrency primitive — a
        // semaphore over a stream of upsert futures — and lets each future borrow
        // `store` without `Arc`/spawning. Each landed batch refreshes the rate.
        futures::stream::iter(points.chunks(batch_size))
            .map(|chunk| async move {
                let n = chunk.len() as u64;
                // Retry transient store errors with backoff; a persistent failure
                // (store down / misconfigured) aborts the run via `?` on try_collect.
                let mut attempt = 0u32;
                loop {
                    match store.upsert_batch(chunk.to_vec()).await {
                        Ok(()) => break,
                        Err(err) => {
                            attempt += 1;
                            if attempt > upsert_retries as u32 {
                                return Err(err);
                            }
                            let backoff =
                                Duration::from_millis((250u64 << (attempt - 1)).min(30_000));
                            tracing::warn!(
                                "upsert batch failed (attempt {}/{}): {err}; retrying in {:?}",
                                attempt,
                                upsert_retries + 1,
                                backoff,
                            );
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
                let done = points_done.fetch_add(n, Ordering::Relaxed) + n;

                // Recompute the rate over the most recent window, then either
                // refresh the bar (TTY) or log it (headless fleet worker).
                let mut w = rate_window.lock().expect("rate window");
                let dt = w.0.elapsed().as_secs_f64();
                if dt >= rate_interval {
                    let rate = (done - w.1) as f64 / dt;
                    *w = (Instant::now(), done);
                    drop(w);
                    if tty {
                        progress.set_message(format!("{done} pts · {rate:.0} pts/s"));
                    } else {
                        tracing::info!("{done} pts · {rate:.0} pts/s");
                    }
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
    if !tty {
        tracing::info!("{total} pts · {avg:.0} pts/s avg");
    }
    if skipped > 0 {
        tracing::warn!(
            "{skipped} file(s) were SKIPPED after exhausting {} retries each — \
             the collection is missing their points",
            file_retries,
        );
    }
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
