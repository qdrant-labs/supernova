pub mod catalog;
pub mod checkpoint;
pub mod config;
pub mod engine;
pub mod plan;
pub mod sources;
pub mod stores;

use std::collections::{BTreeSet, HashMap};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;

use config::{LoadConfig, LoaderConfig, VectorSpec};
use plan::Partition;
use sources::{DataSource, DataSourceConfig, FileRef};
use stores::{CollectionSchema, Point, StoreError, VectorStore};

use crate::checkpoint::{CheckpointMeta, CheckpointState};

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
    #[error("invalid loader options: {0}")]
    InvalidOptions(String),
    #[error(transparent)]
    Checkpoint(#[from] checkpoint::CheckpointError),
}

#[derive(Debug, Clone, Default)]
pub struct LoadRuntimeOptions {
    pub resume: bool,
    pub checkpoint_path: Option<PathBuf>,
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
    let schema = create_collection(store.as_ref(), &config.vectors, dims).await?;
    store.ensure_payload_indexes(&config.datasource.reader().payload_fields).await?;
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
        &LoadRuntimeOptions::default(),
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
    create_collection(store.as_ref(), &config.vectors, dims).await?;
    store.ensure_payload_indexes(&config.datasource.reader().payload_fields).await?;
    store.defer_indexing().await?;
    tracing::info!("prepared collection on {store}");
    Ok(())
}

/// Worker step: load this worker's slice of the files. Assumes the collection
/// already exists (run `prepare` first) and does NOT manage indexing.
pub async fn load(
    config: LoadConfig,
    partition: Partition,
    runtime: LoadRuntimeOptions,
) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    let all_files = config.datasource.list_files().await?;
    let n = load_files(
        store.as_ref(),
        &config.datasource,
        &config.vectors,
        &config.loader,
        all_files,
        partition,
        &runtime,
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
        duckdb_memory_limit: datasource.reader().duckdb_memory_limit.clone(),
        duckdb_threads: datasource.reader().duckdb_threads,
    };
    let sample = tokio::task::spawn_blocking(move || read_job.run()).await??;
    Ok(engine::infer_dims(&sample, vectors))
}

async fn create_collection(
    store: &dyn VectorStore,
    vectors: &HashMap<String, VectorSpec>,
    dims: HashMap<String, u64>,
) -> Result<CollectionSchema, LoadError> {
    let schema = CollectionSchema { vectors: vectors.clone(), dims };
    store.ensure_collection(&schema).await?;
    Ok(schema)
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

/// A file's read in progress: points arrive in `batch_size` chunks over `rx`
/// instead of being materialized into one `Vec` up front (some files in this
/// corpus decode to tens of GB once turned into `Point`s). Holding `_local`
/// keeps the temp download alive until the read finishes; `task` is awaited
/// once `rx` closes, to surface a DuckDB-side error or a panic.
struct StreamedRead {
    rx: tokio::sync::mpsc::Receiver<Result<Vec<Point>, engine::EngineError>>,
    task: tokio::task::JoinHandle<()>,
    _local: sources::LocalFile,
}

/// Download one file and start a bounded-memory streamed DuckDB read of it —
/// the unit retried per file. The channel bound (2) is what actually caps
/// memory: the blocking read task parks once it's a couple of chunks ahead of
/// whatever's draining `rx`, so at most a couple of `batch_size` chunks are
/// ever resident, regardless of how many rows the file has.
async fn fetch_and_read_streamed(
    datasource: &DataSourceConfig,
    file: &FileRef,
    vectors: &HashMap<String, VectorSpec>,
    payload: &HashMap<String, String>,
    id_expression: &str,
    batch_size: usize,
) -> Result<StreamedRead, LoadError> {
    let local = datasource.fetch(file).await?;
    let read_job = engine::ReadJob {
        path: local.path().to_path_buf(),
        filename: local.source.key.clone(),
        vectors: vectors.clone(),
        payload: payload.clone(),
        id_expression: id_expression.to_string(),
        limit: None,
        duckdb_memory_limit: datasource.reader().duckdb_memory_limit.clone(),
        duckdb_threads: datasource.reader().duckdb_threads,
    };
    let (tx, rx) = tokio::sync::mpsc::channel(2);
    let task = tokio::task::spawn_blocking(move || read_job.run_streamed(batch_size, tx));
    Ok(StreamedRead { rx, task, _local: local })
}

/// Upsert one already-`batch_size`-capped chunk, retrying transient store
/// errors with backoff. A persistent failure (store down / misconfigured) is
/// fatal — the caller propagates it immediately, unlike a read failure, which
/// retries the whole file.
#[allow(clippy::too_many_arguments)]
async fn upsert_one_batch(
    store: &dyn VectorStore,
    chunk: Vec<Point>,
    upsert_retries: usize,
    points_done: &AtomicU64,
    rate_window: &std::sync::Mutex<(Instant, u64)>,
    tty: bool,
    rate_interval: f64,
    progress: &ProgressBar,
) -> Result<u64, StoreError> {
    let n = chunk.len() as u64;
    let mut attempt = 0u32;
    loop {
        match store.upsert_batch(chunk.clone()).await {
            Ok(()) => break,
            Err(err) => {
                attempt += 1;
                if attempt > upsert_retries as u32 {
                    return Err(err);
                }
                let backoff = Duration::from_millis((250u64 << (attempt - 1)).min(30_000));
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
    Ok(n)
}

/// Drain an already-started file read and upsert its chunks as they arrive,
/// applying `remaining_row_offset`/`points_budget` in receive order (so those
/// shared, cross-file counters stay correct) before handing each chunk to a
/// pool of up to `concurrency` concurrent upserts — the same pipelining the
/// old "collect the whole file, then `.chunks(batch_size)`" code had, just
/// without ever holding more than `concurrency` chunks at once.
///
/// Returns `(points loaded from this file, whether it stopped early on
/// budget)`. An `Err` here means the read failed (DuckDB/download/panic) or
/// upserting exhausted its own retries — the caller distinguishes the two via
/// `LoadError::Store` (fatal, propagate as-is) vs. anything else (retry the
/// whole file).
#[allow(clippy::too_many_arguments)]
async fn drain_and_upsert(
    mut streamed: StreamedRead,
    store: &dyn VectorStore,
    concurrency: usize,
    upsert_retries: usize,
    remaining_row_offset: &mut u64,
    points_budget: &mut Option<u64>,
    points_done: &AtomicU64,
    rate_window: &std::sync::Mutex<(Instant, u64)>,
    tty: bool,
    rate_interval: f64,
    progress: &ProgressBar,
) -> Result<(u64, bool), LoadError> {
    let mut inflight = FuturesUnordered::new();
    let mut file_total = 0u64;
    let mut budget_hit = false;
    let mut channel_closed = false;

    loop {
        while inflight.len() < concurrency && !channel_closed && !budget_hit {
            if *points_budget == Some(0) {
                budget_hit = true;
                break;
            }
            let Some(item) = streamed.rx.recv().await else {
                channel_closed = true;
                break;
            };
            let chunk = item?;
            let (chunk, partial) =
                slice_for_offset_and_budget(chunk, remaining_row_offset, points_budget);
            if !chunk.is_empty() {
                inflight.push(upsert_one_batch(
                    store,
                    chunk,
                    upsert_retries,
                    points_done,
                    rate_window,
                    tty,
                    rate_interval,
                    progress,
                ));
            }
            if partial {
                budget_hit = true;
                break;
            }
        }

        let Some(result) = inflight.next().await else {
            // Nothing in flight, and the loop above stopped pulling — either
            // the channel closed (file fully read) or we hit budget.
            break;
        };
        file_total += result?;
    }

    // Signal the blocking read task to stop early if it's still going (budget
    // hit before the channel closed on its own), then wait for it to actually
    // finish so any DuckDB-side error or panic surfaces here, not silently.
    drop(streamed.rx);
    streamed.task.await.map_err(LoadError::Join)?;
    Ok((file_total, budget_hit))
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
    runtime: &LoadRuntimeOptions,
) -> Result<u64, LoadError> {
    let batch_size = loader.batch_size.max(1);
    let concurrency = loader.concurrency.max(1);
    let look_ahead = loader.file_look_ahead.max(1);
    let file_retries = loader.file_retries;
    let upsert_retries = loader.upsert_retries;
    let max_failed_files = loader.max_failed_files;
    let max_points = loader.max_points;
    let row_offset = loader.row_offset.unwrap_or(0);
    let file_seed = loader.file_seed;
    let id_expression = datasource.reader().id_expression.clone();
    let payload = datasource.reader().payload_fields.clone();

    if runtime.resume && row_offset > 0 {
        return Err(LoadError::InvalidOptions(
            "loader.row_offset cannot be used with --resume; checkpoint resume is file-based and \
             incompatible with row-level offsets".to_string(),
        ));
    }

    let total_files = all_files.len();
    let mut files = plan::partition(&all_files, partition);
    shuffle_files(&mut files, file_seed);
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

    let mut checkpoint_ctx =
        maybe_init_checkpoint(datasource, vectors, loader, partition, runtime, &files)?;
    if let Some(ctx) = checkpoint_ctx.as_mut() {
        let (pending, resumed) = filter_pending_files(files, &ctx.state.completed_files);
        files = pending;
        if resumed > 0 {
            tracing::info!(
                "resume: skipping {resumed} already-completed file(s) from checkpoint `{}`",
                ctx.path.display()
            );
        }
        if files.is_empty() {
            tracing::info!(
                "resume: worker {}/{} has no pending files left to process",
                partition.rank,
                partition.num_jobs
            );
            return Ok(0);
        }
    }

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

    // Prefetch pipeline: download + start the streamed DuckDB read for up to
    // `look_ahead` files concurrently while the current file's batches are
    // still uploading, so the store connection isn't stalled on S3/parse time.
    // `buffered` keeps file order and bounds how many downloads+reads are
    // kicked off at once — it no longer bounds RAM directly (the streamed read
    // itself does that, via `fetch_and_read_streamed`'s bounded channel), so
    // this is now mainly about not opening unbounded concurrent S3 downloads.
    let reads = futures::stream::iter(files.iter())
        .map(|file| {
            let datasource = &datasource;
            let vectors = &vectors;
            let payload = &payload;
            let id_expression = &id_expression;
            async move {
                match fetch_and_read_streamed(
                    datasource,
                    file,
                    vectors,
                    payload,
                    id_expression,
                    batch_size,
                )
                .await
                {
                    Ok(streamed) => Ok((file, streamed)),
                    Err(err) => Err((file, err)),
                }
            }
        })
        .buffered(look_ahead);
    tokio::pin!(reads);

    let progress = &progress;
    let points_done = &points_done;
    let rate_window = &rate_window;

    let mut points_budget = max_points;
    let mut remaining_row_offset = row_offset;
    let mut bounded_stop = false;
    let mut skipped = 0u64;
    while let Some(prefetched) = reads.next().await {
        if points_budget == Some(0) {
            bounded_stop = true;
            break;
        }
        let file = match &prefetched {
            Ok((file, _)) => *file,
            Err((file, _)) => *file,
        };
        let key = file.key.clone();

        // Attempt 1 is whatever `look_ahead` already prefetched. A retry (rare
        // — read failures, not upsert failures, which are fatal and returned
        // immediately below) starts a fresh, non-prefetched streamed read;
        // retries are the exceptional path, so losing prefetch for them isn't
        // a meaningful throughput hit.
        let mut attempt = 0u32;
        let mut current: Result<StreamedRead, LoadError> = match prefetched {
            Ok((_, streamed)) => Ok(streamed),
            Err((_, err)) => Err(err),
        };
        let file_outcome = loop {
            match current {
                Ok(streamed) => {
                    match drain_and_upsert(
                        streamed,
                        store,
                        concurrency,
                        upsert_retries,
                        &mut remaining_row_offset,
                        &mut points_budget,
                        points_done,
                        rate_window,
                        tty,
                        rate_interval,
                        progress,
                    )
                    .await
                    {
                        Ok(outcome) => break Ok(outcome),
                        // Upsert exhausted its own retries — fatal, matches the
                        // original code's `?` propagation out of `load_files`.
                        Err(LoadError::Store(err)) => return Err(LoadError::Store(err)),
                        Err(err) => {
                            attempt += 1;
                            if attempt > file_retries as u32 {
                                break Err(err);
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
                            current = fetch_and_read_streamed(
                                datasource,
                                file,
                                vectors,
                                &payload,
                                &id_expression,
                                batch_size,
                            )
                            .await;
                        }
                    }
                }
                Err(err) => {
                    attempt += 1;
                    if attempt > file_retries as u32 {
                        break Err(err);
                    }
                    let backoff = Duration::from_millis((250u64 << (attempt - 1)).min(30_000));
                    tracing::warn!(
                        "file `{}` failed (attempt {}/{}): {err}; retrying in {:?}",
                        file.key,
                        attempt,
                        file_retries + 1,
                        backoff,
                    );
                    tokio::time::sleep(backoff).await;
                    current = fetch_and_read_streamed(
                        datasource,
                        file,
                        vectors,
                        &payload,
                        &id_expression,
                        batch_size,
                    )
                    .await;
                }
            }
        };

        match file_outcome {
            Ok((n, partial_file_due_to_budget)) => {
                total += n;
                points_budget = points_budget.map(|remaining| remaining.saturating_sub(n));
                if let Some(0) = points_budget {
                    bounded_stop = true;
                }
                if let Some(ctx) = checkpoint_ctx.as_mut()
                    && !partial_file_due_to_budget
                {
                    ctx.state.completed_files.insert(key);
                    ctx.completed_since_flush += 1;
                    if ctx.completed_since_flush >= ctx.flush_every_files {
                        checkpoint::save(&ctx.path, &ctx.meta, &ctx.state)?;
                        ctx.completed_since_flush = 0;
                    }
                }
                progress.inc(1);
                if bounded_stop {
                    break;
                }
            }
            Err(err) => {
                // Exhausted retries: log and skip — don't abort the whole run
                // for one bad file.
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
            }
        }
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
    if bounded_stop {
        tracing::info!(
            "stopped early due to loader.max_points={:?}; loaded {total} point(s) this run",
            max_points
        );
    }
    if let Some(ctx) = checkpoint_ctx.as_mut() {
        checkpoint::save(&ctx.path, &ctx.meta, &ctx.state)?;
    }
    Ok(total)
}

fn shuffle_files(files: &mut [FileRef], seed: Option<u64>) {
    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        files.shuffle(&mut rng);
    }
}

fn slice_for_offset_and_budget<T>(
    mut items: Vec<T>,
    remaining_row_offset: &mut u64,
    points_budget: &mut Option<u64>,
) -> (Vec<T>, bool) {
    if *remaining_row_offset > 0 {
        let skip = (*remaining_row_offset).min(items.len() as u64) as usize;
        if skip > 0 {
            items.drain(0..skip);
            *remaining_row_offset -= skip as u64;
        }
    }

    let mut partial_file_due_to_budget = false;
    if let Some(remaining) = *points_budget
        && (items.len() as u64) > remaining
    {
        items.truncate(remaining as usize);
        partial_file_due_to_budget = true;
    }
    (items, partial_file_due_to_budget)
}

struct CheckpointContext {
    path: PathBuf,
    meta: CheckpointMeta,
    state: CheckpointState,
    flush_every_files: usize,
    completed_since_flush: usize,
}

fn maybe_init_checkpoint(
    datasource: &DataSourceConfig,
    vectors: &HashMap<String, VectorSpec>,
    loader: &LoaderConfig,
    partition: Partition,
    runtime: &LoadRuntimeOptions,
    assigned_files: &[FileRef],
) -> Result<Option<CheckpointContext>, LoadError> {
    let cfg = loader.checkpoint.as_ref();
    let enabled = runtime.resume
        || runtime.checkpoint_path.is_some()
        || cfg.is_some_and(|c| c.enabled);
    if !enabled {
        return Ok(None);
    }

    let meta = CheckpointMeta {
        rank: partition.rank,
        num_jobs: partition.num_jobs,
        datasource_identity: datasource_identity(datasource),
        config_fingerprint: config_fingerprint(datasource, vectors, loader),
    };
    let path = match &runtime.checkpoint_path {
        Some(path) => checkpoint::scoped_path(path, partition.rank, partition.num_jobs),
        None => checkpoint::default_checkpoint_path(
            cfg.and_then(|c| c.path_prefix.as_deref()),
            &meta,
        ),
    };
    let state = checkpoint::load_for_run(&path, &meta, runtime.resume)?;

    let assigned: BTreeSet<String> = assigned_files.iter().map(|f| f.key.clone()).collect();
    let unknown = state
        .completed_files
        .iter()
        .filter(|k| !assigned.contains(*k))
        .count();
    if unknown > 0 {
        tracing::warn!(
            "checkpoint `{}` has {unknown} completed file(s) not in this worker partition",
            path.display()
        );
    }

    Ok(Some(CheckpointContext {
        path,
        meta,
        state,
        flush_every_files: cfg.map_or(1, |c| c.flush_every_files.max(1)),
        completed_since_flush: 0,
    }))
}

fn datasource_identity(datasource: &DataSourceConfig) -> String {
    match datasource {
        DataSourceConfig::Local(c) => {
            let mut out = format!("local:path={}", c.path);
            if let Some(list) = &c.file_list {
                out.push_str(&format!(":file_list={}", normalize_list(list)));
            }
            out
        }
        DataSourceConfig::S3(c) => {
            let mut out = format!("s3:path={}", c.path);
            if let Some(list) = &c.file_list {
                out.push_str(&format!(":file_list={}", normalize_list(list)));
            }
            if let Some(catalog) = &c.catalog {
                out.push_str(&format!(":catalog={catalog}"));
            }
            out
        }
    }
}

fn config_fingerprint(
    datasource: &DataSourceConfig,
    vectors: &HashMap<String, VectorSpec>,
    loader: &LoaderConfig,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("source={};", datasource_identity(datasource)));

    let mut vector_names = vectors.keys().cloned().collect::<Vec<_>>();
    vector_names.sort();
    for name in vector_names {
        if let Some(v) = vectors.get(&name) {
            out.push_str(&format!(
                "vec:{name}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?};",
                v.kind,
                v.column,
                v.size,
                v.distance,
                v.comparator,
                v.datatype,
                v.on_disk,
                v.modifier
            ));
        }
    }
    out.push_str(&format!(
        "loader:{}:{}:{}:{}:{}:{:?}:{:?}:{:?}:{:?};",
        loader.batch_size,
        loader.concurrency,
        loader.file_look_ahead,
        loader.file_retries,
        loader.upsert_retries,
        loader.max_failed_files,
        loader.max_points,
        loader.row_offset,
        loader.file_seed
    ));
    out
}

fn normalize_list(list: &[String]) -> String {
    let mut v = list.to_vec();
    v.sort();
    v.join(",")
}

fn filter_pending_files(
    files: Vec<FileRef>,
    completed: &BTreeSet<String>,
) -> (Vec<FileRef>, usize) {
    let total = files.len();
    let pending = files
        .into_iter()
        .filter(|f| !completed.contains(&f.key))
        .collect::<Vec<_>>();
    let skipped = total - pending.len();
    (pending, skipped)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn refs(keys: &[&str]) -> Vec<FileRef> {
        keys.iter()
            .map(|k| FileRef {
                key: (*k).to_string(),
                size: None,
            })
            .collect()
    }

    #[test]
    fn filter_pending_files_skips_completed_entries() {
        let files = refs(&["a.parquet", "b.parquet", "c.parquet"]);
        let completed = BTreeSet::from([
            "a.parquet".to_string(),
            "c.parquet".to_string(),
            "outside.parquet".to_string(),
        ]);

        let (pending, skipped) = filter_pending_files(files, &completed);
        let keys = pending.into_iter().map(|f| f.key).collect::<Vec<_>>();

        assert_eq!(keys, vec!["b.parquet".to_string()]);
        assert_eq!(skipped, 2);
    }

    #[test]
    fn normalize_list_is_stable_and_sorted() {
        let list = vec![
            "z/train.parquet".to_string(),
            "a/train.parquet".to_string(),
            "m/train.parquet".to_string(),
        ];
        assert_eq!(
            normalize_list(&list),
            "a/train.parquet,m/train.parquet,z/train.parquet"
        );
    }

    #[test]
    fn shuffle_files_is_deterministic_for_same_seed() {
        let base = refs(&["a.parquet", "b.parquet", "c.parquet", "d.parquet"]);
        let mut first = base.clone();
        let mut second = base;
        shuffle_files(&mut first, Some(42));
        shuffle_files(&mut second, Some(42));
        assert_eq!(
            first.iter().map(|f| f.key.clone()).collect::<Vec<_>>(),
            second.iter().map(|f| f.key.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn slice_for_offset_and_budget_skips_and_truncates() {
        let mut offset = 2u64;
        let mut budget = Some(3u64);
        let (out, partial) = slice_for_offset_and_budget(vec![1, 2, 3, 4, 5, 6], &mut offset, &mut budget);
        assert_eq!(out, vec![3, 4, 5]);
        assert!(partial);
        assert_eq!(offset, 0);
    }

    #[test]
    fn slice_for_offset_and_budget_skips_whole_file_when_needed() {
        let mut offset = 10u64;
        let mut budget = Some(5u64);
        let (out, partial) = slice_for_offset_and_budget(vec![1, 2, 3], &mut offset, &mut budget);
        assert!(out.is_empty());
        assert!(!partial);
        assert_eq!(offset, 7);
        assert_eq!(budget, Some(5));
    }
}
