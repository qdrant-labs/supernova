pub mod config;
pub mod engine;
pub mod plan;
pub mod rate;
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
use stores::{CollectionSchema, Point, PointId, StoreError, VectorStore};

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
        0,
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
///
/// With `resume` (the `--continue` flag), first find where a previous run of
/// this same slice stopped — a binary search probing the store for each
/// file's first point id — and skip the files already fully loaded. Safe to
/// pass unconditionally: against a fresh collection every probe misses and
/// the whole slice loads; on a finished worker only the final file is redone
/// (upserts are idempotent). Requires the same corpus and `--num-jobs` as the
/// interrupted run, and a deterministic `id_expression`.
pub async fn load(config: LoadConfig, partition: Partition, resume: bool) -> Result<(), LoadError> {
    let store = config.vectorstore.connect().await?;
    let all_files = config.datasource.list_files().await?;

    let skip_files = if resume {
        let slice = plan::partition(&all_files, partition);
        let start = resume_start(store.as_ref(), &config.datasource, &slice).await?;
        if start > 0 {
            tracing::info!(
                "--continue: resuming at file {start}/{} of this slice (`{}`); {start} file(s) \
                 probe as loaded and are skipped — the boundary file itself is re-upserted, \
                 since the previous run may have died mid-file (idempotent)",
                slice.len(),
                slice[start].key,
            );
        } else {
            tracing::info!("--continue: no prior progress found for this slice; loading all of it");
        }
        start
    } else {
        0
    };

    let n = load_files(
        store.as_ref(),
        &config.datasource,
        &config.vectors,
        &config.loader,
        all_files,
        partition,
        skip_files,
    )
    .await?;
    tracing::info!("worker {}/{} done: {n} points", partition.rank, partition.num_jobs);
    Ok(())
}

/// Where a previous run of this worker's slice stopped: the index of the
/// first file to (re)load. Files complete strictly in order within a worker
/// (see `load_files`), so the loaded files form a prefix of the slice and a
/// binary search over "does this file's first point exist in the store" finds
/// the boundary in ~log2(slice) probes.
///
/// Returns the boundary *inclusive*: the last loaded-looking file is
/// re-uploaded rather than skipped, because batches within a file land out of
/// order — the file in flight at the moment of death can have its first
/// batch stored while later ones are missing. One file of redundant,
/// idempotent work buys the no-gaps guarantee.
async fn resume_start(
    store: &dyn VectorStore,
    datasource: &DataSourceConfig,
    files: &[FileRef],
) -> Result<usize, LoadError> {
    let id_expression = datasource.reader().id_expression.clone();
    let probe = async |i: usize| -> Result<bool, LoadError> {
        let file = &files[i];
        let loaded = match first_point_id(datasource, file, &id_expression).await? {
            Some(id) => store.point_exists(&id).await?,
            // An empty file has no first row to probe; calling it "not loaded"
            // errs early (re-scanning it is free — it has no points).
            None => false,
        };
        tracing::info!(
            "--continue probe: file[{i}] `{}` → {}",
            file.key,
            if loaded { "loaded" } else { "not loaded" },
        );
        Ok(loaded)
    };
    Ok(bisect_last_loaded(files.len(), probe).await?.unwrap_or(0))
}

/// The id of `file`'s first row (row 0), or `None` for an empty file. Tries
/// the no-IO virtual evaluation first — id expressions over only `filename` /
/// `file_row_number` (the default `vf_point_id`) never touch the file — and
/// falls back to downloading the file and reading a single row when the
/// expression references real data columns.
async fn first_point_id(
    datasource: &DataSourceConfig,
    file: &FileRef,
    id_expression: &str,
) -> Result<Option<PointId>, LoadError> {
    let (key, expr) = (file.key.clone(), id_expression.to_string());
    let virtual_id = tokio::task::spawn_blocking(move || engine::eval_virtual_id(&key, 0, &expr))
        .await?;
    if let Ok(id) = virtual_id {
        return Ok(Some(id));
    }

    let local = datasource.fetch(file).await?;
    let read_job = engine::ReadJob {
        path: local.path().to_path_buf(),
        filename: local.source.key.clone(),
        vectors: HashMap::new(), // id only — no vectors, no payload
        payload: HashMap::new(),
        id_expression: id_expression.to_string(),
        shard_key_expr: None,
        limit: Some(1),
    };
    let points = tokio::task::spawn_blocking(move || read_job.run()).await??;
    Ok(points.into_iter().next().map(|p| p.id))
}

/// Find the greatest index in `0..n` whose probe reports "loaded", assuming
/// the loaded indexes form a prefix (`probe` is monotone true→false). `None`
/// when index 0 already misses. Costs at most `2 + log2(n)` probes.
///
/// A stray miss inside the loaded prefix (a file the original run *skipped*
/// after exhausting its retries) leaves the search landing on some
/// loaded-probing index at or before the true boundary. Everything from the
/// result onward gets (re)loaded, so nothing the original run had loaded or
/// hadn't reached is ever skipped; a file the original run skipped stays
/// skipped — the same outcome (and warning) the original run already
/// reported.
async fn bisect_last_loaded<F>(n: usize, mut probe: F) -> Result<Option<usize>, LoadError>
where
    F: AsyncFnMut(usize) -> Result<bool, LoadError>,
{
    if n == 0 || !probe(0).await? {
        return Ok(None);
    }
    if probe(n - 1).await? {
        return Ok(Some(n - 1));
    }
    // Invariant: probe(lo) == true, probe(hi) == false.
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if probe(mid).await? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(Some(lo))
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
    if let Some(rate) = config.loader.max_points_per_sec {
        println!("rate limit:  {}", describe_rate_limit(rate, partition.num_jobs));
    }
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
        shard_key_expr: None, // nor a shard key
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
    shard_key_expr: Option<&str>,
) -> Result<Vec<Point>, LoadError> {
    let local = datasource.fetch(file).await?;
    let read_job = engine::ReadJob {
        path: local.path().to_path_buf(),
        filename: local.source.key.clone(),
        vectors: vectors.clone(),
        payload: payload.clone(),
        id_expression: id_expression.to_string(),
        shard_key_expr: shard_key_expr.map(str::to_string),
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
    skip_files: usize,
) -> Result<u64, LoadError> {
    let batch_size = loader.batch_size.max(1);
    let concurrency = loader.concurrency.max(1);
    let look_ahead = loader.file_look_ahead.max(1);
    // Per-worker upsert ceiling (see `rate`). Every attempt draws tokens, so
    // retries can't push the store past the cap either.
    let limiter = loader.max_points_per_sec.map(|r| rate::RateLimiter::new(r, batch_size));
    if let Some(r) = loader.max_points_per_sec {
        tracing::info!("rate limit: {}", describe_rate_limit(r, partition.num_jobs));
    }
    let cap = loader.max_points_per_sec.map(|r| format!(" (cap {r})")).unwrap_or_default();
    let file_retries = loader.file_retries;
    let upsert_retries = loader.upsert_retries;
    let max_failed_files = loader.max_failed_files;
    let id_expression = datasource.reader().id_expression.clone();
    let payload = datasource.reader().payload_fields.clone();
    // Custom sharding lives on the store (the vectorstore config was consumed
    // by `connect`), but the key is computed by the reader — so pull the
    // expression back off the store and hand it to every read job.
    let shard_key_expr = store.custom_sharding().map(|c| c.shard_key.clone());
    let sharding = shard_key_expr.is_some();

    let total_files = all_files.len();
    let mut files = plan::partition(&all_files, partition);
    // `--continue` resume: the first `skip_files` of this slice probed as
    // already loaded (see `resume_start`).
    files.drain(..skip_files.min(files.len()));
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
            let shard_key_expr = &shard_key_expr;
            async move {
                // Retry the whole download+read with exponential backoff; a
                // re-fetch also recovers from a truncated/corrupt prior download.
                let mut attempt = 0u32;
                loop {
                    match fetch_and_read(
                        datasource,
                        file,
                        vectors,
                        payload,
                        id_expression,
                        shard_key_expr.as_deref(),
                    )
                    .await
                    {
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
    let limiter = &limiter;
    let cap = &cap;

    let mut skipped = 0u64;
    let mut fragmentation_warned = false;
    while let Some(outcome) = reads.next().await {
        let mut points = match outcome {
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

        // Under custom sharding an upsert's shard_key_selector scopes the whole
        // request, so a chunk must never mix keys: make the file's points
        // key-contiguous (stable sort keeps row order within a key), then let
        // `shard_chunks` split at key boundaries as well as `batch_size`.
        if sharding {
            points.sort_by(|a, b| a.shard_key.cmp(&b.shard_key));
        }
        let chunks = shard_chunks(&points, batch_size);
        if sharding && !fragmentation_warned && chunks.len() > 1 {
            let mean = points.len() / chunks.len();
            if mean < batch_size / 8 {
                fragmentation_warned = true;
                tracing::warn!(
                    "shard key fragments upserts: a file split into {} requests averaging {mean} \
                     points (batch_size is {batch_size}); a high-cardinality key interleaved \
                     within files costs throughput — consider partitioning/sorting the source \
                     files by the shard key",
                    chunks.len(),
                );
            }
        }

        // Upsert this file's batches with up to `concurrency` requests in flight.
        // `buffer_unordered` is the idiomatic bounded-concurrency primitive — a
        // semaphore over a stream of upsert futures — and lets each future borrow
        // `store` without `Arc`/spawning. Each landed batch refreshes the rate.
        futures::stream::iter(chunks)
            .map(|chunk| async move {
                let n = chunk.len() as u64;
                // Retry transient store errors with backoff; a persistent failure
                // (store down / misconfigured) aborts the run via `?` on try_collect.
                let mut attempt = 0u32;
                loop {
                    if let Some(limiter) = limiter {
                        limiter.acquire(chunk.len()).await;
                    }
                    match store.upsert_batch(chunk.to_vec()).await {
                        Ok(()) => break,
                        Err(err) => {
                            // A bad request / auth / missing collection can't
                            // succeed on retry — don't burn the budget (and
                            // don't keep re-sending the same points).
                            if !err.is_retryable() {
                                tracing::error!("upsert batch failed with a non-retryable error: {err}");
                                return Err(err);
                            }
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
                        progress.set_message(format!("{done} pts · {rate:.0} pts/s{cap}"));
                    } else {
                        tracing::info!("{done} pts · {rate:.0} pts/s{cap}");
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

/// Split `points` into upsert chunks that are at most `batch_size` long AND
/// never mix shard keys (a Qdrant upsert's `shard_key_selector` scopes the
/// whole request). Callers sort by key first so each key yields contiguous,
/// maximally-full chunks. With no shard keys (custom sharding off) this is
/// exactly `points.chunks(batch_size)`.
fn shard_chunks(points: &[Point], batch_size: usize) -> Vec<&[Point]> {
    let mut chunks = Vec::with_capacity(points.len() / batch_size.max(1) + 1);
    let mut start = 0;
    while start < points.len() {
        let key = &points[start].shard_key;
        let len = points[start..]
            .iter()
            .take(batch_size)
            .take_while(|p| p.shard_key == *key)
            .count();
        chunks.push(&points[start..start + len]);
        start += len;
    }
    chunks
}

/// The rate cap as configured (per worker) plus what that means for the
/// whole fleet — so both the "this process" and "the cluster sees" mental
/// models get their number without changing the knob's semantics.
fn describe_rate_limit(per_worker: u64, num_jobs: usize) -> String {
    if num_jobs > 1 {
        format!(
            "{per_worker} pts/s per worker (× {num_jobs} workers = {} pts/s fleet-wide)",
            per_worker as u128 * num_jobs as u128
        )
    } else {
        format!("{per_worker} pts/s")
    }
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
    use stores::ShardKeyValue;

    fn point(shard_key: Option<ShardKeyValue>) -> Point {
        Point {
            id: stores::PointId::Integer(0),
            vectors: HashMap::new(),
            payload: serde_json::Map::new(),
            shard_key,
        }
    }

    fn keyed(keys: &[&str]) -> Vec<Point> {
        keys.iter().map(|k| point(Some(ShardKeyValue::Keyword(k.to_string())))).collect()
    }

    /// No shard keys → identical to `chunks(batch_size)`.
    #[test]
    fn shard_chunks_without_keys_is_plain_chunking() {
        let points: Vec<Point> = (0..7).map(|_| point(None)).collect();
        let chunks = shard_chunks(&points, 3);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [3, 3, 1]);
    }

    /// Chunks split at key boundaries as well as batch_size, cover every point,
    /// and never mix keys.
    #[test]
    fn shard_chunks_split_at_key_boundaries_and_batch_size() {
        // sorted by key, as load_files guarantees: a×5, b×2, c×1
        let points = keyed(&["a", "a", "a", "a", "a", "b", "b", "c"]);
        let chunks = shard_chunks(&points, 3);
        assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [3, 2, 2, 1]);
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, points.len());
        for chunk in &chunks {
            let first = &chunk[0].shard_key;
            assert!(chunk.iter().all(|p| p.shard_key == *first), "chunk mixes shard keys");
        }
    }

    async fn bisect(states: &[bool]) -> Option<usize> {
        bisect_last_loaded(states.len(), async |i| Ok(states[i])).await.unwrap()
    }

    /// Every prefix length resolves to its exact boundary (resume-inclusive).
    #[tokio::test]
    async fn bisect_finds_the_loaded_prefix_boundary() {
        assert_eq!(bisect(&[]).await, None);
        assert_eq!(bisect(&[false]).await, None);
        assert_eq!(bisect(&[true]).await, Some(0));
        assert_eq!(bisect(&[true, false]).await, Some(0));
        assert_eq!(bisect(&[true, true, false, false, false]).await, Some(1));
        for k in 0..=33 {
            let states: Vec<bool> = (0..33).map(|i| i < k).collect();
            let want = if k == 0 { None } else { Some(k - 1) };
            assert_eq!(bisect(&states).await, want, "prefix length {k}");
        }
    }

    /// A miss poked into the prefix (a file the original run skipped after
    /// retries) must leave the result on a loaded-probing index at or before
    /// the true boundary — never past it.
    #[tokio::test]
    async fn bisect_with_skipped_file_stays_at_or_before_the_boundary() {
        let states = [true, false, true, true, false, false];
        let got = bisect(&states).await.expect("prefix starts loaded");
        assert!(got <= 3, "resume index {got} is past the true boundary");
        assert!(states[got], "resume index {got} must probe as loaded");
    }

    #[tokio::test]
    async fn bisect_probe_count_is_logarithmic() {
        use std::cell::Cell;
        let (n, boundary) = (1024usize, 700usize); // loaded prefix 0..700
        let calls = Cell::new(0usize);
        let got = bisect_last_loaded(n, async |i| {
            calls.set(calls.get() + 1);
            Ok(i < boundary)
        })
        .await
        .unwrap();
        assert_eq!(got, Some(boundary - 1));
        assert!(calls.get() <= 2 + n.ilog2() as usize, "{} probes for n={n}", calls.get());
    }

    #[test]
    fn rate_limit_description_shows_fleet_math_only_for_fleets() {
        assert_eq!(describe_rate_limit(5000, 1), "5000 pts/s");
        assert_eq!(
            describe_rate_limit(5000, 32),
            "5000 pts/s per worker (× 32 workers = 160000 pts/s fleet-wide)"
        );
    }

    /// Numbers sort before keywords (enum variant order) — the exact order is
    /// unimportant, but sorting must make equal keys contiguous for
    /// `shard_chunks` even when keyword and number keys mix.
    #[test]
    fn sorted_mixed_keys_group_contiguously() {
        let mut points = vec![
            point(Some(ShardKeyValue::Keyword("a".into()))),
            point(Some(ShardKeyValue::Number(2))),
            point(Some(ShardKeyValue::Keyword("a".into()))),
            point(Some(ShardKeyValue::Number(2))),
        ];
        points.sort_by(|a, b| a.shard_key.cmp(&b.shard_key));
        let chunks = shard_chunks(&points, 10);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
    }
}
