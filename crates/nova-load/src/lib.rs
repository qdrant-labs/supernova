pub mod config;
pub mod engine;
pub mod sources;
pub mod stores;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::{StreamExt, TryStreamExt};
use indicatif::{ProgressBar, ProgressStyle};

use config::LoadConfig;
use sources::DataSource;
use stores::{CollectionSchema, Point, StoreError};

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
/// Inspect what a load *would* do without connecting to the store or
/// downloading anything: dump the resolved config (secrets redacted) and list
/// the source files. Listing does hit the source (e.g. an S3 `ListObjects`),
/// since enumerating files is the whole point.
pub async fn dry_run(config: LoadConfig, num_jobs: usize, job_rank: usize) -> Result<(), LoadError> {
    println!("== nova-load dry run ==\n");
    println!("partition:   job_rank={job_rank} num_jobs={num_jobs}");
    if num_jobs > 1 {
        println!("             (note: file partitioning is not yet applied during load)");
    }
    println!("loader:      {:?}", config.loader);
    println!("vectorstore: {:?}", config.vectorstore);
    println!("datasource:  {:?}", config.datasource);
    println!("vectors:     {:?}", config.vectors);

    let files = config.datasource.list_files().await?;
    let total: u64 = files.iter().filter_map(|f| f.size).sum();
    println!("\nfiles: {} found, {} total", files.len(), human_size(total));
    for f in &files {
        let size = f.size.map(human_size).unwrap_or_else(|| "?".to_string());
        println!("  {} ({size})", f.key);
    }
    Ok(())
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

/// Files are processed one at a time; within a file, batches upsert with
/// bounded concurrency (`loader.concurrency`). File-level partitioning for
/// distributed runs is still a TODO.
pub async fn run_loader(config: LoadConfig) -> Result<(), LoadError> {
    let LoadConfig { datasource, vectorstore, vectors, loader } = config;
    let batch_size = loader.batch_size.max(1);
    let concurrency = loader.concurrency.max(1);
    let look_ahead = loader.file_look_ahead.max(1);
    let id_expression = datasource.reader().id_expression.clone();
    let payload = datasource.reader().payload_fields.clone();

    let store = vectorstore.connect().await?;
    let files = datasource.list_files().await?;
    if files.is_empty() {
        tracing::warn!("no files to load");
        return Ok(());
    }
    tracing::info!("loading {} file(s) into {store}", files.len());

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

    // Aggregate counters. `points_done` is the running total; `rate_window`
    // holds the last sample (time, count) for an instantaneous rate, vs. the
    // cumulative average reported at the end.
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
                    vectors: vectors.clone(),
                    payload: payload.clone(),
                    id_expression: id_expression.clone(),
                };
                let points = tokio::task::spawn_blocking(move || read_job.run()).await??;
                drop(local); // read done; delete the temp download
                Ok::<Vec<Point>, LoadError>(points)
            }
        })
        .buffered(look_ahead);
    tokio::pin!(reads);

    let store = &store;
    let progress = &progress;
    let points_done = &points_done;
    let rate_window = &rate_window;

    let mut first = true;
    while let Some(points) = reads.next().await {
        let points = points?;

        // Create the collection from the first file's inferred dimensions, then
        // turn off the indexing optimizer for the bulk load.
        if first {
            let dims = engine::infer_dims(&points, &vectors);
            let schema = CollectionSchema { vectors: vectors.clone(), dims };
            store.ensure_collection(&schema).await?;
            store.defer_indexing().await?;
            first = false;
        }

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

    // Re-enable indexing and wait for it to settle before declaring done.
    tracing::info!("re-enabling indexing…");
    store.enable_indexing().await?;
    store.wait_for_indexing().await?;
    store.close().await?;
    tracing::info!("done: {total} points loaded");
    Ok(())
}
