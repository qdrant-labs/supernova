//! Orchestration: stream a [`DataReader`] into a [`VectorStore`].
//!
//! A blocking producer task drives the (synchronous) reader and forwards
//! upsert-sized batches into a bounded channel; a pool of async workers, capped
//! by a semaphore, upserts them concurrently. The bounded channel is the
//! backpressure valve — when all workers are busy the channel fills and the
//! producer parks on `blocking_send`, so memory stays bounded and the achieved
//! rate reflects what the store can actually sustain.

use std::collections::HashMap;
use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};
use nova_metrics::MetricsSink;
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::JoinSet;

use crate::config::{LoaderConfig, VectorSpec, resolve_schema};
use crate::errors::{LoadError, ReaderError, StoreError};
use crate::sources::DataReader;
use crate::stores::{Point, VectorStore};

/// How often the sampler emits the cumulative count + velocity. Off the hot
/// path (workers just bump an atomic), so this only sets the plot resolution.
const METRICS_INTERVAL: Duration = Duration::from_secs(1);

/// Outcome of a load, for the caller to report.
#[derive(Debug, Clone, Copy)]
pub struct LoadStats {
    pub total: u64,
    pub loaded: u64,
    pub errors: u64,
    pub elapsed_secs: f64,
}

/// Stream `reader` into `store`.
///
/// When `manage_indexing` is true the runner owns the collection lifecycle:
/// create it, defer indexing for the bulk load, then re-enable and wait for the
/// index to build. Distributed workers pass `false` so only the master does it.
///
/// Per-batch upsert errors are counted, not fatal (mirroring the Python loader);
/// a reader failure or a lifecycle failure aborts the load.
pub async fn run_loader(
    reader: Box<dyn DataReader>,
    store: Arc<dyn VectorStore>,
    vectors_spec: &HashMap<String, VectorSpec>,
    cfg: &LoaderConfig,
    manage_indexing: bool,
    sink: Arc<dyn MetricsSink>,
) -> Result<LoadStats, LoadError> {
    tracing::info!("scanning corpus (counting records + reading dimensions)…");
    // Metadata reads hit DuckDB (blocking), so run them off the async runtime.
    let (dims, total, reader) = tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        let dims = reader.dimensions()?;
        let total = reader.total_count()?;
        Ok::<_, ReaderError>((dims, total, reader))
    })
    .await
    .map_err(join_panic)??;

    tracing::info!(store = %store, total, "starting load");

    if manage_indexing {
        let schema = resolve_schema(vectors_spec, &dims);
        store.ensure_collection(&schema).await?;
        tracing::info!("deferring indexing for bulk load");
        store.defer_indexing().await?;
    }

    sink.event("bulk upload started");
    let started = Instant::now();
    let (loaded, errors) = drive(reader, store.clone(), cfg, total, sink.clone()).await?;
    tracing::info!(
        loaded,
        errors,
        elapsed_secs = started.elapsed().as_secs_f64(),
        "upload finished"
    );

    if manage_indexing {
        sink.event("re-enabling indexing");
        tracing::info!("re-enabling indexing");
        store.enable_indexing().await?;
        tracing::info!("waiting for indexing to complete");
        store.wait_for_indexing().await?;
        sink.event("indexing complete");
    }
    store.close().await?;

    Ok(LoadStats {
        total,
        loaded,
        errors,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

/// Control-plane only: probe vector dimensions, create/verify the collection,
/// and defer indexing for the bulk load — then exit. The distributed controller
/// runs this once (`nova-load --setup-only`); workers then load with
/// `--no-manage-indexing`, so the collection is created exactly once with the
/// configured params.
pub async fn setup_collection(
    reader: Box<dyn DataReader>,
    store: Arc<dyn VectorStore>,
    vectors_spec: &HashMap<String, VectorSpec>,
) -> Result<(), LoadError> {
    // The reader's dimension probe hits DuckDB (blocking) — keep it off the runtime.
    let dims = tokio::task::spawn_blocking(move || {
        let mut reader = reader;
        reader.dimensions()
    })
    .await
    .map_err(join_panic)??;

    let schema = resolve_schema(vectors_spec, &dims);
    store.ensure_collection(&schema).await?;
    store.defer_indexing().await?;
    store.close().await?;
    Ok(())
}

/// Control-plane only: re-enable indexing and block until the build completes —
/// then exit. The distributed controller runs this (`nova-load --finalize`)
/// after every worker has finished loading.
pub async fn finalize(store: Arc<dyn VectorStore>) -> Result<(), LoadError> {
    store.enable_indexing().await?;
    store.wait_for_indexing().await?;
    store.close().await?;
    Ok(())
}

/// The producer/consumer core: returns (loaded, errors).
async fn drive(
    reader: Box<dyn DataReader>,
    store: Arc<dyn VectorStore>,
    cfg: &LoaderConfig,
    total: u64,
    sink: Arc<dyn MetricsSink>,
) -> Result<(u64, u64), LoadError> {
    let (tx, mut rx) = mpsc::channel::<Vec<Point>>(cfg.concurrency.max(1) * 2);
    let batch_size = cfg.batch_size.max(1);
    // points/sec → per-point spacing; None or <=0 means unbounded.
    let per_point = cfg
        .wps
        .filter(|w| *w > 0.0)
        .map(|w| Duration::from_secs_f64(1.0 / w));

    // Producer: drive the blocking reader, slice chunks into batches, pace, send.
    let producer = tokio::task::spawn_blocking(move || {
        let mut next = Instant::now();
        let mut send = |batch: Vec<Point>| -> Result<(), ReaderError> {
            pace(&mut next, per_point, batch.len());
            tx.blocking_send(batch)
                .map_err(|_| ReaderError::Other("upsert channel closed early".into()))
        };
        reader.read(&mut |chunk| {
            let mut batch = Vec::with_capacity(batch_size);
            for p in chunk {
                batch.push(p);
                if batch.len() >= batch_size {
                    send(std::mem::take(&mut batch))?;
                }
            }
            if !batch.is_empty() {
                send(batch)?;
            }
            Ok(())
        })
    });

    // Live counters: workers bump these per batch (a single atomic add —
    // negligible against an upsert round-trip), and a 1 Hz sampler turns them
    // into a cumulative `points_loaded` plus a `wps` velocity sample. All the
    // metric work happens on the sampler task, off the upsert hot path.
    let loaded_pts = Arc::new(AtomicU64::new(0));
    let errors_pts = Arc::new(AtomicU64::new(0));
    let sampler_stop = Arc::new(Notify::new());
    let sampler = tokio::spawn(sample_progress(
        sink,
        loaded_pts.clone(),
        sampler_stop.clone(),
    ));

    // Consumer: spawn a semaphore-capped upsert task per batch.
    let sem = Arc::new(Semaphore::new(cfg.concurrency.max(1)));
    let mut workers: JoinSet<(usize, Result<(), StoreError>)> = JoinSet::new();

    // Interactive terminal → live bar; otherwise (CI, redirected) periodic logs.
    let interactive = std::io::stderr().is_terminal();
    let progress = if interactive && total > 0 {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} pts ({per_sec}, eta {eta}) {msg}",
            )
            .unwrap(),
        );
        // Animate immediately so the first-chunk read (DuckDB fetching row
        // groups, slow over httpfs) shows as working rather than frozen.
        pb.set_message("reading first batch…");
        pb.enable_steady_tick(Duration::from_millis(120));
        pb
    } else {
        ProgressBar::hidden()
    };
    let step = (total / 20).max(1);
    let mut dispatched = 0u64;
    let mut next_log = step;
    let mut started = false;

    while let Some(batch) = rx.recv().await {
        if !started {
            started = true;
            progress.set_message("");
        }
        let n = batch.len() as u64;
        progress.inc(n);
        if !interactive && total > 0 {
            dispatched += n;
            if dispatched >= next_log {
                tracing::info!(dispatched, total, "loading progress");
                next_log = dispatched + step;
            }
        }
        let permit = sem.clone().acquire_owned().await.expect("semaphore open");
        let store = store.clone();
        let loaded_pts = loaded_pts.clone();
        let errors_pts = errors_pts.clone();
        workers.spawn(async move {
            let n = batch.len();
            let res = store.upsert_batch(batch).await;
            // Bump the live counter the sampler reads. Confirmed upserts only,
            // so `points_loaded` tracks what's actually in the store.
            match &res {
                Ok(()) => loaded_pts.fetch_add(n as u64, Ordering::Relaxed),
                Err(_) => errors_pts.fetch_add(n as u64, Ordering::Relaxed),
            };
            drop(permit);
            (n, res)
        });
    }
    progress.finish_and_clear();

    while let Some(joined) = workers.join_next().await {
        // A persistent upsert failure (already retried in the store) is counted
        // via the atomic above; surface it here. A panicked worker (join Err)
        // is counted toward neither, matching the prior behaviour.
        if let Ok((n, Err(e))) = joined {
            tracing::warn!(points = n, error = %e, "upsert batch failed");
        }
    }

    // Stop the sampler (it emits one final cumulative sample) and read the
    // authoritative totals from the same counters it was sampling.
    sampler_stop.notify_one();
    let _ = sampler.await;
    let loaded = loaded_pts.load(Ordering::Relaxed);
    let errors = errors_pts.load(Ordering::Relaxed);

    // Surface a reader failure now that the channel has drained.
    producer.await.map_err(join_panic)??;
    Ok((loaded, errors))
}

/// Periodically emit the cumulative loaded count and an instantaneous velocity
/// (`wps`) until signalled to stop, then emit one final cumulative sample.
/// Lives on its own task so emission cadence is decoupled from the upsert path.
async fn sample_progress(sink: Arc<dyn MetricsSink>, loaded: Arc<AtomicU64>, stop: Arc<Notify>) {
    let mut ticker = tokio::time::interval(METRICS_INTERVAL);
    let mut last = 0u64;
    let mut last_t = Instant::now();
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let total = loaded.load(Ordering::Relaxed);
                let now = Instant::now();
                let dt = now.duration_since(last_t).as_secs_f64();
                let wps = if dt > 0.0 { (total - last) as f64 / dt } else { 0.0 };
                sink.log("points_loaded", total as f64);
                sink.log("wps", wps);
                last = total;
                last_t = now;
            }
            _ = stop.notified() => {
                sink.log("points_loaded", loaded.load(Ordering::Relaxed) as f64);
                break;
            }
        }
    }
}

/// Advance the virtual schedule by one batch's worth of points and sleep to it.
/// Falling behind (the deadline is already past) admits the next batch
/// immediately, so the average tracks the target.
fn pace(next: &mut Instant, per_point: Option<Duration>, points: usize) {
    if let Some(pp) = per_point {
        *next += pp.mul_f64(points as f64);
        let now = Instant::now();
        if *next > now {
            std::thread::sleep(*next - now);
        }
    }
}

fn join_panic(_: tokio::task::JoinError) -> ReaderError {
    ReaderError::Other("reader task panicked".into())
}

#[cfg(all(test, feature = "local"))]
mod tests {
    use super::*;
    use crate::stores::Point;
    use std::sync::Mutex;

    /// Records everything upserted, so we can assert the pipeline delivered it.
    struct MockStore {
        points: Arc<Mutex<Vec<Point>>>,
    }

    impl std::fmt::Display for MockStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock")
        }
    }

    #[async_trait::async_trait]
    impl VectorStore for MockStore {
        async fn ensure_collection(
            &self,
            _schema: &crate::config::CollectionSchema,
        ) -> Result<(), StoreError> {
            Ok(())
        }
        async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
            self.points.lock().unwrap().extend(points);
            Ok(())
        }
        async fn close(&self) -> Result<(), StoreError> {
            Ok(())
        }
    }

    fn write_parquet(path: &std::path::Path, rows: usize) {
        let conn = duckdb::Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "COPY (SELECT i AS row_id, [i::FLOAT, i::FLOAT] AS dense_embedding \
             FROM range({rows}) t(i)) TO '{}' (FORMAT PARQUET)",
            path.display()
        ))
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn loads_all_points_into_the_store() {
        let dir = std::env::temp_dir().join(format!("nova_runner_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_parquet(&dir.join("data.parquet"), 1000);

        let cfg: crate::sources::local::LocalConfig =
            serde_yaml::from_str(&format!("path: {}\n", dir.join("data.parquet").display()))
                .unwrap();
        let vectors: HashMap<String, VectorSpec> =
            serde_yaml::from_str("dense:\n  type: dense\n  column: dense_embedding\n").unwrap();

        // chunk_size 250, batch_size 64, concurrency 4 → exercises slicing + fan-out.
        let reader = Box::new(cfg.into_reader(&vectors, 250));
        let collected = Arc::new(Mutex::new(Vec::new()));
        let store = Arc::new(MockStore {
            points: collected.clone(),
        });

        let loader_cfg = LoaderConfig {
            batch_size: 64,
            concurrency: 4,
            prefetch_size: None,
            wps: None,
        };

        let stats = run_loader(
            reader,
            store,
            &vectors,
            &loader_cfg,
            true,
            Arc::new(nova_metrics::NullSink),
        )
        .await
        .unwrap();

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(stats.total, 1000);
        assert_eq!(stats.loaded, 1000);
        assert_eq!(stats.errors, 0);
        assert_eq!(collected.lock().unwrap().len(), 1000);
    }
}
