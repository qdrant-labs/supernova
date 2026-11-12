//! Pure-Rust core: split byte ranges into parts, fetch them concurrently with
//! retries, and write every part straight into its slice of one destination
//! buffer. No Python here -- `python.rs` wraps this and releases the GIL.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use object_store::aws::{resolve_bucket_region, AmazonS3Builder};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    ClientOptions, GetOptions, GetRange, GetResultPayload, ObjectStore, RetryConfig,
};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("{0}")]
    Invalid(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Io(String),
    /// The object no longer matches the version the read started from
    /// (ETag precondition failed): it was overwritten mid-read.
    #[error("object changed during the read: {0}")]
    Changed(String),
    /// Stopped by the caller (a Python signal) before finishing.
    #[error("cancelled")]
    Cancelled,
}

/// Per-call options shared by every part of one read.
#[derive(Debug, Clone, Default)]
pub struct ReadOpts {
    /// Set by the caller to stop issuing work (checked before every attempt,
    /// after every permit acquire, and before spawning new parts). Parts
    /// already writing still finish -- everything is joined before return.
    pub stop: Arc<AtomicBool>,
    /// ETag every part must match (HTTP If-Match / local equivalent), so a
    /// multi-part read can never stitch two versions of an object together.
    pub if_match: Option<String>,
}

pub type Result<T> = std::result::Result<T, FetchError>;

/// A parsed object location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    S3 { bucket: String, key: String },
    File { path: String },
}

impl Location {
    pub fn parse(url: &str) -> Result<Self> {
        if let Some(rest) = url.strip_prefix("s3://").or_else(|| url.strip_prefix("s3a://")) {
            let (bucket, key) = rest
                .split_once('/')
                .ok_or_else(|| FetchError::Invalid(format!("{url}: expected s3://bucket/key")))?;
            if bucket.is_empty() || key.is_empty() {
                return Err(FetchError::Invalid(format!("{url}: expected s3://bucket/key")));
            }
            // object_store's Path normalises away empty segments, so a key
            // like "/x" or "a//b" would silently read a DIFFERENT object.
            if key.starts_with('/') || key.ends_with('/') || key.contains("//") {
                return Err(FetchError::Invalid(format!("{url}: empty path segment in key")));
            }
            Ok(Location::S3 { bucket: bucket.to_string(), key: key.to_string() })
        } else if let Some(rest) = url.strip_prefix("file://") {
            if !rest.starts_with('/') {
                return Err(FetchError::Invalid(format!("{url}: file:// URLs must be absolute")));
            }
            Ok(Location::File { path: rest.to_string() })
        } else if url.starts_with('/') {
            Ok(Location::File { path: url.to_string() })
        } else {
            Err(FetchError::Invalid(format!(
                "{url}: unsupported URL (expected s3://bucket/key, file:///abs/path or /abs/path)"
            )))
        }
    }
}

/// Settings for the S3 backend; `None` means "from the environment".
#[derive(Debug, Clone, Default)]
pub struct S3Settings {
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub allow_http: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// Max in-flight GETs across ALL calls on this fetcher.
    pub concurrency: usize,
    pub part_size: u64,
    /// Extra attempts per part after the first.
    pub retries: u32,
    /// Wall-clock cap on one attempt of one part (request + body).
    pub part_timeout: Duration,
    pub connect_timeout: Duration,
    pub backoff_base: Duration,
    pub backoff_max: Duration,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            concurrency: 64,
            part_size: 16 << 20,
            retries: 5,
            part_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(5),
            backoff_base: Duration::from_millis(100),
            backoff_max: Duration::from_secs(5),
        }
    }
}

/// One contiguous request: `len` bytes from object offset `src` into
/// destination offset `dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub src: u64,
    pub len: u64,
    pub dst: u64,
}

/// Raw destination. The caller guarantees the memory stays alive and is not
/// otherwise written for the duration of `fetch_spans` (python.rs holds a
/// PyBuffer view across the call; the Rust tests hold a Vec).
#[derive(Debug, Clone, Copy)]
pub struct Dest {
    pub addr: usize,
    pub len: u64,
}

/// Validate spans against the destination and split them into parts of at
/// most `part_size`. Rejects out-of-bounds and overlapping destination ranges
/// (overlap would be a data race between concurrent part writers).
pub fn plan_parts(spans: &[Span], dest_len: u64, part_size: u64) -> Result<Vec<Span>> {
    if part_size == 0 {
        return Err(FetchError::Invalid("part_size must be > 0".into()));
    }
    let mut sorted: Vec<Span> = spans.iter().copied().filter(|s| s.len > 0).collect();
    sorted.sort_by_key(|s| s.dst);
    let mut prev_end = 0u64;
    for s in &sorted {
        let end = s.dst.checked_add(s.len).ok_or_else(|| FetchError::Invalid("range overflow".into()))?;
        s.src.checked_add(s.len).ok_or_else(|| FetchError::Invalid("range overflow".into()))?;
        if end > dest_len {
            return Err(FetchError::Invalid(format!(
                "destination too small: need {end} bytes, buffer has {dest_len}"
            )));
        }
        if s.dst < prev_end {
            return Err(FetchError::Invalid(format!(
                "destination ranges overlap at dest offset {}",
                s.dst
            )));
        }
        prev_end = end;
    }
    let mut parts = Vec::new();
    // Keep caller order (usually ascending source offset) for issue order.
    for s in spans.iter().filter(|s| s.len > 0) {
        let mut off = 0;
        while off < s.len {
            let n = part_size.min(s.len - off);
            parts.push(Span { src: s.src + off, len: n, dst: s.dst + off });
            off += n;
        }
    }
    Ok(parts)
}

#[derive(Debug)]
enum PartErr {
    Fatal(FetchError),
    Retryable(String),
}

fn classify(e: object_store::Error) -> PartErr {
    use object_store::Error as E;
    match e {
        E::NotFound { path, .. } => PartErr::Fatal(FetchError::NotFound(path)),
        E::Precondition { .. } => PartErr::Fatal(FetchError::Changed(e.to_string())),
        E::PermissionDenied { .. } | E::Unauthenticated { .. }
        | E::NotSupported { .. } | E::InvalidPath { .. } => {
            PartErr::Fatal(FetchError::Io(e.to_string()))
        }
        other => {
            // Skip retries for invalid ranges (HTTP 416), checking error
            // messages at their source to avoid false matches.
            let mut unsatisfiable = false;
            let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(&other);
            while let Some(e) = cur {
                let m = e.to_string();
                if m.starts_with("Server returned non-2xx status code: 416")
                    || m.starts_with("Requested range was invalid")
                {
                    unsatisfiable = true;
                    break;
                }
                // A local permission error never clears up on retry.
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::PermissionDenied)
                {
                    unsatisfiable = true;
                    break;
                }
                cur = e.source();
            }
            if unsatisfiable {
                PartErr::Fatal(FetchError::Io(other.to_string()))
            } else {
                PartErr::Retryable(other.to_string())
            }
        }
    }
}

/// Resolves once `stop` is raised (polled; the flag is a plain atomic).
pub async fn stopped(stop: &AtomicBool) {
    while !stop.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Handle for a local read that must complete before its buffer is released,
/// even if the attempt times out.
type LocalRead = tokio::task::JoinHandle<std::io::Result<()>>;

async fn fetch_part_once(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
    part: Span,
    dest: Dest,
    if_match: Option<String>,
) -> std::result::Result<Option<LocalRead>, PartErr> {
    let want_etag = if_match.clone();
    let opts = GetOptions {
        range: Some(GetRange::Bounded(part.src..part.src + part.len)),
        if_match,
        ..Default::default()
    };
    let res = store.get_opts(path, opts).await.map_err(classify)?;
    if let (Some(want), Some(got)) = (&want_etag, &res.meta.e_tag) {
        if want != got {
            return Err(PartErr::Fatal(FetchError::Changed(format!("etag {got} != {want}"))));
        }
    }
    let base = dest.addr + part.dst as usize;
    match res.payload {
        // Local files: pread straight into the destination slice (object_store's
        // own stream would copy through 8 KiB Vecs, one spawn_blocking each).
        GetResultPayload::File(file, _) => {
            let len = part.len as usize;
            let src = part.src;
            Ok(Some(tokio::task::spawn_blocking(move || {
                use std::os::unix::fs::FileExt;
                // SAFETY: [base, base+len) lies inside dest (plan_parts checked)
                // and is written by this part only.
                let slice = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, len) };
                file.read_exact_at(slice, src)
            })))
        }
        GetResultPayload::Stream(mut stream) => {
            // Writes happen synchronously between awaits, so cancelling this
            // future (timeout) can never leave a write in flight.
            let mut written: u64 = 0;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(classify)?;
                let n = chunk.len() as u64;
                if written + n > part.len {
                    return Err(PartErr::Retryable(format!(
                        "server returned more bytes than requested for [{}, {})",
                        part.src,
                        part.src + part.len
                    )));
                }
                // SAFETY: bounds checked above; this part owns this slice.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        chunk.as_ptr(),
                        (base + written as usize) as *mut u8,
                        chunk.len(),
                    );
                }
                written += n;
            }
            if written != part.len {
                // S3 truncates a range that runs past EOF -- a short body that
                // completed cleanly means the object is too small: don't retry.
                return Err(PartErr::Fatal(FetchError::Io(format!(
                    "object shorter than requested range [{}, {}): got {} of {} bytes",
                    part.src,
                    part.src + part.len,
                    written,
                    part.len
                ))));
            }
            Ok(None)
        }
    }
}

async fn finish_local(h: LocalRead, part: Span) -> std::result::Result<(), PartErr> {
    h.await
        .map_err(|e| PartErr::Retryable(format!("read task failed: {e}")))?
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                PartErr::Fatal(FetchError::Io(format!(
                    "object shorter than requested range [{}, {})",
                    part.src,
                    part.src + part.len
                )))
            } else if e.kind() == std::io::ErrorKind::PermissionDenied {
                PartErr::Fatal(FetchError::Io(e.to_string()))
            } else {
                PartErr::Retryable(e.to_string())
            }
        })
}

fn jitter(max: Duration) -> Duration {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    let m = max.as_nanos().max(1) as u64;
    Duration::from_nanos((nanos as u64).wrapping_mul(2654435761) % m)
}

async fn fetch_part(
    store: Arc<dyn ObjectStore>,
    path: Path,
    part: Span,
    dest: Dest,
    cfg: Arc<FetchConfig>,
    sem: Arc<Semaphore>,
    opts: ReadOpts,
) -> Result<()> {
    let stop = &opts.stop;
    let mut last = String::new();
    for attempt in 0..=cfg.retries {
        if stop.load(Ordering::Relaxed) {
            return Err(FetchError::Cancelled);
        }
        if attempt > 0 {
            let exp = cfg.backoff_base.saturating_mul(1u32 << (attempt - 1).min(16));
            let d = exp.min(cfg.backoff_max);
            tokio::select! {
                _ = tokio::time::sleep(d / 2 + jitter(d / 2 + Duration::from_nanos(1))) => {}
                _ = stopped(stop) => return Err(FetchError::Cancelled),
            }
        }
        let outcome = {
            let _permit = tokio::select! {
                p = sem.acquire() => p.map_err(|_| FetchError::Io("fetcher closed".into()))?,
                _ = stopped(stop) => return Err(FetchError::Cancelled),
            };
            // Re-check after waiting for a slot: parts queued behind a
            // failure must not each go on to issue a GET.
            if stop.load(Ordering::Relaxed) {
                return Err(FetchError::Cancelled);
            }
            // Race the attempt against the stop flag. Cancelling is safe at
            // any await point: a stream part writes only synchronously
            // between awaits, and a local part spawns its pread and returns
            // the handle within one poll -- that handle is then awaited
            // below, outside the race, to completion.
            let attempt = tokio::time::timeout(
                cfg.part_timeout,
                fetch_part_once(&store, &path, part, dest, opts.if_match.clone()),
            );
            let attempt = tokio::select! {
                r = attempt => r,
                _ = stopped(stop) => return Err(FetchError::Cancelled),
            };
            match attempt {
                // The local read is awaited to completion, never abandoned.
                Ok(Ok(Some(local))) => Ok(finish_local(local, part).await),
                Ok(Ok(None)) => Ok(Ok(())),
                Ok(Err(e)) => Ok(Err(e)),
                Err(elapsed) => Err(elapsed),
            }
        };
        match outcome {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(PartErr::Fatal(e))) => return Err(e),
            Ok(Err(PartErr::Retryable(msg))) => last = msg,
            Err(_) => last = format!("attempt timed out after {:?}", cfg.part_timeout),
        }
    }
    Err(FetchError::Io(format!(
        "bytes [{}, {}) failed after {} attempts: {}",
        part.src,
        part.src + part.len,
        cfg.retries + 1,
        last
    )))
}

/// Fetches all spans into `dest`, returning bytes written.
///
/// Joins all tasks before returning, including on error, to prevent
/// writes after `dest` is released. A shared flag cancels further attempts.
pub async fn fetch_spans(
    store: Arc<dyn ObjectStore>,
    path: Path,
    spans: &[Span],
    dest: Dest,
    cfg: Arc<FetchConfig>,
    sem: Arc<Semaphore>,
    opts: ReadOpts,
) -> Result<u64> {
    let parts = plan_parts(spans, dest.len, cfg.part_size)?;
    let total: u64 = parts.iter().map(|p| p.len).sum();
    let stop = opts.stop.clone();
    // Spawn lazily: a few more tasks than can run at once, so memory tracks
    // `concurrency`, not the number of parts.
    let window = cfg.concurrency.saturating_mul(2).max(1);
    let mut pending = parts.into_iter();
    let mut set = JoinSet::new();
    let mut first_err: Option<FetchError> = None;
    loop {
        while set.len() < window && !stop.load(Ordering::Relaxed) {
            let Some(part) = pending.next() else { break };
            set.spawn(fetch_part(store.clone(), path.clone(), part, dest, cfg.clone(), sem.clone(), opts.clone()));
        }
        let Some(joined) = set.join_next().await else { break };
        let r = joined.unwrap_or_else(|e| Err(FetchError::Io(format!("part task panicked: {e}"))));
        if let Err(e) = r {
            // A Cancelled part only echoes an earlier error / caller stop.
            if first_err.is_none() && !matches!(e, FetchError::Cancelled) {
                first_err = Some(e);
            }
            stop.store(true, Ordering::Relaxed);
        }
    }
    match first_err {
        Some(e) => Err(e),
        None if stop.load(Ordering::Relaxed) || pending.next().is_some() => Err(FetchError::Cancelled),
        None => Ok(total),
    }
}

/// Long-lived fetcher: config, a shared GET semaphore, and a cache of object
/// stores (one per bucket; each owns a reqwest connection pool).
pub struct Fetcher {
    pub cfg: Arc<FetchConfig>,
    pub s3: S3Settings,
    sem: Arc<Semaphore>,
    stores: Mutex<HashMap<String, Arc<dyn ObjectStore>>>,
}

impl Fetcher {
    pub fn new(cfg: FetchConfig, s3: S3Settings) -> Result<Self> {
        if cfg.concurrency == 0 {
            return Err(FetchError::Invalid("concurrency must be > 0".into()));
        }
        if cfg.part_size == 0 {
            return Err(FetchError::Invalid("part_size must be > 0".into()));
        }
        Ok(Self {
            sem: Arc::new(Semaphore::new(cfg.concurrency)),
            cfg: Arc::new(cfg),
            s3,
            stores: Mutex::new(HashMap::new()),
        })
    }

    fn client_options(&self) -> ClientOptions {
        let mut o = ClientOptions::new()
            .with_connect_timeout(self.cfg.connect_timeout)
            // The per-attempt deadline is enforced in fetch_part; keep reqwest's
            // own total timeout out of the way (it would cap large parts).
            .with_timeout(self.cfg.part_timeout + Duration::from_secs(5))
            .with_pool_idle_timeout(Duration::from_secs(60))
            .with_pool_max_idle_per_host(self.cfg.concurrency.max(1) * 2);
        if let Some(h) = self.s3.allow_http {
            o = o.with_allow_http(h);
        }
        o
    }

    async fn s3_store(&self, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let key = format!("s3://{bucket}");
        if let Some(s) = self.stores.lock().unwrap().get(&key) {
            return Ok(s.clone());
        }
        let mut b = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .with_client_options(self.client_options())
            // Retries are ours (per part, covering the body too).
            .with_retry(RetryConfig { max_retries: 0, ..Default::default() });
        if let Some(e) = &self.s3.endpoint {
            b = b.with_endpoint(e);
        }
        if let Some(h) = self.s3.allow_http {
            b = b.with_allow_http(h);
        }
        let env_region = std::env::var("AWS_REGION").ok().or_else(|| std::env::var("AWS_DEFAULT_REGION").ok());
        let env_endpoint = std::env::var("AWS_ENDPOINT_URL").ok().or_else(|| std::env::var("AWS_ENDPOINT").ok());
        if let Some(r) = &self.s3.region {
            b = b.with_region(r);
        } else if env_region.is_none() && self.s3.endpoint.is_none() && env_endpoint.is_none() {
            // No region anywhere: ask S3 where the bucket lives instead of
            // silently defaulting to us-east-1 (which 301s for other regions).
            let r = resolve_bucket_region(bucket, &self.client_options())
                .await
                .map_err(|e| FetchError::Io(format!("resolving region of bucket {bucket}: {e}")))?;
            b = b.with_region(r);
        }
        let store: Arc<dyn ObjectStore> =
            Arc::new(b.build().map_err(|e| FetchError::Invalid(format!("S3 config: {e}")))?);
        self.stores.lock().unwrap().insert(key, store.clone());
        Ok(store)
    }

    fn local_store(&self) -> Arc<dyn ObjectStore> {
        let mut m = self.stores.lock().unwrap();
        m.entry("file://".into()).or_insert_with(|| Arc::new(LocalFileSystem::new())).clone()
    }

    pub async fn resolve(&self, url: &str) -> Result<(Arc<dyn ObjectStore>, Path)> {
        match Location::parse(url)? {
            Location::S3 { bucket, key } => {
                let path = Path::parse(&key).map_err(|e| FetchError::Invalid(format!("{url}: {e}")))?;
                Ok((self.s3_store(&bucket).await?, path))
            }
            Location::File { path } => {
                let p = Path::from_absolute_path(&path)
                    .map_err(|e| FetchError::Invalid(format!("{url}: {e}")))?;
                Ok((self.local_store(), p))
            }
        }
    }

    pub async fn size(&self, url: &str) -> Result<u64> {
        Ok(self.head(url).await?.0)
    }

    /// (size, ETag) from a HEAD, with the same retry policy as parts.
    pub async fn head(&self, url: &str) -> Result<(u64, Option<String>)> {
        let (store, path) = self.resolve(url).await?;
        let mut last = String::new();
        for attempt in 0..=self.cfg.retries {
            if attempt > 0 {
                tokio::time::sleep(self.cfg.backoff_base.saturating_mul(1u32 << (attempt - 1).min(16)).min(self.cfg.backoff_max)).await;
            }
            match tokio::time::timeout(self.cfg.part_timeout, store.head(&path)).await {
                Ok(Ok(meta)) => return Ok((meta.size, meta.e_tag)),
                Ok(Err(e)) => match classify(e) {
                    PartErr::Fatal(FetchError::NotFound(_)) => return Err(FetchError::NotFound(url.to_string())),
                    PartErr::Fatal(e) => return Err(FetchError::Io(format!("{url}: {e}"))),
                    PartErr::Retryable(m) => last = m,
                },
                Err(_) => last = "HEAD timed out".into(),
            }
        }
        Err(FetchError::Io(format!("{url}: HEAD failed after {} attempts: {last}", self.cfg.retries + 1)))
    }

    pub async fn read_spans(&self, url: &str, spans: &[Span], dest: Dest) -> Result<u64> {
        self.read_spans_opts(url, spans, dest, ReadOpts::default()).await
    }

    pub async fn read_spans_opts(&self, url: &str, spans: &[Span], dest: Dest, opts: ReadOpts) -> Result<u64> {
        // Resolution may look the bucket's region up over the network:
        // raced with stop (dropping it is safe -- nothing written yet).
        let (store, path) = tokio::select! {
            r = self.resolve(url) => r?,
            _ = stopped(&opts.stop) => return Err(FetchError::Cancelled),
        };
        fetch_spans(store, path, spans, dest, self.cfg.clone(), self.sem.clone(), opts)
            .await
            .map_err(|e| match e {
                FetchError::NotFound(_) => FetchError::NotFound(url.to_string()),
                FetchError::Io(m) => FetchError::Io(format!("{url}: {m}")),
                other => other,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::PutPayload;

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect()
    }

    async fn mem_store(bytes: &[u8]) -> (Arc<dyn ObjectStore>, Path) {
        let s: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let p = Path::from("a/b.bin");
        s.put(&p, PutPayload::from(bytes.to_vec())).await.unwrap();
        (s, p)
    }

    fn cfg(part: u64, conc: usize) -> (Arc<FetchConfig>, Arc<Semaphore>) {
        let c = FetchConfig { part_size: part, concurrency: conc, retries: 1, ..Default::default() };
        (Arc::new(c), Arc::new(Semaphore::new(conc)))
    }

    #[test]
    fn parse_urls() {
        assert_eq!(
            Location::parse("s3://b/k/x.parquet").unwrap(),
            Location::S3 { bucket: "b".into(), key: "k/x.parquet".into() }
        );
        assert_eq!(Location::parse("file:///tmp/x").unwrap(), Location::File { path: "/tmp/x".into() });
        assert_eq!(Location::parse("/tmp/x").unwrap(), Location::File { path: "/tmp/x".into() });
        assert!(Location::parse("s3://b").is_err());
        assert!(Location::parse("s3://b/").is_err());
        assert!(Location::parse("gs://b/k").is_err());
        assert!(Location::parse("rel/path").is_err());
    }

    #[test]
    fn plan_splits_and_validates() {
        let p = plan_parts(&[Span { src: 10, len: 25, dst: 0 }], 25, 10).unwrap();
        assert_eq!(
            p,
            vec![
                Span { src: 10, len: 10, dst: 0 },
                Span { src: 20, len: 10, dst: 10 },
                Span { src: 30, len: 5, dst: 20 }
            ]
        );
        assert!(plan_parts(&[Span { src: 0, len: 26, dst: 0 }], 25, 10).is_err());
        assert!(plan_parts(
            &[Span { src: 0, len: 10, dst: 0 }, Span { src: 50, len: 10, dst: 5 }],
            100,
            4
        )
        .is_err());
        assert!(plan_parts(&[Span { src: 0, len: 1, dst: 0 }], 1, 0).is_err());
        assert!(plan_parts(&[Span { src: 0, len: 0, dst: 100 }], 1, 4).unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn full_and_partial_reads_are_byte_exact() {
        let src = data(1_000_003);
        let (store, path) = mem_store(&src).await;
        for &(part, conc) in &[(1u64, 3usize), (7, 1), (4096, 8), (65536, 64), (1 << 20, 2), (5_000_000, 4)] {
            if part == 1 && src.len() > 100_000 {
                // 1-byte parts over 1 MB is just slow; exercise them on a slice.
                let mut out = vec![0u8; 1000];
                let (c, s) = cfg(part, conc);
                let n = fetch_spans(store.clone(), path.clone(), &[Span { src: 777, len: 1000, dst: 0 }],
                    Dest { addr: out.as_mut_ptr() as usize, len: 1000 }, c, s, ReadOpts::default()).await.unwrap();
                assert_eq!(n, 1000);
                assert_eq!(&out[..], &src[777..1777]);
                continue;
            }
            let mut out = vec![0u8; src.len()];
            let (c, s) = cfg(part, conc);
            let n = fetch_spans(store.clone(), path.clone(), &[Span { src: 0, len: src.len() as u64, dst: 0 }],
                Dest { addr: out.as_mut_ptr() as usize, len: out.len() as u64 }, c.clone(), s.clone(), ReadOpts::default()).await.unwrap();
            assert_eq!(n as usize, src.len());
            assert_eq!(out, src, "part={part} conc={conc}");
            // Multiple spans scattered into one buffer.
            let spans = [
                Span { src: 999_000, len: 1003, dst: 0 },
                Span { src: 3, len: 70_000, dst: 2000 },
                Span { src: 500_000, len: 1, dst: 1003 },
            ];
            let mut out = vec![0xAAu8; 80_000];
            fetch_spans(store.clone(), path.clone(), &spans,
                Dest { addr: out.as_mut_ptr() as usize, len: out.len() as u64 }, c, s, ReadOpts::default()).await.unwrap();
            assert_eq!(&out[..1003], &src[999_000..]);
            assert_eq!(out[1003], src[500_000]);
            assert!(out[1004..2000].iter().all(|&b| b == 0xAA));
            assert_eq!(&out[2000..72_000], &src[3..70_003]);
            assert!(out[72_000..].iter().all(|&b| b == 0xAA));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn past_eof_and_missing_are_errors() {
        let src = data(1000);
        let (store, path) = mem_store(&src).await;
        let (c, s) = cfg(300, 4);
        let mut out = vec![0u8; 2000];
        let e = fetch_spans(store.clone(), path.clone(), &[Span { src: 0, len: 1500, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 2000 }, c.clone(), s.clone(), ReadOpts::default()).await;
        assert!(e.is_err(), "reading past EOF must fail, got {e:?}");
        let e = fetch_spans(store.clone(), Path::from("nope"), &[Span { src: 0, len: 10, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 2000 }, c, s, ReadOpts::default()).await;
        assert!(matches!(e, Err(FetchError::NotFound(_))), "{e:?}");
    }

    /// Fails the first `fail_first` GETs outright, then the next
    /// `break_streams` GETs mid-body (half the bytes, then an error), then
    /// behaves -- exercises both request-level and body-level retries.
    #[derive(Debug)]
    struct Flaky {
        inner: InMemory,
        fail_first: std::sync::atomic::AtomicUsize,
        break_streams: std::sync::atomic::AtomicUsize,
        delay_first: std::sync::atomic::AtomicUsize,
    }

    impl std::fmt::Display for Flaky {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Flaky")
        }
    }

    fn take(a: &std::sync::atomic::AtomicUsize) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        a.fetch_update(SeqCst, SeqCst, |v| v.checked_sub(1)).is_ok()
    }

    #[async_trait::async_trait]
    impl ObjectStore for Flaky {
        async fn put_opts(&self, l: &Path, p: PutPayload, o: object_store::PutOptions) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(l, p, o).await
        }
        async fn put_multipart_opts(&self, l: &Path, o: object_store::PutMultipartOptions) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(l, o).await
        }
        async fn get_opts(&self, l: &Path, o: GetOptions) -> object_store::Result<object_store::GetResult> {
            if take(&self.delay_first) {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
            if take(&self.fail_first) {
                return Err(object_store::Error::Generic { store: "flaky", source: "injected 503".into() });
            }
            let mut r = self.inner.get_opts(l, o).await?;
            if take(&self.break_streams) {
                let GetResultPayload::Stream(s) = r.payload else { unreachable!() };
                let body: Vec<bytes::Bytes> = s.map(|c| c.unwrap()).collect().await;
                let all: Vec<u8> = body.concat();
                let half = bytes::Bytes::from(all[..all.len() / 2].to_vec());
                let items: Vec<object_store::Result<bytes::Bytes>> = vec![
                    Ok(half),
                    Err(object_store::Error::Generic { store: "flaky", source: "connection reset".into() }),
                ];
                r.payload = GetResultPayload::Stream(futures::stream::iter(items).boxed());
            }
            Ok(r)
        }
        async fn delete(&self, l: &Path) -> object_store::Result<()> { self.inner.delete(l).await }
        fn list(&self, p: Option<&Path>) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> { self.inner.list(p) }
        async fn list_with_delimiter(&self, p: Option<&Path>) -> object_store::Result<object_store::ListResult> { self.inner.list_with_delimiter(p).await }
        async fn copy(&self, a: &Path, b: &Path) -> object_store::Result<()> { self.inner.copy(a, b).await }
        async fn copy_if_not_exists(&self, a: &Path, b: &Path) -> object_store::Result<()> { self.inner.copy_if_not_exists(a, b).await }
    }

    fn flaky(fail: usize, brk: usize, delay: usize) -> Flaky {
        use std::sync::atomic::AtomicUsize;
        Flaky { inner: InMemory::new(), fail_first: AtomicUsize::new(fail), break_streams: AtomicUsize::new(brk), delay_first: AtomicUsize::new(delay) }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retries_request_and_body_failures_and_stalls() {
        let src = data(100_000);
        let f = flaky(0, 0, 0);
        let p = Path::from("x");
        f.inner.put(&p, PutPayload::from(src.clone())).await.unwrap();
        f.fail_first.store(5, std::sync::atomic::Ordering::SeqCst);
        f.break_streams.store(5, std::sync::atomic::Ordering::SeqCst);
        f.delay_first.store(2, std::sync::atomic::Ordering::SeqCst);
        let store: Arc<dyn ObjectStore> = Arc::new(f);
        let c = Arc::new(FetchConfig {
            part_size: 10_000, concurrency: 4, retries: 12,
            part_timeout: Duration::from_millis(300), backoff_base: Duration::from_millis(1),
            ..Default::default()
        });
        let mut out = vec![0u8; src.len()];
        let n = fetch_spans(store, p, &[Span { src: 0, len: src.len() as u64, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: out.len() as u64 }, c, Arc::new(Semaphore::new(4)), ReadOpts::default()).await.unwrap();
        assert_eq!(n as usize, src.len());
        assert_eq!(out, src);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gives_up_after_retries_with_clear_error() {
        let f = flaky(1000, 0, 0);
        let p = Path::from("x");
        f.inner.put(&p, PutPayload::from(data(1000))).await.unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(f);
        let c = Arc::new(FetchConfig { part_size: 100, concurrency: 2, retries: 2, backoff_base: Duration::from_millis(1), ..Default::default() });
        let mut out = vec![0u8; 1000];
        let e = fetch_spans(store, p, &[Span { src: 0, len: 1000, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 1000 }, c, Arc::new(Semaphore::new(2)), ReadOpts::default()).await.unwrap_err();
        let m = e.to_string();
        assert!(m.contains("failed after 3 attempts") && m.contains("injected 503"), "{m}");
    }

    /// Counts GETs; the first one fails with a non-retryable error.
    #[derive(Debug)]
    struct DenyFirst {
        inner: InMemory,
        gets: std::sync::atomic::AtomicUsize,
    }

    impl std::fmt::Display for DenyFirst {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "DenyFirst")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for DenyFirst {
        async fn put_opts(&self, l: &Path, p: PutPayload, o: object_store::PutOptions) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(l, p, o).await
        }
        async fn put_multipart_opts(&self, l: &Path, o: object_store::PutMultipartOptions) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(l, o).await
        }
        async fn get_opts(&self, l: &Path, o: GetOptions) -> object_store::Result<object_store::GetResult> {
            if self.gets.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return Err(object_store::Error::PermissionDenied { path: l.to_string(), source: "denied".into() });
            }
            self.inner.get_opts(l, o).await
        }
        async fn delete(&self, l: &Path) -> object_store::Result<()> { self.inner.delete(l).await }
        fn list(&self, p: Option<&Path>) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> { self.inner.list(p) }
        async fn list_with_delimiter(&self, p: Option<&Path>) -> object_store::Result<object_store::ListResult> { self.inner.list_with_delimiter(p).await }
        async fn copy(&self, a: &Path, b: &Path) -> object_store::Result<()> { self.inner.copy(a, b).await }
        async fn copy_if_not_exists(&self, a: &Path, b: &Path) -> object_store::Result<()> { self.inner.copy_if_not_exists(a, b).await }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parts_queued_behind_a_fatal_error_issue_no_gets() {
        let store = Arc::new(DenyFirst { inner: InMemory::new(), gets: Default::default() });
        let p = Path::from("x");
        store.inner.put(&p, PutPayload::from(vec![7u8; 100 * 16])).await.unwrap();
        let cfg = Arc::new(FetchConfig { concurrency: 1, part_size: 16, retries: 3, ..Default::default() });
        let mut out = vec![0u8; 100 * 16];
        let e = fetch_spans(store.clone(), p, &[Span { src: 0, len: 1600, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 1600 }, cfg, Arc::new(Semaphore::new(1)),
            ReadOpts::default()).await.unwrap_err();
        assert!(matches!(e, FetchError::Io(_)), "{e:?}");
        let gets = store.gets.load(std::sync::atomic::Ordering::SeqCst);
        assert!(gets <= 2, "{gets} GETs issued after a fatal error on the first");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn etag_mismatch_is_changed_not_retried() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let p = Path::from("x");
        store.put(&p, PutPayload::from(vec![1u8; 64])).await.unwrap();
        let cfg = Arc::new(FetchConfig { part_size: 16, ..Default::default() });
        let mut out = vec![0u8; 64];
        let opts = ReadOpts { if_match: Some("\"not-the-etag\"".into()), ..Default::default() };
        let e = fetch_spans(store.clone(), p.clone(), &[Span { src: 0, len: 64, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 64 }, cfg.clone(), Arc::new(Semaphore::new(4)), opts)
            .await.unwrap_err();
        assert!(matches!(e, FetchError::Changed(_)), "{e:?}");
        let etag = store.head(&p).await.unwrap().e_tag;
        let ok = ReadOpts { if_match: etag, ..Default::default() };
        fetch_spans(store, p, &[Span { src: 0, len: 64, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 64 }, cfg, Arc::new(Semaphore::new(4)), ok)
            .await.unwrap();
        assert_eq!(out, vec![1u8; 64]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn caller_stop_returns_cancelled_and_issues_nothing() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let p = Path::from("x");
        store.put(&p, PutPayload::from(vec![1u8; 64])).await.unwrap();
        let opts = ReadOpts::default();
        opts.stop.store(true, Ordering::Relaxed);
        let mut out = vec![0u8; 64];
        let e = fetch_spans(store, p, &[Span { src: 0, len: 64, dst: 0 }],
            Dest { addr: out.as_mut_ptr() as usize, len: 64 }, Arc::new(FetchConfig::default()),
            Arc::new(Semaphore::new(4)), opts).await.unwrap_err();
        assert!(matches!(e, FetchError::Cancelled), "{e:?}");
    }

}
