//! Python bindings: `rust_fetch.Client`. Every transfer runs with the GIL
//! released, writing straight into the caller's buffer.

use std::sync::Arc;
use std::time::Duration;

use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::{Dest, FetchConfig, FetchError, Fetcher, ReadOpts, S3Settings, Span};
use crate::dense::{self, DenseError};

create_exception!(
    rust_fetch,
    UnsupportedLayout,
    PyValueError,
    "The parquet file is valid, but its dense column is not a layout the native decoder handles exactly (fall back to pyarrow)."
);
create_exception!(
    rust_fetch,
    CorruptParquet,
    PyOSError,
    "The native decoder found bytes that contradict the parquet footer or page headers."
);

fn dense_to_py(e: DenseError) -> PyErr {
    match e {
        DenseError::Unsupported(m) => UnsupportedLayout::new_err(m),
        DenseError::Corrupt(m) => CorruptParquet::new_err(m),
        // The file was rewritten mid-read: a per-file fallback (pyarrow
        // reads the current version), not a transport failure.
        DenseError::Fetch(e @ FetchError::Changed(_)) => UnsupportedLayout::new_err(e.to_string()),
        DenseError::Fetch(f) => to_py(f),
    }
}

/// Footer-derived plan for decoding one file's dense column; from
/// `Client.dense_plan`. Read-only.
#[pyclass(frozen, module = "rust_fetch")]
struct DensePlan {
    p: Arc<dense::DensePlan>,
}

#[pymethods]
impl DensePlan {
    #[getter]
    fn url(&self) -> &str {
        &self.p.url
    }
    #[getter]
    fn column(&self) -> &str {
        &self.p.column
    }
    #[getter]
    fn n_rows(&self) -> u64 {
        self.p.n_rows
    }
    #[getter]
    fn dim(&self) -> u64 {
        self.p.dim
    }
    /// "float32" or "float16" -- the stored width, copied bit-for-bit.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.p.dtype()
    }
    /// Bytes of the decoded [n_rows, dim] output.
    #[getter]
    fn nbytes(&self) -> u64 {
        self.p.nbytes()
    }
    /// Bytes fetched from storage (this column's chunks only).
    #[getter]
    fn column_bytes(&self) -> u64 {
        self.p.column_bytes()
    }
    #[getter]
    fn num_row_groups(&self) -> usize {
        self.p.chunks.len()
    }
    /// ETag of the object version this plan (and every read made with it)
    /// is pinned to; None if the store reports none.
    #[getter]
    fn etag(&self) -> Option<String> {
        self.p.etag.clone()
    }
    #[getter]
    fn codec(&self) -> &'static str {
        self.p.codec.name()
    }
    fn __repr__(&self) -> String {
        format!(
            "DensePlan(url={:?}, column={:?}, n_rows={}, dim={}, dtype={}, codec={}, row_groups={})",
            self.p.url, self.p.column, self.p.n_rows, self.p.dim, self.p.dtype(), self.p.codec.name(), self.p.chunks.len()
        )
    }
}

fn is_main_thread(py: Python<'_>) -> bool {
    (|| -> PyResult<bool> {
        let threading = py.import("threading")?;
        let cur = threading.call_method0("current_thread")?;
        let main = threading.call_method0("main_thread")?;
        Ok(cur.is(&main))
    })()
    .unwrap_or(false)
}

fn to_py(e: FetchError) -> PyErr {
    match e {
        FetchError::Invalid(m) => PyValueError::new_err(m),
        FetchError::NotFound(m) => PyFileNotFoundError::new_err(m),
        FetchError::Io(m) => PyOSError::new_err(m),
        e @ FetchError::Changed(_) => PyOSError::new_err(e.to_string()),
        FetchError::Cancelled => PyOSError::new_err("cancelled"),
    }
}

/// Borrow `obj` as a writable C-contiguous byte buffer. The returned PyBuffer
/// must outlive every use of the Dest (it pins the exporter's memory).
fn writable_dest(obj: &Bound<'_, PyAny>) -> PyResult<(PyBuffer<u8>, Dest)> {
    let buf = PyBuffer::<u8>::get(obj).map_err(|e| {
        PyTypeError::new_err(format!(
            "dest must expose a byte buffer (numpy uint8 array, bytearray, mmap, memoryview of those; \
             use arr.view(np.uint8) for other dtypes): {e}"
        ))
    })?;
    if buf.readonly() {
        return Err(PyValueError::new_err("dest buffer is read-only"));
    }
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("dest buffer must be C-contiguous"));
    }
    let dest = Dest { addr: buf.buf_ptr() as usize, len: buf.len_bytes() as u64 };
    Ok((buf, dest))
}

/// Long-lived native downloader. Thread-safe: share one per process.
///
/// Client(region=None, concurrency=64, part_size=16 MiB, connect_timeout_s=5,
///        part_timeout_s=60, retries=5, endpoint=None, allow_http=None, threads=None)
///
/// Credentials/region/endpoint come from the standard AWS_* env vars or the
/// instance profile; `region`/`endpoint`/`allow_http` override them. With no
/// region anywhere (and no custom endpoint) the bucket's region is resolved.
/// `concurrency` caps in-flight GETs across ALL calls on this client.
#[pyclass(frozen, module = "rust_fetch")]
struct Client {
    // ManuallyDrop: see `Drop for Client`.
    rt: std::mem::ManuallyDrop<tokio::runtime::Runtime>,
    f: Arc<Fetcher>,
    pid: u32,
    /// Set by `cancel()`: every in-flight and future call on this client
    /// stops (still joining its tasks) and raises OSError("cancelled").
    cancelled: Arc<AtomicBool>,
}

impl Drop for Client {
    fn drop(&mut self) {
        // A Client inherited across fork() has a runtime whose worker
        // threads do not exist in the child: dropping it there waits on
        // them forever (e.g. at interpreter exit). Leak it instead.
        if std::process::id() == self.pid {
            // SAFETY: dropped exactly once, here; never used again.
            unsafe { std::mem::ManuallyDrop::drop(&mut self.rt) };
        }
    }
}

impl Client {
    /// Run `fut` on the runtime with the GIL released, polling Python's
    /// signal handlers every 100 ms. On a signal (Ctrl-C) the shared `stop`
    /// flag is raised so no new work starts, the future is STILL awaited to
    /// completion (every task joined -- nothing may write into the caller's
    /// buffer afterwards), and then the signal's exception is raised.
    fn wait<R, F>(&self, py: Python<'_>, stop: Arc<AtomicBool>, fut: F) -> PyResult<R>
    where
        F: std::future::Future<Output = R> + Send,
        R: Send,
    {
        let cancelled = self.cancelled.clone();
        if cancelled.load(Ordering::Relaxed) {
            stop.store(true, Ordering::Relaxed);
        }
        // Signal handlers only ever run on the main thread (PyErr_CheckSignals
        // is a no-op elsewhere), so only the main thread polls -- other
        // threads' transfers never wait on the GIL.
        let on_main = is_main_thread(py);
        let (r, sig) = py.allow_threads(|| {
            self.rt.block_on(async move {
                tokio::pin!(fut);
                let mut sig: Option<PyErr> = None;
                let mut tick = tokio::time::interval(Duration::from_millis(50));
                tick.tick().await;
                loop {
                    tokio::select! {
                        r = &mut fut => return (r, sig),
                        _ = tick.tick() => {
                            if cancelled.load(Ordering::Relaxed) {
                                stop.store(true, Ordering::Relaxed);
                            }
                            if on_main && sig.is_none() {
                                // A handler can panic (PanicException, or a
                                // nested rust_fetch call in a handler). A
                                // panic unwinding out of this block_on would
                                // drop the in-flight future WITHOUT joining
                                // its tasks -- catch it and treat it as a
                                // signal: stop, keep awaiting, then raise.
                                let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    Python::with_gil(|py| py.check_signals())
                                }));
                                let err = match polled {
                                    Ok(Ok(())) => None,
                                    Ok(Err(e)) => Some(e),
                                    Err(_) => Some(PyRuntimeError::new_err(
                                        "a Python signal handler panicked during a rust_fetch call",
                                    )),
                                };
                                if let Some(e) = err {
                                    sig = Some(e);
                                    stop.store(true, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            })
        });
        match sig {
            Some(e) => Err(e),
            None => Ok(r),
        }
    }

    /// Read `spans`; when that takes more than one GET, HEAD first and pin
    /// every part to that ETag, so the result can never mix two versions of
    /// an object rewritten mid-read (a single GET is atomic already).
    fn pinned_read(
        &self,
        py: Python<'_>,
        url: &str,
        spans: &[Span],
        d: Dest,
        known: Option<Option<String>>,
    ) -> PyResult<u64> {
        let ps = self.f.cfg.part_size;
        let parts: u64 = spans.iter().map(|s| s.len.div_ceil(ps)).sum();
        let stop = Arc::new(AtomicBool::new(false));
        let st = stop.clone();
        let f = self.f.clone();
        let fut = async move {
            // `known`: the ETag of a HEAD the caller already made (e.g. for
            // the size) -- pin to THAT version, even for a single part.
            let if_match = if let Some(e) = known {
                e
            } else if parts > 1 {
                tokio::select! {
                    r = f.head(url) => r?.1,
                    _ = crate::core::stopped(&st) => return Err(FetchError::Cancelled),
                }
            } else {
                None
            };
            f.read_spans_opts(url, spans, d, ReadOpts { stop: st, if_match }).await
        };
        self.wait(py, stop, fut)?.map_err(to_py)
    }

    fn check_pid(&self) -> PyResult<()> {
        if std::process::id() != self.pid {
            return Err(PyRuntimeError::new_err(
                "rust_fetch.Client was created in another process (before fork); create a new Client",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (region=None, concurrency=64, part_size=16<<20, connect_timeout_s=5.0,
                        part_timeout_s=60.0, retries=5, endpoint=None, allow_http=None, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        region: Option<String>,
        concurrency: usize,
        part_size: u64,
        connect_timeout_s: f64,
        part_timeout_s: f64,
        retries: u32,
        endpoint: Option<String>,
        allow_http: Option<bool>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let secs = |name: &str, v: f64| -> PyResult<Duration> {
            if !(v >= 0.001 && v <= 86_400.0) {
                return Err(PyValueError::new_err(format!("{name} must be in [0.001, 86400] seconds, got {v}")));
            }
            Duration::try_from_secs_f64(v).map_err(|e| PyValueError::new_err(format!("{name}: {e}")))
        };
        let (connect_timeout, part_timeout) =
            (secs("connect_timeout_s", connect_timeout_s)?, secs("part_timeout_s", part_timeout_s)?);
        if !(1..=65_536).contains(&concurrency) {
            return Err(PyValueError::new_err(format!("concurrency must be in [1, 65536], got {concurrency}")));
        }
        if !(1..=(16u64 << 30)).contains(&part_size) {
            return Err(PyValueError::new_err(format!("part_size must be in [1, 16 GiB], got {part_size}")));
        }
        if retries > 100 {
            return Err(PyValueError::new_err(format!("retries must be <= 100, got {retries}")));
        }
        let threads = threads
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8));
        if !(1..=1024).contains(&threads) {
            return Err(PyValueError::new_err(format!("threads must be in [1, 1024], got {threads}")));
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("rust-fetch")
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
        let cfg = FetchConfig {
            concurrency,
            part_size,
            retries,
            part_timeout,
            connect_timeout,
            ..Default::default()
        };
        let f = Fetcher::new(cfg, S3Settings { region, endpoint, allow_http }).map_err(to_py)?;
        Ok(Self {
            rt: std::mem::ManuallyDrop::new(rt),
            f: Arc::new(f),
            pid: std::process::id(),
            cancelled: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Stop every in-flight call on this client (and refuse new ones) as
    /// fast as safely possible: S3 parts are cancelled at once, local reads
    /// finish their current pread, every task is joined, and each call
    /// raises OSError("cancelled"). Irreversible. Callable from any thread.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// (size, etag) from a HEAD; etag may be None if the store has none.
    fn head(&self, py: Python<'_>, url: String) -> PyResult<(u64, Option<String>)> {
        self.check_pid()?;
        let stop = Arc::new(AtomicBool::new(false));
        let st = stop.clone();
        self.wait(py, stop, async move {
            tokio::select! {
                r = self.f.head(&url) => r,
                _ = crate::core::stopped(&st) => Err(FetchError::Cancelled),
            }
        })?
        .map_err(to_py)
    }

    /// Object size in bytes (HEAD).
    fn size(&self, py: Python<'_>, url: String) -> PyResult<u64> {
        self.check_pid()?;
        let stop = Arc::new(AtomicBool::new(false));
        let st = stop.clone();
        self.wait(py, stop, async move {
            tokio::select! {
                r = self.f.size(&url) => r,
                _ = crate::core::stopped(&st) => Err(FetchError::Cancelled),
            }
        })?
        .map_err(to_py)
    }

    /// Download bytes [offset, offset+length) of `url` into the start of
    /// `dest`. length=None means "to the end of the object" (costs a HEAD).
    /// Returns the number of bytes written. Raises on any failure; on failure
    /// `dest` contents are unspecified.
    #[pyo3(signature = (url, dest, offset=0, length=None))]
    fn read_into(
        &self,
        py: Python<'_>,
        url: String,
        dest: &Bound<'_, PyAny>,
        offset: u64,
        length: Option<u64>,
    ) -> PyResult<u64> {
        self.check_pid()?;
        let (buf, d) = writable_dest(dest)?;
        let mut known: Option<Option<String>> = None;
        let length = match length {
            Some(n) => n,
            None => {
                let stop = Arc::new(AtomicBool::new(false));
                let st = stop.clone();
                let (size, etag) = self
                    .wait(py, stop, async {
                        tokio::select! {
                            r = self.f.head(&url) => r,
                            _ = crate::core::stopped(&st) => Err(FetchError::Cancelled),
                        }
                    })?
                    .map_err(to_py)?;
                // The length came from this version: read exactly it.
                known = Some(etag);
                if offset > size {
                    return Err(PyValueError::new_err(format!(
                        "offset {offset} is past the end of {url} ({size} bytes)"
                    )));
                }
                size - offset
            }
        };
        if length > d.len {
            return Err(PyValueError::new_err(format!(
                "dest too small: need {length} bytes, buffer has {}",
                d.len
            )));
        }
        let spans = [Span { src: offset, len: length, dst: 0 }];
        let r = self.pinned_read(py, &url, &spans, d, known);
        drop(buf);
        r
    }

    /// Projected read: each (offset, length, dest_offset) copies object bytes
    /// [offset, offset+length) to dest[dest_offset : dest_offset+length].
    /// Destination ranges must not overlap. Returns total bytes written.
    /// `if_match`: an ETag (e.g. from `head`) every GET must match -- pins
    /// this read, even a single-GET one, to that object version (raises
    /// OSError "object changed during the read" otherwise).
    #[pyo3(signature = (url, dest, ranges, if_match=None))]
    fn read_ranges_into(
        &self,
        py: Python<'_>,
        url: String,
        dest: &Bound<'_, PyAny>,
        ranges: Vec<(u64, u64, u64)>,
        if_match: Option<String>,
    ) -> PyResult<u64> {
        self.check_pid()?;
        let (buf, d) = writable_dest(dest)?;
        let spans: Vec<Span> = ranges.into_iter().map(|(src, len, dst)| Span { src, len, dst }).collect();
        let r = self.pinned_read(py, &url, &spans, d, if_match.map(Some));
        drop(buf);
        r
    }

    /// Read the footer of `url` and plan decoding of top-level list column
    /// `column` (one GET for the file tail; `size` skips the HEAD). Raises
    /// UnsupportedLayout if the column is not a layout the native decoder
    /// reproduces exactly.
    #[pyo3(signature = (url, column, size=None))]
    fn dense_plan(&self, py: Python<'_>, url: String, column: String, size: Option<u64>) -> PyResult<DensePlan> {
        self.check_pid()?;
        let stop = Arc::new(AtomicBool::new(false));
        let p = self
            .wait(py, stop.clone(), dense::plan_dense(&self.f, &url, &column, size, stop))?
            .map_err(dense_to_py)?;
        Ok(DensePlan { p: Arc::new(p) })
    }

    /// Fetch `plan`'s column chunks and decode them into `dest` as a
    /// row-major [n_rows, dim] array of plan.dtype (dest: writable byte
    /// buffer of at least plan.nbytes bytes, e.g. arr.reshape(-1).view(np.uint8)).
    /// Up to `chunks_in_flight` row groups download/decode at once. Returns
    /// bytes written. On any error dest contents are unspecified.
    #[pyo3(signature = (plan, dest, chunks_in_flight=8))]
    fn read_dense_into(
        &self,
        py: Python<'_>,
        plan: &DensePlan,
        dest: &Bound<'_, PyAny>,
        chunks_in_flight: usize,
    ) -> PyResult<u64> {
        self.check_pid()?;
        let (buf, d) = writable_dest(dest)?;
        let (f, p) = (self.f.clone(), plan.p.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let r = self.wait(py, stop.clone(), dense::read_dense(f, p, d, chunks_in_flight, stop));
        drop(buf);
        r?.map_err(dense_to_py)
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.f.cfg.concurrency
    }

    #[getter]
    fn part_size(&self) -> u64 {
        self.f.cfg.part_size
    }

    fn __repr__(&self) -> String {
        format!(
            "rust_fetch.Client(concurrency={}, part_size={}, retries={}, region={:?}, endpoint={:?})",
            self.f.cfg.concurrency, self.f.cfg.part_size, self.f.cfg.retries, self.f.s3.region, self.f.s3.endpoint
        )
    }
}

#[pymodule]
fn _rust_fetch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<DensePlan>()?;
    m.add("UnsupportedLayout", m.py().get_type::<UnsupportedLayout>())?;
    m.add("CorruptParquet", m.py().get_type::<CorruptParquet>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
