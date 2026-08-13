//! Transient (time-series) reporting: one row per batch dispatch, as it lands.
//!
//! The end-of-run [`Summary`](crate::runner::Summary) collapses the time axis —
//! a latency spike at t=90s from the cluster re-optimizing is indistinguishable
//! from the same samples spread evenly. A [`Recorder`] preserves it: the
//! collector task (which already sees every [`DispatchSample`] off the hot
//! path) forwards each sample to a dedicated writer thread ([`spawn_writer`])
//! that owns the sink, so acute load behavior and cluster-adjustment transients
//! can be plotted afterwards — and so the sink's blocking writes stay off the
//! runtime's worker threads.
//!
//! Rows are deliberately raw and unopinionated — per-dispatch samples with a
//! timestamp, no bucketing or smoothing. Any aggregation (rolling percentiles,
//! 1s buckets) is derivable downstream from raw rows; the reverse is not.
//!
//! [`Recorder`] is the abstraction point: `csv` and `jsonl` sinks exist today;
//! a database sink (sqlite, …) is a new impl, not a redesign — `begin()` is
//! where it would create its tables, the same way the file sinks create their
//! file (and the CSV sink its header) there.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use serde::Deserialize;

use crate::runner::DispatchSample;

/// Bounded depth of the collector → writer-thread queue. Full means the sink
/// can't keep pace with dispatch (a slow/remote disk, `path: "-"` piped into a
/// slow consumer, a future db sink). We drop-and-count rather than either grow
/// unbounded (OOM on a multi-minute high-rps run) or block the collector (which
/// would push backpressure onto the load loop and bias the very latency numbers
/// being measured). A local-disk CSV/JSONL sink outpaces dispatch by orders of
/// magnitude, so this only fills under genuinely slow sinks.
const WRITER_QUEUE_DEPTH: usize = 8192;

/// The `report:` config section. Absent → no time-series output (just the
/// end-of-run summary, exactly as before).
///
/// ```yaml
/// report:
///   format: csv            # csv | jsonl
///   path: storm_ts.csv     # "-" = stdout
/// ```
///
/// Note on `path: "-"`: stdout otherwise carries ONLY the end-of-run summary
/// (one JSON line under `--json`) for callers like `nova sweep` to parse —
/// streaming rows into it breaks that contract, so "-" is for interactive
/// piping, not for `--json` runs.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportConfig {
    pub format: ReportFormat,
    pub path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReportFormat {
    Csv,
    Jsonl,
}

impl ReportConfig {
    /// Build the configured sink. Cheap: no I/O until [`Recorder::begin`].
    pub fn build(&self) -> Box<dyn Recorder> {
        Box::new(FileRecorder { path: self.path.clone(), format: self.format, out: None })
    }
}

/// A sink for per-dispatch samples.
///
/// Lifecycle: `begin()` once before the load starts (create the file / write
/// the CSV header / create tables — fail HERE, not mid-run, so a bad path
/// dies before load is offered, called on the caller's thread), then `record()`
/// per dispatch in arrival order and `finish()` once at the end — both on the
/// dedicated writer thread ([`spawn_writer`]), never on a tokio runtime worker.
///
/// That thread placement is deliberate: `record()` may block (a file write's
/// `write()` syscall, a db round-trip). Running it on a runtime worker would
/// pin that worker for the syscall's duration — and `#[tokio::main]` sizes the
/// worker pool to the core count, so on a 1-vCPU box that worker is the *only*
/// one and a blocking write would stall every task, dispatch completions
/// included. On its own thread the blocking wait parks in the kernel and yields
/// the CPU back to the runtime, so implementations may block freely; they need
/// `Send` (moved onto the thread) but never internal synchronization.
pub trait Recorder: Send {
    fn begin(&mut self) -> io::Result<()>;
    fn record(&mut self, sample: &DispatchSample) -> io::Result<()>;
    fn finish(&mut self) -> io::Result<()>;
}

/// Move an already-`begin()`-ed recorder onto a dedicated OS thread and return
/// the bounded sender the collector forwards samples on, plus the join handle.
///
/// The thread owns the recorder and does all blocking I/O; the collector only
/// `try_send`s (see [`WRITER_QUEUE_DEPTH`] for the drop-on-full rationale).
/// A `record()` error disables recording — the thread flushes what it has via
/// `finish()` and exits, which disconnects the channel so the collector stops
/// forwarding — but never touches the load test. The thread ends (and `finish()`
/// runs) when the collector drops the sender.
pub fn spawn_writer(
    mut recorder: Box<dyn Recorder>,
) -> (SyncSender<DispatchSample>, JoinHandle<()>) {
    let (tx, rx) = sync_channel::<DispatchSample>(WRITER_QUEUE_DEPTH);
    let handle = thread::Builder::new()
        .name("storm-report-writer".into())
        .spawn(move || {
            while let Ok(s) = rx.recv() {
                if let Err(e) = recorder.record(&s) {
                    // Time-series is auxiliary: losing it mid-run (disk full,
                    // closed pipe) must not kill the load test. Warn once, stop
                    // recording, keep the summary intact.
                    tracing::warn!("report sink failed, disabling time-series output: {e}");
                    break;
                }
            }
            if let Err(e) = recorder.finish() {
                tracing::warn!("report sink failed on finish: {e}");
            }
        })
        .expect("spawn storm report-writer thread");
    (tx, handle)
}

/// CSV/JSONL file sink (`path: "-"` = stdout).
///
/// CSV columns: `t_s,latency_ms,ok,recalls_full,recalls_short` — each recall
/// column is `;`-joined (one value per ground-truthed query in the dispatch
/// that fell in that bucket; usually 0 or 1 values total at batch_size=1),
/// empty when none. `recalls_full` holds queries scored against `top_k`;
/// `recalls_short` holds queries whose ground truth was shorter than `top_k`
/// and were scored against their own length (see [`recall_at_k`](
/// crate::runner)). JSONL carries the same split as two real arrays,
/// `recalls_full` / `recalls_short`.
struct FileRecorder {
    path: String,
    format: ReportFormat,
    out: Option<BufWriter<Box<dyn Write + Send>>>,
}

impl Recorder for FileRecorder {
    fn begin(&mut self) -> io::Result<()> {
        let sink: Box<dyn Write + Send> = if self.path == "-" {
            Box::new(io::stdout())
        } else {
            // File::create truncates — flag it so an operator doesn't silently
            // clobber a prior run's series by reusing a path (e.g. across a
            // sweep's iterations).
            if std::path::Path::new(&self.path).exists() {
                tracing::warn!("report path `{}` already exists — overwriting", self.path);
            }
            Box::new(File::create(&self.path)?)
        };
        let mut out = BufWriter::new(sink);
        if self.format == ReportFormat::Csv {
            writeln!(out, "t_s,latency_ms,ok,recalls_full,recalls_short")?;
        }
        self.out = Some(out);
        Ok(())
    }

    fn record(&mut self, s: &DispatchSample) -> io::Result<()> {
        let out = self.out.as_mut().ok_or_else(|| {
            io::Error::other("record() before begin()")
        })?;
        // One dispatch's recall samples split into the two buckets — a query is
        // in exactly one, so the two joins together reproduce every sample.
        let join = |short: bool, sep: &str| {
            s.recalls
                .iter()
                .filter(|r| r.short == short)
                .map(|r| format!("{:.4}", r.recall))
                .collect::<Vec<_>>()
                .join(sep)
        };
        match self.format {
            ReportFormat::Csv => writeln!(
                out,
                "{:.6},{:.3},{},{},{}",
                s.t_s,
                s.latency_ms,
                s.ok,
                join(false, ";"),
                join(true, ";"),
            ),
            ReportFormat::Jsonl => {
                // Hand-rolled: fields are two floats, a bool, and two float
                // arrays — no serde derive needed on the hot sample type.
                writeln!(
                    out,
                    r#"{{"t_s":{:.6},"latency_ms":{:.3},"ok":{},"recalls_full":[{}],"recalls_short":[{}]}}"#,
                    s.t_s,
                    s.latency_ms,
                    s.ok,
                    join(false, ","),
                    join(true, ","),
                )
            }
        }
    }

    fn finish(&mut self) -> io::Result<()> {
        match self.out.as_mut() {
            Some(out) => out.flush(),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::RecallSample;

    /// `full` / `short` are the recall values for each bucket; the helper tags
    /// them and concatenates into the one `recalls` vec a real dispatch carries.
    fn sample(t_s: f64, latency_ms: f64, ok: bool, full: Vec<f64>, short: Vec<f64>) -> DispatchSample {
        let recalls = full
            .into_iter()
            .map(|recall| RecallSample { recall, short: false })
            .chain(short.into_iter().map(|recall| RecallSample { recall, short: true }))
            .collect();
        DispatchSample { t_s, latency_ms, ok, timed_out: false, recalls, empty_ground_truth: 0, filter_overreturn: 0 }
    }

    fn run_recorder(cfg: &ReportConfig, samples: &[DispatchSample]) -> String {
        let mut rec = cfg.build();
        rec.begin().expect("begin creates the file");
        for s in samples {
            rec.record(s).expect("record");
        }
        rec.finish().expect("finish");
        std::fs::read_to_string(&cfg.path).expect("file exists after begin()")
    }

    #[test]
    fn csv_has_header_and_one_row_per_dispatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.csv").to_string_lossy().into_owned();
        let cfg = ReportConfig { format: ReportFormat::Csv, path };

        let text = run_recorder(
            &cfg,
            &[
                // one full-depth (0.9) and one short (0.5) recall in the same dispatch
                sample(0.001, 12.5, true, vec![0.9, 1.0], vec![0.5]),
                sample(0.052, 8.25, false, vec![], vec![]),
            ],
        );
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "t_s,latency_ms,ok,recalls_full,recalls_short");
        assert_eq!(lines[1], "0.001000,12.500,true,0.9000;1.0000,0.5000");
        assert_eq!(lines[2], "0.052000,8.250,false,,");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn jsonl_rows_parse_and_roundtrip_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.jsonl").to_string_lossy().into_owned();
        let cfg = ReportConfig { format: ReportFormat::Jsonl, path };

        let text = run_recorder(&cfg, &[sample(1.5, 20.0, true, vec![0.5], vec![0.25])]);
        let row: serde_json::Value = serde_json::from_str(text.trim()).expect("valid json");
        assert_eq!(row["t_s"], 1.5);
        assert_eq!(row["latency_ms"], 20.0);
        assert_eq!(row["ok"], true);
        assert_eq!(row["recalls_full"][0], 0.5);
        assert_eq!(row["recalls_short"][0], 0.25);
    }

    #[test]
    fn begin_fails_fast_on_an_unwritable_path() {
        let cfg = ReportConfig {
            format: ReportFormat::Csv,
            path: "/nonexistent-dir/ts.csv".into(),
        };
        assert!(cfg.build().begin().is_err()); // dies BEFORE load is offered
    }
}
