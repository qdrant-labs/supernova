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
use std::path::{Path, PathBuf};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use serde::Deserialize;

use crate::runner::DispatchSample;
use crate::runner::Summary;

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
        Box::new(FileRecorder {
            path: self.path.clone(),
            format: self.format,
            out: None,
        })
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
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| io::Error::other("record() before begin()"))?;
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

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReportRow {
    pub source: String,
    pub tool: String,
    pub requests: u64,
    pub errors: u64,
    pub error_rate: f64,
    pub requests_per_sec: f64,
    pub qps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub recall_total_mean: Option<f64>,
    pub recall_total_n: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkRollup {
    pub inputs: usize,
    pub total_requests: u64,
    pub total_errors: u64,
    pub error_rate: f64,
    pub summed_requests_per_sec: f64,
    pub summed_qps: f64,
    pub approx_weighted_p95_ms: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReportOutput {
    pub schema_version: u32,
    pub rows: Vec<ReportRow>,
    pub rollup: BenchmarkRollup,
}

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error("no input files were provided")]
    EmptyInputs,
    #[error("failed to read `{path}`: {source}")]
    Read { path: String, source: io::Error },
    #[error("could not parse `{path}` as storm JSON summary or Locust CSV")]
    Unsupported { path: String },
    #[error("failed to parse `{path}` as CSV: {source}")]
    Csv { path: String, source: csv::Error },
    #[error("failed to parse `{path}` as JSON: {source}")]
    Json {
        path: String,
        source: serde_json::Error,
    },
}

pub fn build_report(inputs: &[PathBuf]) -> Result<ReportOutput, ReportError> {
    if inputs.is_empty() {
        return Err(ReportError::EmptyInputs);
    }
    let mut rows = Vec::with_capacity(inputs.len());
    for input in inputs {
        rows.push(parse_report_row(input)?);
    }
    let rollup = rollup_rows(&rows);
    Ok(ReportOutput {
        schema_version: 1,
        rows,
        rollup,
    })
}

pub fn print_report_table(out: &ReportOutput) {
    println!(
        "{:<28} {:<7} {:>10} {:>8} {:>8} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>12}",
        "source",
        "tool",
        "requests",
        "errors",
        "err%",
        "req/s",
        "qps",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "recall_total",
    );
    println!("{}", "-".repeat(150));
    for row in &out.rows {
        let recall = match (row.recall_total_mean, row.recall_total_n) {
            (Some(mean), Some(n)) => format!("{mean:.4} (n={n})"),
            _ => "-".to_string(),
        };
        println!(
            "{:<28} {:<7} {:>10} {:>8} {:>8.2} {:>10.1} {:>10.1} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>12}",
            shorten_source(&row.source, 28),
            row.tool,
            row.requests,
            row.errors,
            row.error_rate * 100.0,
            row.requests_per_sec,
            row.qps,
            row.p50_ms,
            row.p95_ms,
            row.p99_ms,
            row.max_ms,
            recall
        );
    }
    println!("{}", "-".repeat(150));
    println!(
        "rollup: inputs={} requests={} errors={} err%={:.2} req/s(sum)={:.1} qps(sum)={:.1} approx_weighted_p95_ms={:.2}",
        out.rollup.inputs,
        out.rollup.total_requests,
        out.rollup.total_errors,
        out.rollup.error_rate * 100.0,
        out.rollup.summed_requests_per_sec,
        out.rollup.summed_qps,
        out.rollup.approx_weighted_p95_ms,
    );
}

fn parse_report_row(path: &Path) -> Result<ReportRow, ReportError> {
    if let Ok(summary) = parse_storm_summary(path) {
        return Ok(storm_row(path, summary));
    }
    if let Ok(row) = parse_locust_csv(path) {
        return Ok(row);
    }
    Err(ReportError::Unsupported {
        path: path.display().to_string(),
    })
}

fn parse_storm_summary(path: &Path) -> Result<Summary, ReportError> {
    let text = std::fs::read_to_string(path).map_err(|source| ReportError::Read {
        path: path.display().to_string(),
        source,
    })?;
    // Accept either a single JSON object file OR logs containing one JSON line.
    if let Ok(summary) = serde_json::from_str::<Summary>(text.trim()) {
        return Ok(summary);
    }
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('{')
            && line.contains("\"requests\"")
            && let Ok(summary) = serde_json::from_str::<Summary>(line)
        {
            return Ok(summary);
        }
    }
    Err(ReportError::Json {
        path: path.display().to_string(),
        source: serde_json::Error::io(io::Error::other("no parseable storm JSON summary found")),
    })
}

fn storm_row(path: &Path, summary: Summary) -> ReportRow {
    ReportRow {
        source: path.display().to_string(),
        tool: "storm".to_string(),
        requests: summary.requests,
        errors: summary.errors,
        error_rate: ratio(summary.errors, summary.requests),
        requests_per_sec: summary.requests_per_sec,
        qps: summary.qps,
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        p99_ms: summary.p99_ms,
        max_ms: summary.max_ms,
        recall_total_mean: summary.total_recall.map(|b| b.mean),
        recall_total_n: summary.total_recall.map(|b| b.n),
    }
}

fn parse_locust_csv(path: &Path) -> Result<ReportRow, ReportError> {
    let mut reader = csv::Reader::from_path(path).map_err(|source| ReportError::Csv {
        path: path.display().to_string(),
        source,
    })?;
    let headers = reader.headers().map_err(|source| ReportError::Csv {
        path: path.display().to_string(),
        source,
    })?;
    let idx = LocustCols::from_headers(headers)?;
    for rec in reader.records() {
        let rec = rec.map_err(|source| ReportError::Csv {
            path: path.display().to_string(),
            source,
        })?;
        let name = rec.get(idx.name).unwrap_or_default().trim();
        let typ = rec.get(idx.typ).unwrap_or_default().trim();
        if !name.eq_ignore_ascii_case("Aggregated") && !typ.eq_ignore_ascii_case("Aggregated") {
            continue;
        }
        let requests = parse_u64(rec.get(idx.requests));
        let errors = parse_u64(rec.get(idx.failures));
        let requests_per_sec = parse_f64(rec.get(idx.rps));
        return Ok(ReportRow {
            source: path.display().to_string(),
            tool: "locust".to_string(),
            requests,
            errors,
            error_rate: ratio(errors, requests),
            requests_per_sec,
            qps: requests_per_sec,
            p50_ms: parse_f64(rec.get(idx.p50)),
            p95_ms: parse_f64(rec.get(idx.p95)),
            p99_ms: parse_f64(rec.get(idx.p99)),
            max_ms: parse_f64(rec.get(idx.max)),
            recall_total_mean: None,
            recall_total_n: None,
        });
    }
    Err(ReportError::Unsupported {
        path: path.display().to_string(),
    })
}

struct LocustCols {
    typ: usize,
    name: usize,
    requests: usize,
    failures: usize,
    rps: usize,
    p50: usize,
    p95: usize,
    p99: usize,
    max: usize,
}

impl LocustCols {
    fn from_headers(headers: &csv::StringRecord) -> Result<Self, ReportError> {
        let find = |options: &[&str]| -> Option<usize> {
            headers.iter().position(|h| {
                options
                    .iter()
                    .any(|candidate| h.trim().eq_ignore_ascii_case(candidate))
            })
        };
        let missing = || ReportError::Unsupported {
            path: "<csv>".to_string(),
        };
        Ok(Self {
            typ: find(&["Type"]).ok_or_else(missing)?,
            name: find(&["Name"]).ok_or_else(missing)?,
            requests: find(&["Request Count"]).ok_or_else(missing)?,
            failures: find(&["Failure Count"]).ok_or_else(missing)?,
            rps: find(&["Requests/s"]).ok_or_else(missing)?,
            p50: find(&["Median Response Time", "50%"]).ok_or_else(missing)?,
            p95: find(&["95%"]).ok_or_else(missing)?,
            p99: find(&["99%"]).ok_or_else(missing)?,
            max: find(&["Max Response Time", "Max"]).ok_or_else(missing)?,
        })
    }
}

fn rollup_rows(rows: &[ReportRow]) -> BenchmarkRollup {
    let total_requests = rows.iter().map(|r| r.requests).sum::<u64>();
    let total_errors = rows.iter().map(|r| r.errors).sum::<u64>();
    let summed_requests_per_sec = rows.iter().map(|r| r.requests_per_sec).sum::<f64>();
    let summed_qps = rows.iter().map(|r| r.qps).sum::<f64>();
    let approx_weighted_p95_ms = weighted_average(rows.iter().map(|r| (r.p95_ms, r.requests)));
    BenchmarkRollup {
        inputs: rows.len(),
        total_requests,
        total_errors,
        error_rate: ratio(total_errors, total_requests),
        summed_requests_per_sec,
        summed_qps,
        approx_weighted_p95_ms,
    }
}

fn weighted_average(items: impl Iterator<Item = (f64, u64)>) -> f64 {
    let mut numer = 0.0;
    let mut denom = 0.0;
    for (value, weight) in items {
        if weight == 0 {
            continue;
        }
        numer += value * weight as f64;
        denom += weight as f64;
    }
    if denom > 0.0 { numer / denom } else { 0.0 }
}

fn parse_u64(raw: Option<&str>) -> u64 {
    raw.unwrap_or_default().trim().parse().unwrap_or(0)
}

fn parse_f64(raw: Option<&str>) -> f64 {
    raw.unwrap_or_default().trim().parse().unwrap_or(0.0)
}

fn ratio(numer: u64, denom: u64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        numer as f64 / denom as f64
    }
}

fn shorten_source(source: &str, max_len: usize) -> String {
    let chars: Vec<char> = source.chars().collect();
    if chars.len() <= max_len {
        return source.to_string();
    }
    let keep = max_len.saturating_sub(3);
    let tail: String = chars[chars.len().saturating_sub(keep)..].iter().collect();
    format!("...{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::OperationKind;
    use crate::runner::RecallSample;

    /// `full` / `short` are the recall values for each bucket; the helper tags
    /// them and concatenates into the one `recalls` vec a real dispatch carries.
    fn sample(
        t_s: f64,
        latency_ms: f64,
        ok: bool,
        full: Vec<f64>,
        short: Vec<f64>,
    ) -> DispatchSample {
        let recalls = full
            .into_iter()
            .map(|recall| RecallSample {
                recall,
                short: false,
            })
            .chain(short.into_iter().map(|recall| RecallSample {
                recall,
                short: true,
            }))
            .collect();
        DispatchSample {
            t_s,
            operation: OperationKind::Query,
            latency_ms,
            ok,
            recalls,
            empty_ground_truth: 0,
            filter_overreturn: 0,
        }
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
        let cfg = ReportConfig {
            format: ReportFormat::Csv,
            path,
        };

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
        let cfg = ReportConfig {
            format: ReportFormat::Jsonl,
            path,
        };

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

    #[test]
    fn build_report_parses_storm_summary_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("storm.json");
        std::fs::write(
            &path,
            r#"{
  "requests": 100,
  "errors": 2,
  "batch_size": 4,
  "requests_per_sec": 50.0,
  "qps": 200.0,
  "p50_ms": 10.0,
  "p95_ms": 20.0,
  "p99_ms": 30.0,
  "max_ms": 40.0,
  "full_recall": null,
  "short_recall": null,
  "total_recall": {"n": 100, "mean": 0.91},
  "empty_ground_truth": 0,
  "filter_overreturn": 0
}"#,
        )
        .unwrap();

        let out = build_report(&[path]).expect("storm report parses");
        assert_eq!(out.rows.len(), 1);
        let row = &out.rows[0];
        assert_eq!(row.tool, "storm");
        assert_eq!(row.requests, 100);
        assert_eq!(row.errors, 2);
        assert_eq!(row.recall_total_n, Some(100));
        assert_eq!(row.recall_total_mean, Some(0.91));
        assert_eq!(out.rollup.total_requests, 100);
    }

    #[test]
    fn build_report_parses_locust_aggregated_csv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.csv");
        std::fs::write(
            &path,
            "Type,Name,Request Count,Failure Count,Median Response Time,95%,99%,Max Response Time,Requests/s\n\
GET,/search,10,1,5,9,12,20,100\n\
Aggregated,Aggregated,1000,25,7,15,30,70,450\n",
        )
        .unwrap();

        let out = build_report(&[path]).expect("locust report parses");
        assert_eq!(out.rows.len(), 1);
        let row = &out.rows[0];
        assert_eq!(row.tool, "locust");
        assert_eq!(row.requests, 1000);
        assert_eq!(row.errors, 25);
        assert_eq!(row.requests_per_sec, 450.0);
        assert_eq!(row.qps, 450.0);
        assert_eq!(row.p95_ms, 15.0);
        assert!(row.recall_total_mean.is_none());
        assert_eq!(out.rollup.total_errors, 25);
    }
}
