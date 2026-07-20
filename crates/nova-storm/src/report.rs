//! Transient (time-series) reporting: one row per batch dispatch, as it lands.
//!
//! The end-of-run [`Summary`](crate::runner::Summary) collapses the time axis —
//! a latency spike at t=90s from the cluster re-optimizing is indistinguishable
//! from the same samples spread evenly. A [`Recorder`] preserves it: the
//! collector task (which already sees every [`DispatchSample`] off the hot
//! path) forwards each sample here as it arrives, so acute load behavior and
//! cluster-adjustment transients can be plotted afterwards.
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

use serde::Deserialize;

use crate::runner::DispatchSample;

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

/// A sink for per-dispatch samples, driven by the collector task.
///
/// Lifecycle: `begin()` once before the load starts (create the file / write
/// the CSV header / create tables — fail HERE, not mid-run, so a bad path
/// dies before load is offered), `record()` per dispatch in completion order,
/// `finish()` once after the last sample (flush/close).
///
/// Called from the collector task only — implementations need `Send` but never
/// synchronization, and buffered blocking writes are fine: the workers are
/// decoupled by the channel, so a sink can't perturb the load loop.
pub trait Recorder: Send {
    fn begin(&mut self) -> io::Result<()>;
    fn record(&mut self, sample: &DispatchSample) -> io::Result<()>;
    fn finish(&mut self) -> io::Result<()>;
}

/// CSV/JSONL file sink (`path: "-"` = stdout).
///
/// CSV columns: `t_s,latency_ms,ok,recalls` — `recalls` is `;`-joined (one
/// value per ground-truthed query in the dispatch; usually 0 or 1 values at
/// batch_size=1), empty when none. JSONL carries the same fields with
/// `recalls` as a real array.
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
            Box::new(File::create(&self.path)?)
        };
        let mut out = BufWriter::new(sink);
        if self.format == ReportFormat::Csv {
            writeln!(out, "t_s,latency_ms,ok,recalls")?;
        }
        self.out = Some(out);
        Ok(())
    }

    fn record(&mut self, s: &DispatchSample) -> io::Result<()> {
        let out = self.out.as_mut().ok_or_else(|| {
            io::Error::other("record() before begin()")
        })?;
        match self.format {
            ReportFormat::Csv => {
                let recalls = s
                    .recalls
                    .iter()
                    .map(|r| format!("{r:.4}"))
                    .collect::<Vec<_>>()
                    .join(";");
                writeln!(out, "{:.6},{:.3},{},{}", s.t_s, s.latency_ms, s.ok, recalls)
            }
            ReportFormat::Jsonl => {
                // Hand-rolled: fields are two floats, a bool, and a float
                // array — no serde derive needed on the hot sample type.
                let recalls = s
                    .recalls
                    .iter()
                    .map(|r| format!("{r:.4}"))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(
                    out,
                    r#"{{"t_s":{:.6},"latency_ms":{:.3},"ok":{},"recalls":[{}]}}"#,
                    s.t_s, s.latency_ms, s.ok, recalls
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

    fn sample(t_s: f64, latency_ms: f64, ok: bool, recalls: Vec<f64>) -> DispatchSample {
        DispatchSample { t_s, latency_ms, ok, recalls }
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
                sample(0.001, 12.5, true, vec![0.9, 1.0]),
                sample(0.052, 8.25, false, vec![]),
            ],
        );
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "t_s,latency_ms,ok,recalls");
        assert_eq!(lines[1], "0.001000,12.500,true,0.9000;1.0000");
        assert_eq!(lines[2], "0.052000,8.250,false,");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn jsonl_rows_parse_and_roundtrip_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ts.jsonl").to_string_lossy().into_owned();
        let cfg = ReportConfig { format: ReportFormat::Jsonl, path };

        let text = run_recorder(&cfg, &[sample(1.5, 20.0, true, vec![0.5])]);
        let row: serde_json::Value = serde_json::from_str(text.trim()).expect("valid json");
        assert_eq!(row["t_s"], 1.5);
        assert_eq!(row["latency_ms"], 20.0);
        assert_eq!(row["ok"], true);
        assert_eq!(row["recalls"][0], 0.5);
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
