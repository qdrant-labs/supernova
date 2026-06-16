//! The local-dev sink: route measurements to `tracing`.
//!
//! The default when a config has no `metrics:` block — local-first, you see
//! your numbers with zero setup. High-rate scalar streams go to DEBUG so they
//! don't flood the terminal: per-sample `observe`, and the per-second `log`
//! samples (rolling QPS/wps, cumulative counts) a long run emits — those are
//! for a real time-series backend, and a TTY already gets the live progress
//! bar while a non-TTY run gets the runner's periodic milestone logs.
//! Lifecycle output (`start`/`event`/`summary`/`finish`) stays at INFO.

use std::sync::Mutex;

use crate::{MetricsSink, RunContext};

pub struct StdoutSink {
    // Only set once at start(); a Mutex keeps the sink `Sync` without making the
    // whole trait take `&mut self`.
    ident: Mutex<Ident>,
}

#[derive(Default, Clone)]
struct Ident {
    run_id: String,
    node_id: Option<String>,
}

impl StdoutSink {
    pub fn new() -> Self {
        Self {
            ident: Mutex::new(Ident::default()),
        }
    }

    fn prefix(&self) -> String {
        let id = self.ident.lock().unwrap();
        match &id.node_id {
            Some(node) => format!("[{}/{}]", id.run_id, node),
            None => format!("[{}]", id.run_id),
        }
    }
}

impl Default for StdoutSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsSink for StdoutSink {
    fn start(&self, run_id: &str, ctx: &RunContext<'_>) {
        {
            let mut id = self.ident.lock().unwrap();
            id.run_id = run_id.to_string();
            id.node_id = ctx.node_id.map(str::to_string);
        }
        tracing::info!("{} run started (command={})", self.prefix(), ctx.command);
    }

    fn log(&self, name: &str, value: f64) {
        // DEBUG, not INFO: a load/storm emits these every second for the whole
        // run. The live progress bar (TTY) and the runner's milestone logs
        // (non-TTY) cover human-facing progress; this stream is for a TSDB sink.
        tracing::debug!("{} {}={}", self.prefix(), name, value);
    }

    fn observe(&self, name: &str, value: f64, ok: bool) {
        tracing::debug!("{} {}={} ok={}", self.prefix(), name, value, ok);
    }

    fn event(&self, message: &str) {
        tracing::info!("{} · {}", self.prefix(), message);
    }

    fn summary(&self, values: &serde_json::Value) {
        tracing::info!("{} summary {}", self.prefix(), values);
    }

    fn finish(&self, status: &str) {
        tracing::info!("{} run finished ({})", self.prefix(), status);
    }
}
