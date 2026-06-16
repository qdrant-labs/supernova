//! `nova storm` — sustained query load testing for vector stores.
//!
//! A *storm* fires nearest-neighbour queries at a target cluster and records
//! the latency distribution. Work is **replicated, not partitioned**: every
//! worker runs the same [`LoadProfile`](config::LoadProfile), so total offered
//! load is roughly `num_workers × {concurrency or qps}`.
//!
//! The split mirrors `nova-load`: a backend ([`QueryTarget`](targets::QueryTarget))
//! is a thin "fire one query, report its latency" adapter, and the runner owns
//! the load *shape* (how many in flight, for how long, at what rate). The same
//! [`run_storm`](runner::run_storm) drives any backend.

#[cfg(not(feature = "qdrant"))]
compile_error!("enable at least one target backend feature, e.g. --features qdrant");

pub mod config;
pub mod errors;
pub mod queries;
pub mod runner;
pub mod targets;
