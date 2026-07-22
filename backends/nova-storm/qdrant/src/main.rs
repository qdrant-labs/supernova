use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use nova_storm::config::StormConfig;

/// Load-test a vector store from this machine (the `nova storm` workload).
#[derive(Debug, Parser)]
#[command(name = "nova-storm", version, about)]
struct Cli {
    /// Path to the storm config YAML.
    config: PathBuf,
    /// Print the summary as a single JSON line instead of the human-readable
    /// table — for a caller (e.g. `nova sweep`) that needs to parse the result
    /// programmatically rather than scrape formatted text.
    #[arg(long)]
    json: bool,
}

/// The stable capabilities descriptor for this backend. Kept in lockstep with
/// `contracts/nova-storm/v1.yaml`: `commands`, `methods`, `search_modes`, and
/// `features` here are exactly what that contract declares required, and
/// `nova-contract check` fails if this drifts from the contract file.
///
/// - `commands`: `run` is the default positional-config invocation
///   (`nova-storm-qdrant <config> [--json]`); `capabilities` is this descriptor.
/// - `methods` mirror the `QueryTarget` trait in `targets/mod.rs`.
/// - `search_modes` mirror `runner.rs`'s closed-loop / open-loop modes.
fn capabilities_json() -> serde_json::Value {
    serde_json::json!({
        "contract": "nova-storm-backend/v1",
        "backend": "qdrant",
        "commands": ["run", "capabilities"],
        "methods": ["query_batch", "close"],
        "search_modes": ["closed_loop", "open_loop"],
        "features": ["recall", "percentiles", "exact_search", "hnsw_ef"]
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    // `capabilities` is a pure descriptor handled before clap so the legacy
    // `nova-storm <config> [--json]` positional CLI is untouched. Accepts an
    // optional `--json` (output is always JSON regardless).
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("capabilities") {
        println!(
            "{}",
            serde_json::to_string_pretty(&capabilities_json())
                .expect("capabilities descriptor is always serializable")
        );
        return ExitCode::SUCCESS;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nova_storm=info".into()),
        )
        // Logs go to stderr so stdout carries only the run's actual output
        // (the human-readable table, or with `--json`, exactly one JSON
        // line) — otherwise a caller like `nova sweep` parsing stdout as
        // JSON gets log lines corrupting it.
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let config = match StormConfig::from_path(&cli.config) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: failed to load config `{}`: {err}", cli.config.display());
            return ExitCode::FAILURE;
        }
    };

    match nova_storm::run(config).await {
        Ok(summary) => {
            if cli.json {
                match serde_json::to_string(&summary) {
                    Ok(json) => println!("{json}"),
                    Err(err) => {
                        eprintln!("error: failed to serialize summary as JSON: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                println!("{}", "=".repeat(50));
                println!("{summary}");
                println!("{}", "=".repeat(50));
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
