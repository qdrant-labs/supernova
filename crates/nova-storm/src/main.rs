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

#[tokio::main]
async fn main() -> ExitCode {
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
            // Dispatch errors are findings, not crashes — a run with SOME
            // failures still measured something and exits 0. But a run where
            // EVERY dispatch failed (wrong dimension, missing collection,
            // absent sparse vector) measured nothing, and a scripted caller
            // gating on the exit code must not read it as a passing run.
            if summary.requests > 0 && summary.errors == summary.requests {
                eprintln!(
                    "error: all {} dispatches failed — nothing was measured (see the first-error \
                     warn above for the cause)",
                    summary.requests
                );
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
