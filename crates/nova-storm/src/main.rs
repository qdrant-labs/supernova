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
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nova_storm=info".into()),
        )
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
            println!("{}", "=".repeat(50));
            println!("{summary}");
            println!("{}", "=".repeat(50));
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
