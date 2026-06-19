use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use nova_load::config::LoadConfig;

/// Load vectors from a datasource into a vector store, per a YAML config.
#[derive(Debug, Parser)]
#[command(name = "nova-load", version, about)]
struct Cli {
    /// Path to the loader config YAML.
    config: PathBuf,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nova_load=info".into()),
        )
        .init();

    let cli = Cli::parse();

    let config = match LoadConfig::from_path(&cli.config) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: failed to load config `{}`: {err}", cli.config.display());
            return ExitCode::FAILURE;
        }
    };

    match nova_load::run_loader(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: load failed: {err}");
            ExitCode::FAILURE
        }
    }
}