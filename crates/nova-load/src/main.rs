use std::path::PathBuf;

use clap::Parser;

use nova_load::config::LoadConfig;
use nova_load::sources::DataSource;

/// Load vectors from a datasource into a vector store, per a YAML config.
#[derive(Debug, Parser)]
#[command(name = "nova-load", version, about)]
struct Cli {
    /// Path to the loader config YAML.
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), std::convert::Infallible> {
    let cli = Cli::parse();

    let config = LoadConfig::from_path(&cli.config).unwrap_or_else(|err| {
        eprintln!("Error parsing config file `{}`: {err}", cli.config.display());
        std::process::exit(1);
    });

    let source_cfg = config.datasource;
    let files = source_cfg.list_files().await.unwrap_or_else(|err| {
        eprintln!("Error listing files from datasource: {err}");
        std::process::exit(1);
    });
    println!("Found {} files to load.", files.len());
    for file in files {
        println!("  {} ({})", file.key, file.size.map_or("unknown size".to_string(), |s| format!("{} mB", s / 1_000_000)));
    }

    Ok(())
}
