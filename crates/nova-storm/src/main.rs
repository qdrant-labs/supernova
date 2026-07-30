use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

use nova_storm::config::StormConfig;
use nova_storm::report;

/// Load-test a vector store from this machine (the `nova storm` workload).
#[derive(Debug, Parser)]
#[command(name = "nova-storm", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// Path to the storm config YAML (legacy shorthand for `run <config>`).
    config: Option<PathBuf>,
    /// Print the summary as a single JSON line instead of the human-readable
    /// table — for a caller (e.g. `nova sweep`) that needs to parse the result
    /// programmatically rather than scrape formatted text.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a storm load test from config.
    Run(RunArgs),
    /// Normalize + compare storm JSON and Locust CSV results.
    Report(ReportArgs),
}

#[derive(Debug, Args)]
struct RunArgs {
    /// Path to the storm config YAML.
    config: PathBuf,
    /// Print the summary as one JSON line.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// Input files: storm JSON summary files and/or Locust stats CSV.
    #[arg(long, required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// Emit normalized report JSON to this file.
    #[arg(long)]
    output_json: Option<PathBuf>,
    /// Print normalized output as JSON to stdout.
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
    let command = match (cli.command, cli.config) {
        (Some(cmd), None) => cmd,
        (None, Some(config)) => Command::Run(RunArgs {
            config,
            json: cli.json,
        }),
        (Some(_), Some(_)) => {
            eprintln!("error: provide either a subcommand or legacy <config> form, not both");
            return ExitCode::FAILURE;
        }
        (None, None) => {
            eprintln!("error: missing command (use `run <config>` or `report --inputs ...`)");
            return ExitCode::FAILURE;
        }
    };

    match command {
        Command::Run(args) => run_storm(args).await,
        Command::Report(args) => run_report(args),
    }
}

async fn run_storm(args: RunArgs) -> ExitCode {
    let config = match StormConfig::from_path(&args.config) {
        Ok(config) => config,
        Err(err) => {
            eprintln!(
                "error: failed to load config `{}`: {err}",
                args.config.display()
            );
            return ExitCode::FAILURE;
        }
    };

    match nova_storm::run(config).await {
        Ok(summary) => {
            if args.json {
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

fn run_report(args: ReportArgs) -> ExitCode {
    let out = match report::build_report(&args.inputs) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(path) = &args.output_json {
        match serde_json::to_vec_pretty(&out) {
            Ok(bytes) => {
                if let Some(parent) = path.parent()
                    && !parent.as_os_str().is_empty()
                    && let Err(err) = std::fs::create_dir_all(parent)
                {
                    eprintln!("error: failed to create `{}`: {err}", parent.display());
                    return ExitCode::FAILURE;
                }
                if let Err(err) = std::fs::write(path, bytes) {
                    eprintln!("error: failed to write `{}`: {err}", path.display());
                    return ExitCode::FAILURE;
                }
            }
            Err(err) => {
                eprintln!("error: failed to serialize output JSON: {err}");
                return ExitCode::FAILURE;
            }
        }
    }

    if args.json {
        match serde_json::to_string_pretty(&out) {
            Ok(json) => println!("{json}"),
            Err(err) => {
                eprintln!("error: failed to serialize output JSON: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        report::print_report_table(&out);
    }
    ExitCode::SUCCESS
}
