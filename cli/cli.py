#!/usr/bin/env python3
"""
Unified CLI for vectorforge.

Each subcommand forwards its raw args to a per-command argparse-based
entry point in this package (cli.run_*). The click layer here only
provides the top-level command dispatch and `--help` listing; per-command
flag parsing, validation, and error messages still come from each
underlying ``main(argv)``.
"""

import click

# Pass-through context settings: click does not interpret subcommand flags,
# so options like --dry-run, --num-jobs, etc. flow straight to the
# underlying argparse parser. help_option_names=[] disables click's --help
# at the subcommand level so `vf <cmd> --help` shows the argparse help
# (which is more detailed than this thin pass-through).
_PASSTHROUGH = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
    "help_option_names": [],
}


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="vectorforge — stream, embed, store, and evaluate massive datasets at scale.",
)
def main():
    """Top-level entry point. Run `vf <command> --help` for command-specific options."""


@main.command(name="embed", context_settings=_PASSTHROUGH, short_help="Embed a dataset locally.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def embed_cmd(args):
    """Embed a dataset locally."""
    from cli.run_embedder import main as embed_main
    embed_main(list(args))


@main.command(name="embed-dist", context_settings=_PASSTHROUGH, short_help="Embed distributed via SkyPilot pool.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def embed_dist_cmd(args):
    """Embed distributed via SkyPilot pool."""
    from cli.run_embed_distributed import main as embed_dist_main
    embed_dist_main(list(args))


@main.command(
    name="partition",
    context_settings=_PASSTHROUGH,
    short_help="Run pipeline with no-op embedder (validate sharding without GPU).",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def partition_cmd(args):
    """Run pipeline with the no-op embedder to validate sharding without GPU."""
    from cli.run_partition import main as partition_main
    partition_main(list(args))


@main.command(name="partition-dist", context_settings=_PASSTHROUGH, short_help="Distributed partition via SkyPilot pool.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def partition_dist_cmd(args):
    """Distributed partition via SkyPilot pool."""
    from cli.run_partition_distributed import main as partition_dist_main
    partition_dist_main(list(args))


@main.command(name="load", context_settings=_PASSTHROUGH, short_help="Load pre-embedded data into a vector store.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def load_cmd(args):
    """Load pre-embedded data into a vector store."""
    from cli.run_loader import main as load_main
    load_main(list(args))


@main.command(name="load-dist", context_settings=_PASSTHROUGH, short_help="Distribute loading across SkyPilot instances.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def load_dist_cmd(args):
    """Distribute loading across SkyPilot instances."""
    from cli.run_load_distributed import main as load_dist_main
    load_dist_main(list(args))


@main.command(name="push-hf", context_settings=_PASSTHROUGH, short_help="Upload S3 parquets to a HuggingFace Hub dataset.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def push_hf_cmd(args):
    """Upload S3 parquets to a HuggingFace Hub dataset."""
    from cli.run_push_hf import main as push_hf_main
    push_hf_main(list(args))


@main.command(name="push-hf-dist", context_settings=_PASSTHROUGH, short_help="Distribute HF upload across SkyPilot instances.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def push_hf_dist_cmd(args):
    """Distribute HF upload across SkyPilot instances."""
    from cli.run_push_hf_distributed import main as push_hf_dist_main
    push_hf_dist_main(list(args))


@main.command(
    name="generate-queries",
    context_settings=_PASSTHROUGH,
    short_help="Sample N rows as eval queries (launches EC2; --local to run here).",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def generate_queries_cmd(args):
    """Sample N rows as eval queries (launches EC2; --local to run here)."""
    from cli.run_generate_queries import main as gen_queries_main
    gen_queries_main(list(args))


@main.command(
    name="brute-force",
    context_settings=_PASSTHROUGH,
    short_help="Exhaustive nearest-neighbor search for recall eval (single GPU).",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def brute_force_cmd(args):
    """Exhaustive nearest-neighbor search for recall eval (single GPU)."""
    from cli.run_brute_force import main as brute_force_main
    brute_force_main(list(args))


@main.command(name="brute-force-dist", context_settings=_PASSTHROUGH, short_help="Distributed brute-force via SkyPilot GPU pool.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def brute_force_dist_cmd(args):
    """Distributed brute-force via SkyPilot GPU pool."""
    from cli.run_brute_force_distributed import main as brute_force_dist_main
    brute_force_dist_main(list(args))


@main.command(
    name="brute-force-merge",
    context_settings=_PASSTHROUGH,
    short_help="Merge partial results from a distributed brute-force run.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def brute_force_merge_cmd(args):
    """Merge partial results from a distributed brute-force run."""
    from cli.run_brute_force import merge_main
    merge_main(list(args))


@main.command(name="analysis", context_settings=_PASSTHROUGH, short_help="Analyze a (distributed) embedding run.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def analysis_cmd(args):
    """Analyze a (distributed) embedding run."""
    from cli.run_analysis import main as analysis_main
    analysis_main(list(args))


@main.command(
    name="throughput-predict",
    context_settings=_PASSTHROUGH,
    short_help="Predict embedding throughput + cost from a config.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def throughput_predict_cmd(args):
    """Predict embedding throughput + cost from a config."""
    from cli.run_throughput_predict import main as throughput_predict_main
    throughput_predict_main(list(args))


if __name__ == "__main__":
    main()
