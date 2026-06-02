#!/usr/bin/env python3
"""
Unified CLI for supernova.

Each subcommand is a real click command defined in its own ``cli/run_*.py``.
This module wires them together via a ``LazyGroup`` so heavy ML deps
(torch, sentence-transformers, FlagEmbedding, …) only load when the
relevant subcommand actually runs — `nova --help` stays fast.
"""

import importlib

import click


# Map of command name → ("module:attribute", short_help). The short_help is
# inlined here so `nova --help` doesn't have to import every subcommand module
# just to render the command list.
_LAZY_COMMANDS: dict[str, tuple[str, str]] = {
    "embed": ("supernova.cli.run_embedder:embed", "Embed a dataset locally."),
    "embed-dist": (
        "supernova.cli.run_embed_distributed:embed_dist",
        "Embed distributed via SkyPilot pool.",
    ),
    "partition": (
        "supernova.cli.run_partition:partition",
        "Run pipeline with no-op embedder (validate sharding without GPU).",
    ),
    "partition-dist": (
        "supernova.cli.run_partition_distributed:partition_dist",
        "Distributed partition via SkyPilot pool.",
    ),
    "load": ("supernova.cli.run_loader:load", "Load pre-embedded data into a vector store."),
    "load-dist": (
        "supernova.cli.run_load_distributed:load_dist",
        "Distribute loading across SkyPilot instances.",
    ),
    "generate-queries": (
        "supernova.cli.run_generate_queries:generate_queries",
        "Sample N rows as eval queries (launches EC2; --local to run here).",
    ),
    "subsample": (
        "supernova.cli.run_subsample:subsample",
        "Sample N random rows from a corpus to a local parquet file.",
    ),
    "brute-force": (
        "supernova.cli.run_brute_force:brute_force",
        "Exhaustive nearest-neighbor search for recall eval (single GPU).",
    ),
    "brute-force-dist": (
        "supernova.cli.run_brute_force_distributed:brute_force_dist",
        "Distributed brute-force via SkyPilot GPU pool.",
    ),
    "brute-force-merge": (
        "supernova.cli.run_brute_force:brute_force_merge",
        "Merge partial results from a distributed brute-force run.",
    ),
    "analysis": ("supernova.cli.run_analysis:analysis", "Analyze a (distributed) embedding run."),
    "throughput-predict": (
        "supernova.cli.run_throughput_predict:throughput_predict",
        "Predict embedding throughput + cost from a config.",
    ),
}


class LazyGroup(click.Group):
    """A ``click.Group`` that imports subcommands on first use.

    ``cli/run_embedder.py`` and friends pull in heavyweight ML libraries at
    import time. We don't want to pay that cost for every ``nova --help`` or
    completion lookup, so we resolve the actual command objects only when
    a user invokes them.
    """

    def __init__(
        self, *args, lazy_commands: dict[str, tuple[str, str]] | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lazy_commands = dict(lazy_commands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*self.commands, *self.lazy_commands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.commands:
            return self.commands[cmd_name]
        if cmd_name in self.lazy_commands:
            module_path, attr = self.lazy_commands[cmd_name][0].split(":", 1)
            module = importlib.import_module(module_path)
            cmd = getattr(module, attr)
            self.add_command(cmd, name=cmd_name)
            return cmd
        return None

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        # Render the command list using cached short_help strings so we
        # don't import every subcommand module just to describe them.
        rows: list[tuple[str, str]] = []
        for name in sorted({*self.commands, *self.lazy_commands}):
            if name in self.commands:
                rows.append((name, self.commands[name].get_short_help_str(limit=80)))
            else:
                rows.append((name, self.lazy_commands[name][1]))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(
    cls=LazyGroup,
    lazy_commands=_LAZY_COMMANDS,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="supernova — stream, embed, store, and evaluate massive datasets at scale.",
)
def main():
    """Top-level entry point. Run `nova <command> --help` for command-specific options."""


if __name__ == "__main__":
    main()
