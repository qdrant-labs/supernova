#!/usr/bin/env python3
"""
Unified CLI for supernova.

``nova`` is the single entry point, but it's polyglot underneath — and the
dispatcher itself imports nothing heavy. Three kinds of subcommand live behind it:

* **Local tools** (embed, storm, load) each run as their *own process*. ``nova
  <tool>`` is a thin passthrough that locates the tool's binary and ``exec``s it
  — the binary owns arg parsing, config, and metrics. storm/load are Rust crates
  (``nova-storm`` / ``nova-load``, via ``cargo install``); embed is a Python
  console script (``nova-embed``, via the ``embed`` extra). Same dispatch either
  way, so the ML stack never loads in the ``nova`` process.
* **Orchestrators** (the ``dist`` subgroup) are Python (SkyPilot SDK), lazily
  imported. On each worker they invoke the *local* ``nova <tool>`` (or its
  binary) on that node.
* **experiment** is a Python click command, lazily imported in-process.

So everything stays under the one ``nova`` roof regardless of implementation
language.
"""

import importlib
import os
import shutil

import click


# Map of command name → ("module:attribute", short_help). The short_help is
# inlined here so `nova --help` doesn't have to import every subcommand module
# just to render the command list. Rust-backed commands (storm, load) are NOT
# here — they're registered as binary-dispatch commands below.
_LAZY_COMMANDS: dict[str, tuple[str, str]] = {
    "experiment": (
        "supernova.cli.run_experiment:experiment",
        "Compose units over a timeline (workload tests).",
    ),
}


# The `dist` subgroup: `nova dist <tool>` orchestrates that tool across a
# SkyPilot fleet. The mental model is `nova <tool>` = run on this machine,
# `nova dist <tool>` = run the fleet. Each worker the orchestrator provisions
# invokes the *local* `nova <tool>` (or its Rust binary) on its own node.
_DIST_COMMANDS: dict[str, tuple[str, str]] = {
    "embed": (
        "supernova.cli.run_embed_distributed:embed_dist",
        "Embed across a SkyPilot GPU pool.",
    ),
    "load": (
        "supernova.cli.run_load_distributed:load_dist",
        "Distribute loading across a SkyPilot pool.",
    ),
    "storm": (
        "supernova.cli.run_storm_distributed:storm_dist",
        "Replicated load test across a SkyPilot pool.",
    ),
}


# Subprocess-backed commands: every local tool runs as its own process, so the
# `nova` dispatcher itself imports nothing heavy. `nova <name>` execs `binary`,
# forwarding all args untouched. The binary is a Rust crate (storm/load) or a
# Python console script (embed) — same dispatch either way; only the install
# hint shown when it's missing differs.
#   name → (binary, short_help, install_hint)
_CARGO = "cargo install --git https://github.com/qdrant-labs/supernova"
_BINARY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "embed": (
        "nova-embed",
        "Embed a dataset locally.",
        "pip install 'supernova[embed]'",
    ),
    "storm": (
        "nova-storm",
        "Load-test a vector store (single machine).",
        f"{_CARGO} nova-storm",
    ),
    "load": (
        "nova-load",
        "Load pre-embedded data into a vector store.",
        f"{_CARGO} nova-load",
    ),
}


def _binary_env_var(binary: str) -> str:
    """Dev override env var for a binary's path, e.g. nova-storm -> NOVA_STORM_BIN."""
    stem = binary.removeprefix("nova-").upper().replace("-", "_")
    return f"NOVA_{stem}_BIN"


def _resolve_binary(binary: str, install_hint: str) -> str:
    """Locate a subcommand binary: explicit override, then PATH."""
    override = os.environ.get(_binary_env_var(binary))
    if override:
        return override
    found = shutil.which(binary)
    if found:
        return found
    env = _binary_env_var(binary)
    raise click.UsageError(
        f"`{binary}` was not found on PATH.\n"
        f"  Install it:\n    {install_hint}\n"
        f"  Or, for local dev, point {env} at the binary:\n"
        f"    export {env}=/path/to/{binary}"
    )


def _exec_binary(binary: str, install_hint: str, args: tuple[str, ...]) -> None:
    """Replace this process with the binary (clean stdio / signals / exit code)."""
    exe = _resolve_binary(binary, install_hint)
    try:
        os.execv(exe, [binary, *args])
    except OSError as e:  # pragma: no cover - exec rarely fails post-resolve
        raise click.UsageError(f"failed to exec {binary}: {e}")


def _make_binary_command(
    name: str, binary: str, short_help: str, install_hint: str
) -> click.Command:
    """A click command that forwards every arg to `binary` and execs it.

    `add_help_option=False` + `ignore_unknown_options` means `nova storm --help`
    (and every flag) passes straight through to the binary's own parser.
    """

    @click.command(
        name=name,
        short_help=short_help,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def _cmd(args: tuple[str, ...]) -> None:
        _exec_binary(binary, install_hint, args)

    return _cmd


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


# Register the Rust-backed commands eagerly. They're cheap (no heavy imports),
# and being real `self.commands` they show up in `nova --help` with their
# short_help and dispatch to the binary on invocation.
for _name, (_binary, _short, _hint) in _BINARY_COMMANDS.items():
    main.add_command(_make_binary_command(_name, _binary, _short, _hint))

# The `dist` subgroup is itself a LazyGroup, so `nova --help` and `nova dist
# --help` stay fast — the orchestrator modules (which import SkyPilot) only load
# when a `nova dist <tool>` actually runs.
main.add_command(
    LazyGroup(
        name="dist",
        lazy_commands=_DIST_COMMANDS,
        help="Orchestrate a tool across a SkyPilot fleet (embed / load / storm).",
    )
)


if __name__ == "__main__":
    main()
