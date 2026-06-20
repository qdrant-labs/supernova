"""nova — a git-style dispatcher for the supernova toolset.

`nova <cmd> [args...]` discovers an executable named `nova-<cmd>` on PATH and
*replaces* this process with it via os.execv — so exit codes, signals, and
stdio pass straight through. The dispatcher itself does nothing but route.

Commands can be written in any language: a Rust binary installed by
`cargo install` (e.g. `nova-load`) and a Python console script installed by
`pip` (e.g. `nova-embed`) look identical from here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

PREFIX = "nova-"


def discover() -> dict[str, Path]:
    """Map command name -> executable for every `nova-*` on PATH.

    Earlier PATH entries win, mirroring normal shell resolution.
    """
    found: dict[str, Path] = {}
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if not raw:
            continue
        directory = Path(raw)
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue  # unreadable/nonexistent PATH entry — skip
        for entry in entries:
            name = entry.name
            if not name.startswith(PREFIX):
                continue
            if entry.is_file() and os.access(entry, os.X_OK):
                found.setdefault(name[len(PREFIX):], entry)
    return found


def print_help() -> None:
    print("nova — dispatcher for the supernova toolset\n")
    print("usage: nova <command> [args...]\n")
    commands = discover()
    if commands:
        print("available commands (nova-* on PATH):")
        width = max(len(c) for c in commands)
        for name in sorted(commands):
            print(f"  {name:<{width}}  ({commands[name]})")
    else:
        print("no nova-* commands found on PATH.")
        print("install some, e.g.:  cargo install --path crates/nova-load")
    print("\nother:")
    print("  nova --help / -h     show this help")
    print("  nova --version / -v  show dispatcher version")


def main() -> int:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        print_help()
        return 0
    if argv[0] in ("-v", "--version"):
        print(f"nova {__version__}")
        return 0

    command, rest = argv[0], argv[1:]
    prog = f"{PREFIX}{command}"
    exe = discover().get(command)

    if exe is None:
        print(f"nova: '{command}' is not a nova command (no '{prog}' on PATH).", file=sys.stderr)
        commands = discover()
        if commands:
            print(f"available: {', '.join(sorted(commands))}", file=sys.stderr)
        return 127

    # Replace this process. argv[0] is the conventional program name.
    os.execv(str(exe), [prog, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
