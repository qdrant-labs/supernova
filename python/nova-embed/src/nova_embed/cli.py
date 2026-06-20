"""Entry point for `nova embed` / `nova-embed`.

Drop the existing embedding-generation code into this package and wire it up
here. Kept dependency-light at import time so `--help` is fast; import torch /
sentence-transformers lazily inside the command that needs them.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="nova embed",
        description="Generate embeddings for the supernova toolset.",
    )
    # TODO: port the existing embed CLI here (model, input/output, batch size…).
    parser.parse_args()

    print("nova-embed: not yet implemented — port the embedding code here.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
