#!/usr/bin/env python3
"""
Sample N rows uniformly at random from a parquet corpus and save them
locally as a single parquet file.

Same source surface as `generate-queries` (s3://, hf://buckets/, file://)
and the same range-request / prefetch fetch modes — minus the upload
back to ``{corpus_uri}/eval/``. Always runs in-process.
"""

import logging

import click

from supernova.destinations import parse_destination
from supernova.eval.subsample import subsample as _subsample


@click.command(
    name="subsample",
    help="Sample N random rows from a corpus and save as a local parquet file.",
)
@click.argument("corpus_uri")
@click.option(
    "-n", "--num-rows", "num_rows", type=int, default=1000, show_default=True
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--columns",
    multiple=True,
    default=(),
    help="Columns to fetch (default: all). Repeat for each: "
    "--columns text --columns url",
)
@click.option(
    "--output",
    default=None,
    help="Local output path (default: subsample_<n>.parquet).",
)
@click.option(
    "--prefetch",
    is_flag=True,
    help="Download each parquet fully before reading (better for large row groups).",
)
def subsample(corpus_uri, num_rows, seed, columns, output, prefetch):
    """Sample N random rows from a parquet corpus to a local file."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger("supernova").setLevel(logging.INFO)

    try:
        parse_destination(corpus_uri)
    except ValueError as e:
        raise click.UsageError(str(e))

    if output is None:
        output = f"subsample_{num_rows}.parquet"

    columns_list = list(columns) if columns else None

    abs_path = _subsample(
        corpus_uri=corpus_uri,
        n=num_rows,
        seed=seed,
        columns=columns_list,
        output=output,
        prefetch=prefetch,
    )
    click.echo(f"Wrote {abs_path}")


if __name__ == "__main__":
    subsample()
