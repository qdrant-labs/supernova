"""`nova sweep <config>` — exec'd by the `nova` dispatcher as `nova-sweep`."""

from __future__ import annotations

import logging
import sys

import click

from nova_sweep.config import load_config
from nova_sweep.report import write_report
from nova_sweep.runner import SweepAbort, collection_exists, run_sweep
from nova_sweep.slices import build_slices


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nova_sweep").setLevel(logging.INFO)


def _print_dry_run(cfg, slices, skip_insert: bool) -> None:
    total_points = 0
    for slc in slices:
        exists = collection_exists(cfg, slc.collection_name)
        points = len(slc.index_variants) * len(slc.searches)
        total_points += points
        needs_flag = exists and cfg.target.recreate == "never" and not skip_insert
        status = "exists" if exists else "new"
        click.echo(
            f"{slc.data_layout_name}: collection={slc.collection_name} ({status}) "
            f"index_variants={len(slc.index_variants)} searches={len(slc.searches)} "
            f"points={points}"
            + ("  [would ERROR — needs --skip-insert or target.recreate: always]" if needs_flag else "")
        )
    click.echo(f"\n{len(slices)} data_layouts, {total_points} total points")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("config")
@click.option(
    "--skip-insert",
    is_flag=True,
    help="Reuse a pre-existing same-named collection instead of erroring.",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Delete every collection this run inserted into, after it finishes.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the expanded slice list and collision check; execute nothing.",
)
def main(config: str, skip_insert: bool, cleanup: bool, dry_run: bool) -> None:
    """Sweep a matrix of nova-load/nova-storm index and search configs."""
    _setup_logging()

    cfg = load_config(config)
    slices = build_slices(cfg)

    if dry_run:
        _print_dry_run(cfg, slices, skip_insert)
        return

    try:
        rows = run_sweep(cfg, slices, skip_insert=skip_insert, cleanup=cleanup)
    except SweepAbort as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)

    dest = write_report(cfg.output.path, rows)
    ok_count = sum(1 for r in rows if r["ok"])
    click.echo(f"wrote {len(rows)} rows ({ok_count} ok, {len(rows) - ok_count} errors) to {dest}")


if __name__ == "__main__":
    main()
