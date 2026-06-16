#!/usr/bin/env python3
"""`nova experiment` — compose units over a timeline against one shared subject.

The local-first spine (V1): a declarative timeline of steps, each shelling a unit
CLI (`storm-dist`, `load`, ...) with `NOVA_EXPERIMENT_ID` forwarded so every child
run is grouped under one experiment. Phase events become Grafana annotations.

Config shape:

    experiment:
      name: write_contention          # optional; used to mint the experiment id
      metrics:                         # where phase events + child runs are linked
        type: postgres
        dsn: ${SN_METRICS_DB_URL}
      cooldown_s: 120                  # keep observing after the last step
      steps:
        - id: reads
          run: storm-dist              # launcher: dispatches a fleet, returns
          config: configs/storm/poshmark.yaml
          launcher: true
        - id: writes
          run: load                    # foreground: runs to completion
          config: ~/.nova/poshmark.yaml
          delay_s: 180                 # warmup: let reads stabilize first
"""

import logging
import os

import click
import yaml

from supernova.metrics import build_metrics, generate_run_name, make_run_id, set_current
from supernova.experiment.runner import run_experiment

logger = logging.getLogger(__name__)


@click.command(name="experiment", help="Compose units over a timeline (workload tests).")
@click.argument("config")
@click.option("--dry-run", is_flag=True, help="Parse + print the plan, launch nothing.")
def experiment(config, dry_run):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    from supernova.cli.config_resolve import resolve_config

    with open(config) as f:
        cfg = resolve_config(yaml.safe_load(f))

    exp = cfg["experiment"]
    steps = exp.get("steps") or []
    if not steps:
        raise click.UsageError("experiment.steps is empty — nothing to run")

    base_name = exp.get("name") or generate_run_name()
    experiment_id = os.environ.get("NOVA_EXPERIMENT_ID") or make_run_id(base_name)
    cooldown = exp.get("cooldown_s", 0)

    # Normalize step keys (delay_s -> delay) so the runner stays terse, and
    # validate dispositions up front (loud over silent — see the Qdrant params guard).
    known = {"id", "run", "config", "launcher", "background", "delay_s", "timeout_s", "args"}
    norm_steps = []
    for s in steps:
        unknown = set(s) - known
        if unknown:
            raise click.UsageError(
                f"step {s.get('id', '?')!r}: unknown keys {sorted(unknown)}. Valid: {sorted(known)}"
            )
        if s.get("launcher") and s.get("background"):
            raise click.UsageError(
                f"step {s['id']!r}: 'launcher' and 'background' are mutually exclusive"
            )
        if s.get("launcher") and not str(s["run"]).endswith("-dist"):
            click.echo(
                f"  warning: step {s['id']!r} is 'launcher' but '{s['run']}' isn't a *-dist "
                f"dispatcher — a local command runs in the foreground and will block the "
                f"timeline (reads won't overlap writes). Did you mean 'background: true'?",
                err=True,
            )
        norm_steps.append(
            {
                "id": s["id"],
                "run": s["run"],
                # Paths in the YAML aren't shell-expanded; do it here so child
                # units receive a real path (open() doesn't grok '~').
                "config": os.path.expanduser(s["config"]),
                "launcher": bool(s.get("launcher")),
                "background": bool(s.get("background")),
                "delay": s.get("delay_s", 0),
                "timeout_s": s.get("timeout_s"),
                "args": s.get("args") or [],
            }
        )

    if dry_run:
        click.echo("=" * 60)
        click.echo(f"experiment: {experiment_id}")
        click.echo(f"subject:    {exp.get('subject', '(unspecified)')}")
        for s in norm_steps:
            disp = "background" if s["background"] else "launcher" if s["launcher"] else "foreground"
            cap = f" timeout={s['timeout_s']}s" if s["timeout_s"] else ""
            extra = (" " + " ".join(s["args"])) if s["args"] else ""
            click.echo(
                f"  + delay {s['delay']:>4}s  {s['id']:<10} {disp:<10}{cap}  "
                f"nova {s['run']} {s['config']}{extra}"
            )
        click.echo(f"  + cooldown {cooldown}s")
        click.echo("=" * 60)
        return

    # The orchestrator itself IS a run (run_id == experiment_id); child runs point
    # back via experiment_id, and phase events land under this run.
    metrics_backend = build_metrics(exp.get("metrics"))
    set_current(metrics_backend)
    metrics_backend.init()
    metrics_backend.start(experiment_id, {"command": "experiment", "config": cfg})
    logger.info("experiment run: %s", experiment_id)

    # Children inherit the experiment id; they each mint their own run_id but tag
    # this experiment so Grafana can group them.
    child_env = dict(os.environ)
    child_env["NOVA_EXPERIMENT_ID"] = experiment_id

    status = "ok"
    try:
        status = run_experiment(experiment_id, norm_steps, cooldown=cooldown, env=child_env)
    finally:
        metrics_backend.finish(status)
        logger.info("experiment %s finished: %s", experiment_id, status)


if __name__ == "__main__":
    experiment()