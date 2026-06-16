"""Experiment runner — drive a timeline of steps against one shared subject.

An experiment composes the existing units (`storm`/`storm-dist`/`load`/...) over
time so you can observe how ONE subject behaves under a sequence of workloads —
e.g. steady reads, then a heavy write burst, then recovery. It is NOT a job
scheduler: steps are expected to share a subject (see issue #11). Unrelated
co-scheduled work doesn't belong here.

Composition is by subprocess over each unit's CLI contract — the runner shells
`nova <run> <config>`, forwarding ``NOVA_EXPERIMENT_ID`` so every child run is
tagged with this experiment, and emits phase events that become Grafana
annotations. It never imports a unit; the unit boundary is a process boundary.

Each step has one of three dispositions:
  * foreground (default) — run to completion; the runner brackets it with
    ``<id>:start`` / ``<id>:end``. Optional ``timeout_s`` caps it into a bounded
    burst (SIGINT then kill), reported as a clean end, not an error. (e.g. local
    ``load`` capped to a write burst.)
  * launcher (``launcher: true``) — the subprocess returns once the work is
    *dispatched* and keeps running elsewhere (e.g. ``storm-dist`` launches a fleet
    that runs on its own ``duration_s``). The runner waits for that fast exit and
    emits ``<id>:launched``; it has no handle on the remote work's lifetime.
  * background (``background: true``) — launched concurrently (``Popen``, not
    waited on) and torn down (SIGINT) at experiment end, emitting ``<id>:end``
    then. This is how a *local* long-running reader overlaps a foreground writer.

storm-dist gets concurrency for free as a launcher (it backgrounds its own
fleet); a local ``storm`` reader needs ``background: true`` to overlap writes.
"""

import logging
import signal
import subprocess
import sys
import time

from supernova import metrics

logger = logging.getLogger(__name__)

_STOP_GRACE_S = 30  # SIGINT then wait this long before SIGKILL


def _run_argv(run: str) -> list[str]:
    """Map a step's ``run`` token to ``nova`` argv. The ``-dist`` dispatchers live
    under the ``dist`` subgroup (``nova dist storm``), so ``storm-dist`` ->
    ``["dist", "storm"]``; a plain local unit (``load``, ``storm``) passes through.
    """
    if run.endswith("-dist"):
        return ["dist", run[: -len("-dist")]]
    return [run]


def _sleep(seconds: float, label: str) -> None:
    if seconds and seconds > 0:
        logger.info("experiment: %s — sleeping %.0fs", label, seconds)
        time.sleep(seconds)


def _stop(proc: subprocess.Popen) -> None:
    """Stop a running child gracefully: SIGINT (so the unit can flush metrics /
    mark its run finished), then SIGKILL if it doesn't exit in time."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_experiment(experiment_id: str, steps: list[dict], cooldown: float = 0.0, env: dict | None = None) -> str:
    """Execute ``steps`` in order, emitting phase events tagged with the experiment.

    Returns the final status ("ok" / "error" / "interrupted"). Assumes the caller
    has built + started a metrics backend with ``run_id == experiment_id`` and
    forwarded ``NOVA_EXPERIMENT_ID`` via ``env``.
    """
    status = "ok"
    background: list[tuple[str, subprocess.Popen]] = []

    def _teardown_background() -> None:
        for step_id, proc in background:
            if proc.poll() is not None:
                metrics.event(f"{step_id}:end", step=step_id, note="self-exited")
                continue
            logger.info("experiment: stopping background step %r", step_id)
            _stop(proc)
            metrics.event(f"{step_id}:end", step=step_id)

    try:
        for step in steps:
            step_id = step["id"]
            _sleep(step.get("delay", 0), f"before {step_id}")

            cmd = [sys.executable, "-m", "supernova.cli.cli", *_run_argv(step["run"]), step["config"]]
            cmd += [str(a) for a in (step.get("args") or [])]  # extra unit flags, e.g. --no-manage-indexing
            disposition = (
                "background" if step.get("background")
                else "launcher" if step.get("launcher")
                else "foreground"
            )
            metrics.event(f"{step_id}:start", step=step_id, run=step["run"], disposition=disposition)
            logger.info("experiment: step %r [%s] -> %s", step_id, disposition, " ".join(cmd[2:]))

            # stdio inherited so SkyPilot/tqdm output streams live; SIGINT reaches
            # children via the shared process group.
            proc = subprocess.Popen(cmd, env=env)

            if disposition == "background":
                background.append((step_id, proc))
                continue  # concurrent — torn down after the foreground timeline

            timeout = step.get("timeout_s") if disposition == "foreground" else None
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # deliberate cap on the burst — a clean stop, not an error.
                _stop(proc)
                metrics.event(f"{step_id}:end", step=step_id, note=f"timeout_s={timeout}")
                logger.info("experiment: step %r hit timeout_s=%s — stopped", step_id, timeout)
                continue

            if rc != 0:
                metrics.event(f"{step_id}:error", step=step_id, returncode=rc)
                logger.error("experiment: step %r exited %d — aborting", step_id, rc)
                status = "error"
                break

            metrics.event(
                f"{step_id}:{'launched' if disposition == 'launcher' else 'end'}", step=step_id
            )

        if status == "ok":
            _sleep(cooldown, "cooldown")
    except KeyboardInterrupt:
        status = "interrupted"
        metrics.event("experiment:interrupted")
        logger.warning("experiment: interrupted")
    finally:
        _teardown_background()
    return status