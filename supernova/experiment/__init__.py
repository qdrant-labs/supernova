"""Experiment orchestration — compose units over a timeline against one subject.

See ``runner.run_experiment``. The CLI lives in ``supernova.cli.run_experiment``.
"""

from .runner import run_experiment

__all__ = ["run_experiment"]