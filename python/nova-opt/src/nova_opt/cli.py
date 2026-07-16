"""`nova opt <cmd>` — exec'd by the `nova` dispatcher as `nova-opt`.

    nova opt tune <config.yaml>       run the tuner (live or replay)
    nova opt train-recall             train + LODO-evaluate the recall classifier
    nova opt stats                    compute dataset/query statistics
"""

from __future__ import annotations

import json
import logging
import sys

import click


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nova_opt").setLevel(logging.INFO)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Cost-aware, recall-constrained BO tuner for vector databases."""
    _setup_logging()


@main.command()
@click.argument("config")
@click.option(
    "--dry-run", is_flag=True,
    help="Load models/stats and print the planned setup; evaluate nothing.",
)
def tune(config: str, dry_run: bool) -> None:
    """Run the tuner against a live target (or a replay table)."""
    import pandas as pd

    from nova_opt.config import load_config
    from nova_opt.evaluate import LiveQdrantEvaluator, ReplayEvaluator
    from nova_opt.optimizer import Optimizer
    from nova_opt.recall import load_or_train
    from nova_opt.record import write_run_meta, write_trials
    from nova_opt.stats import compute_workload_stats

    cfg, run_name = load_config(config)

    click.echo(f"[{run_name}] training/loading recall classifier from {cfg.data_csv}")
    clf, lodo = load_or_train(
        data_csv=cfg.data_csv,
        model_dir=cfg.recall_model_dir,
        seed=cfg.optimizer.seed,
    )
    if lodo is not None:
        pooled = {k: v["pooled_auc"] for k, v in clf.lodo_metrics.items()}
        click.echo(f"[{run_name}] LODO pooled AUC per threshold: {pooled}")

    click.echo(f"[{run_name}] computing workload statistics")
    features, provenance = compute_workload_stats(
        corpus_path=cfg.workload.corpus.path,
        corpus_column=cfg.workload.corpus.dense_column,
        queries_path=cfg.workload.queries.uri,
        queries_column=cfg.workload.queries.column,
        distance_metric=cfg.workload.distance_metric,
        params=cfg.stats.params(),
    )
    workload = dict(features)
    workload.update(
        corpus_size=int(features["number_of_embeddings"]),
        query_count=int(features["query_count"]),
        vector_dim=int(features["dimensionality"]),
        distance_metric=cfg.workload.distance_metric,
    )

    if cfg.replay:
        table = (
            pd.read_parquet(cfg.replay)
            if cfg.replay.endswith(".parquet")
            else pd.read_csv(cfg.replay)
        )
        evaluator = ReplayEvaluator(table)
        click.echo(f"[{run_name}] replay mode: {len(table)} measured rows")
    else:
        evaluator = LiveQdrantEvaluator(
            run_name=run_name,
            corpus_path=cfg.workload.corpus.path,
            corpus_column=cfg.workload.corpus.dense_column,
            queries_uri=cfg.workload.queries.uri,
            queries_column=cfg.workload.queries.column,
            ground_truth_column=cfg.workload.queries.ground_truth_column,
            queries_limit=cfg.workload.queries.limit,
            url=cfg.target.url,
            api_key=cfg.target.api_key,
            distance=cfg.workload.distance_metric,
            duration_s=cfg.workload.duration_s,
        )

    space = cfg.space.space()
    settings = cfg.settings()
    if dry_run:
        click.echo(
            f"space size: {space.size()} candidates | strategy={settings.strategy} "
            f"target_recall>={settings.target_recall} budget={settings.budget_seconds}s "
            f"max_evaluations={settings.max_evaluations}"
        )
        click.echo(f"stats provenance: {json.dumps(provenance, indent=2)}")
        return

    opt = Optimizer(
        space=space,
        evaluator=evaluator,
        recall_model=clf,
        workload=workload,
        stats_meta=provenance,
        cost_model=cfg.cost_priors.model(),
        settings=settings,
    )
    trials = opt.run()

    dest = write_trials(cfg.output.path, opt.rows())
    meta = {
        "run_name": run_name,
        "settings": settings.__dict__,
        "stats_provenance": provenance,
        "lodo_metrics": clf.lodo_metrics,
        "spent_seconds": opt.spent_seconds,
        "n_trials": len(trials),
    }
    best = opt.best_feasible()
    if best is not None and best.outcome is not None:
        meta["best_feasible"] = {
            "qps": best.outcome.qps,
            "p95_ms": best.outcome.p95_ms,
            "mean_recall": best.outcome.mean_recall,
            "search_key": repr(best.candidate.search_key),
        }
        click.echo(
            f"best feasible: qps={best.outcome.qps:.1f} "
            f"p95={best.outcome.p95_ms:.2f}ms recall={best.outcome.mean_recall}"
        )
    else:
        click.echo("no feasible configuration found within budget")
    write_run_meta(cfg.output.path, meta)
    ok = sum(1 for t in trials if t.ok)
    click.echo(
        f"wrote {len(trials)} trials ({ok} ok) to {dest}; "
        f"spent {opt.spent_seconds:.0f}s of {settings.budget_seconds:.0f}s budget"
    )


@main.command("train-recall")
@click.option("--data", default="data.csv", show_default=True, help="Training CSV.")
@click.option("--out", required=True, help="Directory to save the trained model.")
@click.option("--report", default=None, help="Optional CSV path for the LODO report.")
@click.option("--seed", default=0, show_default=True)
def train_recall(data: str, out: str, report: str | None, seed: int) -> None:
    """Train the XGBoost recall feasibility classifier with LODO evaluation."""
    import pandas as pd

    from nova_opt.recall import RecallClassifier

    df = pd.read_csv(data)
    clf = RecallClassifier()
    lodo = clf.train(df, seed=seed)
    clf.save(out)
    if report:
        lodo.to_csv(report, index=False)
        click.echo(f"wrote LODO report to {report}")
    for key, m in clf.lodo_metrics.items():
        click.echo(
            f"P(recall >= {key}): pooled AUC {m['pooled_auc']:.3f}, "
            f"brier {m['pooled_brier']:.3f}, positive rate {m['positive_rate']:.2f}"
        )
    click.echo(f"saved model to {out}")


@main.command()
@click.option("--vectors", required=True, help="Base matrix X (.npy or parquet).")
@click.option("--column", default=None, help="Vector column (parquet input).")
@click.option("--queries", required=True, help="Query matrix Q (.npy or parquet).")
@click.option("--query-column", default=None, help="Query vector column (parquet).")
@click.option("--metric", default="cosine", show_default=True,
              help="cosine | euclidean/L2 | dot/IP.")
@click.option("--out", default=None, help="Write features+provenance JSON here.")
@click.option("--sample-size", default=1000, show_default=True)
@click.option("--pair-sample-size", default=100_000, show_default=True)
@click.option("--nn-query-sample-size", default=256, show_default=True)
@click.option("--nn-reference-sample-size", default=5000, show_default=True)
@click.option("--knn-k", default=100, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--full-pass-row-limit", default=2_000_000, show_default=True)
def stats(
    vectors: str, column: str | None, queries: str, query_column: str | None,
    metric: str, out: str | None, sample_size: int, pair_sample_size: int,
    nn_query_sample_size: int, nn_reference_sample_size: int, knn_k: int,
    seed: int, full_pass_row_limit: int,
) -> None:
    """Compute dataset/query statistics for the recall-model feature pipeline."""
    from nova_opt.stats import StatsParams, compute_workload_stats

    params = StatsParams(
        sample_size=sample_size,
        pair_sample_size=pair_sample_size,
        nn_query_sample_size=nn_query_sample_size,
        nn_reference_sample_size=nn_reference_sample_size,
        knn_k=knn_k,
        seed=seed,
        full_pass_row_limit=full_pass_row_limit,
    )
    features, provenance = compute_workload_stats(
        corpus_path=vectors,
        corpus_column=column,
        queries_path=queries,
        queries_column=query_column,
        distance_metric=metric,
        params=params,
    )
    payload = json.dumps(
        {"features": features, "provenance": provenance}, indent=2, default=str
    )
    if out:
        with open(out, "w") as f:
            f.write(payload)
        click.echo(f"wrote stats to {out}")
    else:
        click.echo(payload)


if __name__ == "__main__":
    main()
