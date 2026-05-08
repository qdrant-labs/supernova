#!/usr/bin/env python3
"""
Upload S3 parquet files to a HuggingFace Hub dataset.

Downloads each file from S3 to local disk, uploads to HF Hub, then deletes
the local copy. Safe to re-run — already-uploaded files are skipped.

In distributed mode (--num-jobs), each worker takes a round-robin slice of
the full file list by rank. Rank is read from $SKYPILOT_JOB_RANK when not
passed explicitly.
"""

import logging
import os
import tempfile
from pathlib import Path

import boto3
import click
from huggingface_hub import HfApi, CommitOperationAdd

from vectorforge.destinations import S3Destination, discover_corpus_parquets

logger = logging.getLogger(__name__)


def list_s3_parquets(bucket: str, prefix: str) -> list[str]:
    """Return bare S3 keys (no scheme, no bucket) for every corpus parquet."""
    dest = S3Destination(bucket=bucket, prefix=prefix.rstrip("/"))
    scheme_prefix = f"s3://{bucket}/"
    return [u[len(scheme_prefix) :] for u in discover_corpus_parquets(dest)]


@click.command(name="push-hf", help="Upload S3 parquets to a HuggingFace Hub dataset.")
@click.argument("s3_uri")
@click.argument("repo_id")
@click.option(
    "--subfolder", default="data", show_default=True, help="Folder inside the HF repo."
)
@click.option("--private", is_flag=True, help="Create repo as private.")
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip files already present in the repo.",
)
@click.option(
    "--num-jobs",
    type=int,
    default=None,
    help="Total parallel jobs — each takes a round-robin slice of files.",
)
@click.option(
    "--job-rank",
    type=int,
    default=None,
    help="This job's rank (0-indexed). Defaults to $SKYPILOT_JOB_RANK.",
)
@click.option(
    "--commit-batch-size",
    type=int,
    default=10,
    show_default=True,
    help="Files per HF commit. HF limits to 128 commits/hr.",
)
def push_hf(
    s3_uri,
    repo_id,
    subfolder,
    private,
    skip_existing,
    num_jobs,
    job_rank,
    commit_batch_size,
):
    """Upload S3 parquets to a HuggingFace Hub dataset."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    if not s3_uri.startswith("s3://"):
        raise click.UsageError("s3_uri must start with s3://")
    without_scheme = s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    api = HfApi()

    # Only the coordinator (rank 0 or single-machine) creates the repo.
    if num_jobs is not None and job_rank is None:
        job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))

    if num_jobs is None or job_rank == 0:
        logger.info("Creating repo %r (if it doesn't exist)...", repo_id)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
        )

    logger.info("Listing parquet files at s3://%s/%s/...", bucket, prefix)
    all_keys = list_s3_parquets(bucket, prefix)
    if not all_keys:
        click.echo("No parquet files found.")
        return

    # Shard by rank if running distributed
    if num_jobs is not None:
        keys = [k for i, k in enumerate(all_keys) if i % num_jobs == job_rank]
        logger.info(
            "Rank %d/%d: %d files (of %d total)",
            job_rank,
            num_jobs,
            len(keys),
            len(all_keys),
        )
    else:
        keys = all_keys
        click.echo(f"Found {len(keys)} parquet files")

    if not keys:
        logger.info("No files assigned to this rank, exiting.")
        return

    s3 = boto3.client("s3")
    skipped = 0
    uploaded = 0
    batch_num = 0

    # Batch files into groups to stay within HF's 128 commits/hr limit.
    # Each group is downloaded, committed in one shot, then deleted.
    batch_size = commit_batch_size

    with tempfile.TemporaryDirectory() as tmpdir:
        pending: list[tuple[str, str, str]] = []  # (key, relative, local_path)

        def flush_batch():
            nonlocal uploaded, batch_num
            if not pending:
                return
            operations = [
                CommitOperationAdd(
                    path_in_repo=f"{subfolder}/{rel}", path_or_fileobj=lp
                )
                for _, rel, lp in pending
            ]
            batch_num += 1
            logger.info("Committing batch %d (%d files)...", batch_num, len(operations))
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add {len(operations)} files (batch {batch_num})",
            )
            for _, _, lp in pending:
                os.remove(lp)
            uploaded += len(pending)
            pending.clear()
            logger.info(
                "  Committed [%d uploaded, %d skipped so far]", uploaded, skipped
            )

        for i, key in enumerate(keys):
            relative = key[len(prefix) :].lstrip("/")
            path_in_repo = f"{subfolder}/{relative}"

            if skip_existing and api.file_exists(
                repo_id=repo_id, filename=path_in_repo, repo_type="dataset"
            ):
                logger.info(
                    "[%d/%d] Skipping (already uploaded): %s",
                    i + 1,
                    len(keys),
                    relative,
                )
                skipped += 1
                continue

            local_path = os.path.join(tmpdir, Path(key).name)
            logger.info(
                "[%d/%d] Downloading s3://%s/%s...", i + 1, len(keys), bucket, key
            )
            s3.download_file(bucket, key, local_path)
            size_gb = os.path.getsize(local_path) / 1e9
            logger.info("  %.2f GB downloaded", size_gb)
            pending.append((key, relative, local_path))

            if len(pending) >= batch_size:
                flush_batch()

        flush_batch()  # commit any remaining files

    click.echo(
        f"Finished: {uploaded} uploaded in {batch_num} commits, {skipped} skipped"
    )
    if num_jobs is None or job_rank == 0:
        click.echo(f"Dataset: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    push_hf()
