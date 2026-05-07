#!/usr/bin/env python3
"""
Upload S3 parquet files to a HuggingFace Hub dataset.

Downloads each file from S3 to local disk, uploads to HF Hub, then deletes
the local copy. Safe to re-run — already-uploaded files are skipped.

In distributed mode (--num-jobs), each worker takes a round-robin slice of
the full file list by rank. Rank is read from $SKYPILOT_JOB_RANK when not
passed explicitly.

Usage:
  vf push-hf s3://bucket/prefix username/dataset-name
  vf push-hf s3://bucket/prefix username/dataset-name --private
  vf push-hf s3://bucket/prefix username/dataset-name --num-jobs 50
  vf push-hf s3://bucket/prefix username/dataset-name --num-jobs 50 --job-rank 3
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import boto3
from huggingface_hub import HfApi, CommitOperationAdd

logger = logging.getLogger(__name__)


def list_s3_parquets(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet") and "/eval/" not in obj["Key"]:
                keys.append(obj["Key"])
    return sorted(keys)


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Upload S3 parquets to HuggingFace Hub")
    parser.add_argument("s3_uri", help="s3://bucket/prefix — all .parquet files under this prefix")
    parser.add_argument("repo_id", help="HF repo id, e.g. 'username/dataset-name'")
    parser.add_argument("--subfolder", default="data", help="Folder inside the HF repo (default: data)")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=True,
                        help="Re-upload files already present in the repo")
    parser.add_argument("--num-jobs", type=int, default=None,
                        help="Total parallel jobs — each takes a round-robin slice of files")
    parser.add_argument("--job-rank", type=int, default=None,
                        help="This job's rank (0-indexed). Defaults to $SKYPILOT_JOB_RANK")
    parser.add_argument("--commit-batch-size", type=int, default=10,
                        help="Files per HF commit (default: 10). HF limits to 128 commits/hr.")
    args = parser.parse_args(argv)

    if not args.s3_uri.startswith("s3://"):
        parser.error("s3_uri must start with s3://")
    without_scheme = args.s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    api = HfApi()

    # Only the coordinator (rank 0 or single-machine) creates the repo.
    job_rank = None
    if args.num_jobs is not None:
        job_rank = args.job_rank
        if job_rank is None:
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))

    if job_rank is None or job_rank == 0:
        logger.info("Creating repo %r (if it doesn't exist)...", args.repo_id)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
        )

    logger.info("Listing parquet files at s3://%s/%s/...", bucket, prefix)
    all_keys = list_s3_parquets(bucket, prefix)
    if not all_keys:
        print("No parquet files found.")
        return

    # Shard by rank if running distributed
    if args.num_jobs is not None:
        keys = [k for i, k in enumerate(all_keys) if i % args.num_jobs == job_rank]
        logger.info(
            "Rank %d/%d: %d files (of %d total)",
            job_rank, args.num_jobs, len(keys), len(all_keys),
        )
    else:
        keys = all_keys
        print(f"Found {len(keys)} parquet files")

    if not keys:
        logger.info("No files assigned to this rank, exiting.")
        return

    s3 = boto3.client("s3")
    skipped = 0
    uploaded = 0
    batch_num = 0

    # Batch files into groups to stay within HF's 128 commits/hr limit.
    # Each group is downloaded, committed in one shot, then deleted.
    batch_size = args.commit_batch_size

    with tempfile.TemporaryDirectory() as tmpdir:
        pending: list[tuple[str, str, str]] = []  # (key, relative, local_path)

        def flush_batch():
            nonlocal uploaded, batch_num
            if not pending:
                return
            operations = [
                CommitOperationAdd(path_in_repo=f"{args.subfolder}/{rel}", path_or_fileobj=lp)
                for _, rel, lp in pending
            ]
            batch_num += 1
            logger.info("Committing batch %d (%d files)...", batch_num, len(operations))
            api.create_commit(
                repo_id=args.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add {len(operations)} files (batch {batch_num})",
            )
            for _, _, lp in pending:
                os.remove(lp)
            uploaded += len(pending)
            pending.clear()
            logger.info("  Committed [%d uploaded, %d skipped so far]", uploaded, skipped)

        for i, key in enumerate(keys):
            relative = key[len(prefix):].lstrip("/")
            path_in_repo = f"{args.subfolder}/{relative}"

            if args.skip_existing and api.file_exists(
                repo_id=args.repo_id, filename=path_in_repo, repo_type="dataset"
            ):
                logger.info("[%d/%d] Skipping (already uploaded): %s", i + 1, len(keys), relative)
                skipped += 1
                continue

            local_path = os.path.join(tmpdir, Path(key).name)
            logger.info("[%d/%d] Downloading s3://%s/%s...", i + 1, len(keys), bucket, key)
            s3.download_file(bucket, key, local_path)
            size_gb = os.path.getsize(local_path) / 1e9
            logger.info("  %.2f GB downloaded", size_gb)
            pending.append((key, relative, local_path))

            if len(pending) >= batch_size:
                flush_batch()

        flush_batch()  # commit any remaining files

    print(f"Finished: {uploaded} uploaded in {batch_num} commits, {skipped} skipped")
    if job_rank is None or job_rank == 0:
        print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()