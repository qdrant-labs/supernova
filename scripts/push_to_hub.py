#!/usr/bin/env python3
"""
Copy pre-embedded parquet files from S3 to a HuggingFace Hub dataset.

Streams one file at a time (download → upload → delete) so disk usage stays
flat regardless of how many files there are. Already-uploaded files are
skipped, so the script is safe to re-run after interruption.

Install hf_transfer for ~5x faster uploads:
  pip install hf_transfer
  export HF_HUB_ENABLE_HF_TRANSFER=1

Usage:
  python scripts/push_to_hub.py s3://qdrant--vectorforge/fineweb/embedder-bge-large-en-v1.5/ \\
      nleroy917/fineweb-bge-large-en-v1.5 \\
      --private \\
      --subfolder data

  python scripts/push_to_hub.py s3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/ \\
      qdrant/ccnews-bge-large-en \\
      --subfolder data/train
"""

import argparse
import os
import tempfile
from pathlib import Path

import boto3
from huggingface_hub import HfApi


def list_s3_parquets(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return sorted(keys)


def already_uploaded(api: HfApi, repo_id: str, path_in_repo: str) -> bool:
    return api.file_exists(repo_id=repo_id, filename=path_in_repo, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Copy S3 parquets to HuggingFace Hub")
    parser.add_argument("s3_uri", help="s3://bucket/prefix (all .parquet files under prefix are uploaded)")
    parser.add_argument("repo_id", help="HF repo id, e.g. 'username/dataset-name'")
    parser.add_argument("--subfolder", default="data", help="Folder inside the HF repo (default: data)")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip files already present in the repo (default: True)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    if not args.s3_uri.startswith("s3://"):
        parser.error("s3_uri must start with s3://")
    without_scheme = args.s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    api = HfApi()

    print(f"Creating repo {args.repo_id!r} (if it doesn't exist)...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    print(f"Listing parquet files at s3://{bucket}/{prefix}/...")
    keys = list_s3_parquets(bucket, prefix)
    if not keys:
        print("No parquet files found.")
        return
    print(f"Found {len(keys)} parquet files")

    s3 = boto3.client("s3")
    skipped = 0
    uploaded = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, key in enumerate(keys):
            # Preserve S3 subdirectory structure relative to the given prefix.
            # e.g. prefix="fineweb/embed-bge" key="fineweb/embed-bge/cc-main/rank00/batch_0.parquet"
            #   -> relative="cc-main/rank00/batch_0.parquet"
            #   -> path_in_repo="data/cc-main/rank00/batch_0.parquet"
            relative = key[len(prefix):].lstrip("/")
            path_in_repo = f"{args.subfolder}/{relative}"

            if args.skip_existing and already_uploaded(api, args.repo_id, path_in_repo):
                print(f"[{i+1}/{len(keys)}] Skipping (already uploaded): {relative}")
                skipped += 1
                continue

            # Use a flat temp filename to avoid needing to recreate subdirs locally
            local_path = os.path.join(tmpdir, Path(key).name)

            print(f"[{i+1}/{len(keys)}] Downloading s3://{bucket}/{key}...")
            s3.download_file(bucket, key, local_path)
            size_gb = os.path.getsize(local_path) / 1e9
            print(f"  Downloaded {size_gb:.2f} GB — uploading to {path_in_repo}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Add {relative}",
            )

            os.remove(local_path)
            uploaded += 1
            print(f"  Done [{uploaded} uploaded, {skipped} skipped]")

    print(f"\nFinished: {uploaded} uploaded, {skipped} skipped")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()