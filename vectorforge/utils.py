import hashlib
import uuid

import boto3

# Subdirectory under any corpus prefix that holds eval artifacts (queries,
# brute-force results, partial results). Excluded from all corpus listings so
# pipeline tools never accidentally process eval files as corpus data.
EVAL_SUBDIR = "eval"


def discover_corpus_parquets(bucket: str, prefix: str) -> list[str]:
    """
    List all corpus parquet files under bucket/prefix, excluding eval/ artifacts.

    Excludes any key containing /{EVAL_SUBDIR}/ as a path component, regardless
    of how deep eval/ sits relative to prefix. This means the exclusion works
    whether eval/ is at the slice level (prefix/eval/) or at a parent level
    (embedder/eval/) when globbing across multiple slices.

    Returns bare S3 keys (no s3:// scheme, no bucket). Callers that need full
    URIs should prepend f"s3://{bucket}/".
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    eval_segment = f"/{EVAL_SUBDIR}/"
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet") and eval_segment not in key:
                keys.append(key)
    return sorted(keys)


def s3_rel_key(key: str, bucket: str, prefix: str) -> str:
    """
    Strip bucket and prefix from an S3 key, returning only the relative path.

    Accepts both bare keys and full s3:// URIs:
      s3_rel_key("s3://my-bucket/my/prefix/rank00/batch_0.parquet", "my-bucket", "my/prefix")
      s3_rel_key("my-bucket/my/prefix/rank00/batch_0.parquet",       "my-bucket", "my/prefix")
      -> "rank00/batch_0.parquet"
    """
    # Normalise: strip scheme and bucket so we're working with just the key path
    if key.startswith("s3://"):
        key = key[5:]
    if key.startswith(bucket + "/"):
        key = key[len(bucket) + 1:]

    # Strip the prefix (with trailing slash)
    stripped = prefix.rstrip("/") + "/"
    if key.startswith(stripped):
        key = key[len(stripped):]

    return key


def make_point_id(source_file: str, source_row: int) -> str:
    """
    Stable, deterministic point ID used across the pipeline:
      - brute-force nearest-neighbor results
      - Qdrant loader (id_expression)
      - query provenance in generate-queries output

    Returns a UUID string derived from md5(source_file:source_row).
    UUID format is required by Qdrant (it rejects arbitrary strings).
    The value is still fully deterministic — same inputs always produce the same ID.
    """
    md5 = hashlib.md5(f"{source_file}:{source_row}".encode()).hexdigest()
    return str(uuid.UUID(md5))