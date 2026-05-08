import hashlib
import uuid

import boto3

# Re-export for callers that imported EVAL_SUBDIR from here historically.
# All new code should import from vectorforge.destinations instead.
from vectorforge.destinations import EVAL_SUBDIR  # noqa: F401


def discover_corpus_parquets(bucket: str, prefix: str) -> list[str]:
    """
    Legacy S3-only shim. Returns bare S3 keys (no scheme, no bucket).

    DEPRECATED: prefer ``vectorforge.destinations.discover_corpus_parquets``
    which accepts a Destination and returns absolute URIs across schemes.
    Kept here so the brute-force / generate-queries / push-hf paths that
    haven't been migrated yet keep working.
    """
    from vectorforge.destinations import S3Destination
    from vectorforge.destinations import discover_corpus_parquets as _new

    dest = S3Destination(bucket=bucket, prefix=prefix.rstrip("/"))
    scheme_prefix = f"s3://{bucket}/"
    return [u[len(scheme_prefix):] for u in _new(dest)]


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


def get_bucket_region(bucket: str) -> str:
    """Return the AWS region of an S3 bucket."""
    resp = boto3.client("s3").get_bucket_location(Bucket=bucket)
    return resp["LocationConstraint"] or "us-east-1"


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