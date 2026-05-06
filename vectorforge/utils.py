import hashlib
import uuid


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