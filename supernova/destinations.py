"""
Destination abstraction for corpus + eval artifact storage.

A ``Destination`` is a logical location for a supernova corpus: today
either an S3 bucket+prefix, an HF Storage Bucket, or a local directory.
The same pipelines (embed, load, generate-queries, brute-force) run over any.

URI schemes
-----------
Today:
  s3://bucket/prefix/...
  hf://buckets/namespace/name[/subdir/...]
  file:///abs/path/...

To add another (gs://, az://, bb://) wire it through ``parse_destination``,
``discover_corpus_parquets``, ``filesystem_for_uri``, and ``fs_path_for_uri``.

Conventions
-----------
- Corpus parquets live under the destination's "data root":
    s3://bucket/prefix/rank00/batch_*.parquet
    hf://buckets/owner/name[/subdir]/rank00/batch_*.parquet
- Eval artifacts (queries, brute-force results) live under
  ``{root}/eval/...``:
    s3://bucket/prefix/eval/queries_1000.parquet
    hf://buckets/owner/name/eval/queries_1000.parquet

HF Storage Buckets are HF's S3-like object storage (powered by Xet);
they are non-versioned, mutable, and addressed by ``hf://buckets/...`` rather
than the git-backed ``hf://datasets/...`` scheme. We don't write to dataset
repos anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pyarrow.fs  # noqa: F401


EVAL_SUBDIR = "eval"


@dataclass(frozen=True)
class S3Destination:
    bucket: str
    prefix: str  # may be empty; always trimmed of trailing slash

    @property
    def scheme(self) -> str:
        return "s3"

    @property
    def root_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}".rstrip("/")

    def child_uri(self, sub: str) -> str:
        sub = sub.lstrip("/")
        if not self.prefix:
            return f"s3://{self.bucket}/{sub}"
        return f"s3://{self.bucket}/{self.prefix}/{sub}"

    def eval_uri(self, filename: str) -> str:
        return self.child_uri(f"{EVAL_SUBDIR}/{filename}")


@dataclass(frozen=True)
class LocalDestination:
    root: str  # absolute filesystem path; trimmed of trailing slash

    @property
    def scheme(self) -> str:
        return "file"

    @property
    def root_uri(self) -> str:
        return f"file://{self.root}".rstrip("/")

    def child_uri(self, sub: str) -> str:
        sub = sub.lstrip("/")
        return f"file://{self.root}/{sub}"

    def eval_uri(self, filename: str) -> str:
        return self.child_uri(f"{EVAL_SUBDIR}/{filename}")


@dataclass(frozen=True)
class HfDestination:
    """An HF Storage Bucket, optionally scoped to a sub-path within the bucket."""

    bucket_id: str  # "namespace/bucket-name"
    subdir: str = ""  # path within the bucket (empty = bucket root)

    @property
    def scheme(self) -> str:
        return "hf"

    @property
    def root_uri(self) -> str:
        if self.subdir:
            return f"hf://buckets/{self.bucket_id}/{self.subdir}".rstrip("/")
        return f"hf://buckets/{self.bucket_id}"

    def child_uri(self, sub: str) -> str:
        sub = sub.lstrip("/")
        if self.subdir:
            return f"hf://buckets/{self.bucket_id}/{self.subdir}/{sub}"
        return f"hf://buckets/{self.bucket_id}/{sub}"

    def eval_uri(self, filename: str) -> str:
        # Eval lives under {bucket}/eval/ regardless of subdir, so brute-force
        # / generate-queries always find it at a stable location.
        return f"hf://buckets/{self.bucket_id}/{EVAL_SUBDIR}/{filename}"


Destination = Union[S3Destination, HfDestination, LocalDestination]


def parse_destination(uri: str) -> Destination:
    """
    Parse an s3://, hf://buckets/, or file:// URI into a Destination.

    Accepts:
      s3://bucket
      s3://bucket/prefix/...
      hf://buckets/namespace/name
      hf://buckets/namespace/name/subdir/...
      file:///abs/path[/...]
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"s3:// URI is missing bucket: {uri!r}")
        return S3Destination(bucket=bucket, prefix=prefix.rstrip("/"))

    if uri.startswith("hf://buckets/"):
        rest = uri[len("hf://buckets/") :]
        parts = rest.split("/", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "hf:// URI must be hf://buckets/{namespace}/{name}[/{subdir}], "
                f"got {uri!r}"
            )
        bucket_id = f"{parts[0]}/{parts[1]}"
        subdir = parts[2].rstrip("/") if len(parts) == 3 else ""
        return HfDestination(bucket_id=bucket_id, subdir=subdir)

    if uri.startswith("file://"):
        # Standard form is file:///abs/path (3 slashes = scheme + empty host +
        # absolute path). After stripping "file://" the remainder must start
        # with "/", giving the absolute filesystem path.
        rest = uri[len("file://") :]
        if not rest.startswith("/"):
            raise ValueError(
                f"file:// URI must be absolute (file:///abs/path), got {uri!r}"
            )
        return LocalDestination(root=rest.rstrip("/"))

    raise ValueError(
        f"Unknown URI scheme in {uri!r}. Supported: s3://, hf://buckets/, file://"
    )


def discover_corpus_parquets(dest: Destination) -> list[str]:
    """
    Return absolute URIs for every corpus parquet at ``dest``, excluding
    eval/ artifacts. Sorted for determinism.
    """
    if isinstance(dest, S3Destination):
        return _discover_s3(dest)
    if isinstance(dest, HfDestination):
        return _discover_hf(dest)
    if isinstance(dest, LocalDestination):
        return _discover_local(dest)
    raise ValueError(f"Unknown destination type: {type(dest).__name__}")


def _discover_s3(dest: S3Destination) -> list[str]:
    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    eval_segment = f"/{EVAL_SUBDIR}/"
    uris: list[str] = []
    for page in paginator.paginate(Bucket=dest.bucket, Prefix=dest.prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet") and eval_segment not in key:
                uris.append(f"s3://{dest.bucket}/{key}")
    return sorted(uris)


def _discover_local(dest: LocalDestination) -> list[str]:
    import os

    eval_segment = f"{os.sep}{EVAL_SUBDIR}{os.sep}"
    uris: list[str] = []
    for dirpath, _dirs, files in os.walk(dest.root):
        for name in files:
            if not name.endswith(".parquet"):
                continue
            full = os.path.join(dirpath, name)
            if eval_segment in full:
                continue
            uris.append(f"file://{full}")
    return sorted(uris)


def _discover_hf(dest: HfDestination) -> list[str]:
    from huggingface_hub import list_bucket_tree

    prefix = dest.subdir.rstrip("/") if dest.subdir else None
    eval_segment = f"/{EVAL_SUBDIR}/"
    uris: list[str] = []
    for entry in list_bucket_tree(dest.bucket_id, prefix=prefix, recursive=True):
        path = getattr(entry, "path", None)
        if not path or not path.endswith(".parquet"):
            continue
        # Skip eval artifacts (sibling tree at bucket root). Match either a
        # leading "eval/" or any "/eval/" segment so we don't grab
        # bucket/eval/queries.parquet when the caller scoped to bucket root.
        if path.startswith(f"{EVAL_SUBDIR}/") or eval_segment in f"/{path}":
            continue
        uris.append(f"hf://buckets/{dest.bucket_id}/{path}")
    return sorted(uris)


def filesystem_for_uri(uri: str):
    """
    Return a pyarrow-compatible filesystem object suitable for
    ``pq.read_table(..., filesystem=fs)``.

    For S3, this is ``pyarrow.fs.S3FileSystem``. For HF, this is
    ``huggingface_hub.HfFileSystem`` (fsspec-based; pyarrow accepts
    fsspec filesystems via duck-typing in ParquetFile / read_table).
    For file://, this is ``pyarrow.fs.LocalFileSystem``.
    """
    if uri.startswith("s3://"):
        import pyarrow.fs as pafs

        return pafs.S3FileSystem()
    if uri.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        return HfFileSystem()
    if uri.startswith("file://"):
        import pyarrow.fs as pafs

        return pafs.LocalFileSystem()
    raise ValueError(f"Unknown URI scheme in {uri!r}")


def fs_path_for_uri(uri: str) -> str:
    """
    Strip the scheme from a URI to get the path the filesystem expects.

    pyarrow.fs.S3FileSystem expects:    bucket/key/...
    huggingface_hub.HfFileSystem expects: buckets/owner/name/path
    pyarrow.fs.LocalFileSystem expects: /abs/path
    """
    if uri.startswith("s3://"):
        return uri[len("s3://") :]
    if uri.startswith("hf://"):
        return uri[len("hf://") :]
    if uri.startswith("file://"):
        return uri[len("file://") :]
    raise ValueError(f"Unknown URI scheme in {uri!r}")


def bare_key_for_uri(uri: str) -> str:
    """
    The "bare key" used as the per-file identifier in
    ``make_point_id(bare_key, row_index)``.

    Strips the scheme + bucket-or-bucket-name identifier; keeps everything else.
    Brute-force, generate-queries, and the loader's vf_point_id macro must
    all agree on this form so IDs match across pipelines.

      s3://bucket/prefix/file.parquet           -> prefix/file.parquet
      hf://buckets/ns/name/sub/file.pq          -> sub/file.pq
      file:///abs/path/file.parquet             -> /abs/path/file.parquet

    For file:// the "container" is the filesystem itself, so the bare key is
    the absolute path. IDs are therefore machine-specific — the same corpus
    moved to a different mount point will hash to different IDs. This matches
    the S3/HF behavior (changing bucket also changes IDs).

    --- ID space anchoring (intentional design) ---
    The anchor is the top-level container: the S3 bucket, or the HF
    bucket repo. Two consequences fall out of this choice:

      Stable across SCOPE within a container. Loading just
      `s3://b/fineweb/cc-2025-26/...` and loading the wider
      `s3://b/fineweb/...` give the *same* IDs for the same physical rows,
      because both produce bare keys like `fineweb/cc-2025-26/rank00/x.pq`.
      You can do incremental / partial loads without invalidating earlier
      ground-truth.

      Reset across CONTAINERS. Migrating data S3-bucket-A → S3-bucket-B,
      or S3 → HF, changes the anchor and therefore the IDs. Recall
      ground-truth (queries + brute-force results) must be regenerated on
      the new side; you cannot reuse Qdrant collections across migrations.

    There is no unforced way to make a hash span containers without
    introducing an external "logical dataset id" registry. The current
    choice prioritises the workflow that's actually common (scoped loads
    within one bucket / repo) over the one that's rare (cross-backend
    migrations). See docs/reference/loader-architecture.md#id-space-anchoring.
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        _, _, key = rest.partition("/")
        return key
    if uri.startswith("hf://buckets/"):
        rest = uri[len("hf://buckets/") :]
        # Skip namespace/name; keep everything after.
        parts = rest.split("/", 2)
        return parts[2] if len(parts) == 3 else ""
    if uri.startswith("file://"):
        return uri[len("file://") :]
    raise ValueError(f"Unknown URI scheme in {uri!r}")


def list_parquets_under(prefix_uri: str) -> list[str]:
    """
    List all .parquet files under a URI prefix. Does NOT filter eval/, since
    this is used to enumerate eval artifacts (queries, brute-force partials)
    that live under {root}/eval/.
    """
    if prefix_uri.startswith("s3://"):
        import boto3

        rest = prefix_uri[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        uris: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    uris.append(f"s3://{bucket}/{obj['Key']}")
        return sorted(uris)

    if prefix_uri.startswith("hf://buckets/"):
        from huggingface_hub import list_bucket_tree

        rest = prefix_uri[len("hf://buckets/") :]
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"Bad hf:// prefix: {prefix_uri!r}")
        bucket_id = f"{parts[0]}/{parts[1]}"
        in_bucket_prefix = parts[2] if len(parts) == 3 else None
        uris: list[str] = []
        for entry in list_bucket_tree(
            bucket_id, prefix=in_bucket_prefix, recursive=True
        ):
            path = getattr(entry, "path", None)
            if path and path.endswith(".parquet"):
                uris.append(f"hf://buckets/{bucket_id}/{path}")
        return sorted(uris)

    if prefix_uri.startswith("file://"):
        import os

        root = prefix_uri[len("file://") :]
        uris: list[str] = []
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".parquet"):
                    uris.append(f"file://{os.path.join(dirpath, name)}")
        return sorted(uris)

    raise ValueError(f"Unknown URI scheme in {prefix_uri!r}")


def upload_file_to_uri(local_path: str, dest_uri: str) -> None:
    """
    Upload ``local_path`` to the given destination URI. Synchronous, intended
    for one-shot eval artifact writes (queries, brute-force partials/final).
    """
    if dest_uri.startswith("s3://"):
        import boto3

        rest = dest_uri[len("s3://") :]
        bucket, _, key = rest.partition("/")
        boto3.client("s3").upload_file(local_path, bucket, key)
        return

    if dest_uri.startswith("hf://buckets/"):
        from huggingface_hub import batch_bucket_files

        rest = dest_uri[len("hf://buckets/") :]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"hf:// upload URI needs in-bucket path: {dest_uri!r}")
        bucket_id = f"{parts[0]}/{parts[1]}"
        path_in_bucket = parts[2]
        batch_bucket_files(bucket_id, add=[(local_path, path_in_bucket)])
        return

    if dest_uri.startswith("file://"):
        import os
        import shutil

        target = dest_uri[len("file://") :]
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(local_path, target)
        return

    raise ValueError(f"Unknown URI scheme in {dest_uri!r}")


def upload_bytes_to_uri(data: bytes, dest_uri: str) -> None:
    """Same as upload_file_to_uri but for in-memory bytes."""
    if dest_uri.startswith("s3://"):
        import boto3

        rest = dest_uri[len("s3://") :]
        bucket, _, key = rest.partition("/")
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
        return

    if dest_uri.startswith("hf://buckets/"):
        from huggingface_hub import batch_bucket_files

        rest = dest_uri[len("hf://buckets/") :]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"hf:// upload URI needs in-bucket path: {dest_uri!r}")
        bucket_id = f"{parts[0]}/{parts[1]}"
        path_in_bucket = parts[2]
        batch_bucket_files(bucket_id, add=[(data, path_in_bucket)])
        return

    if dest_uri.startswith("file://"):
        import os

        target = dest_uri[len("file://") :]
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
        return

    raise ValueError(f"Unknown URI scheme in {dest_uri!r}")


def datasource_to_destination(ds_cfg: dict) -> Destination:
    """
    Build a Destination from the loader's ``datasource:`` config block.

    This is *only* used for discovery/sharding of file lists — the underlying
    DataReader (S3DataReader / HuggingFaceDataReader) still owns its own
    config keys.
    """
    ds_type = ds_cfg.get("type", "s3")
    if ds_type == "s3":
        return S3Destination(
            bucket=ds_cfg["bucket"],
            prefix=ds_cfg.get("prefix", "").rstrip("/"),
        )
    if ds_type == "huggingface":
        bucket_id = ds_cfg.get("bucket_id") or ds_cfg.get("repo_id")
        if not bucket_id:
            raise ValueError(
                "datasource type='huggingface' requires 'bucket_id' (HF bucket like 'owner/name')"
            )
        return HfDestination(
            bucket_id=bucket_id,
            subdir=ds_cfg.get("subdir") or "",
        )
    if ds_type == "local":
        root = ds_cfg.get("root") or ds_cfg.get("path")
        if not root:
            raise ValueError("datasource type='local' requires 'root' (or 'path')")
        return LocalDestination(root=root.rstrip("/"))
    raise ValueError(
        f"Unknown datasource.type for discovery: {ds_type!r}. "
        f"Supported: 's3', 'huggingface', 'local'."
    )