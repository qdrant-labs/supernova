"""
Destination abstraction for corpus + eval artifact storage.

A ``Destination`` is a logical location for a vectorforge corpus: today
either an S3 bucket+prefix or a HuggingFace dataset repo. The same pipelines
(embed, load, generate-queries, brute-force) run over either.

URI schemes
-----------
Today:
  s3://bucket/prefix/...
  hf://datasets/namespace/repo[/subdir/...]

To add another (gs://, az://, bb://) wire it through ``parse_destination``,
``discover_corpus_parquets``, ``filesystem_for_uri``, and ``fs_path_for_uri``.

Conventions
-----------
- Corpus parquets live under the destination's "data root":
    s3://bucket/prefix/rank00/batch_*.parquet
    hf://datasets/repo/data/rank00/batch_*.parquet  (HF auto-detects data/)
  Callers that read from S3 see flat keys; for HF the StorageBackend hides
  the data/ prefix when *writing* (vectorforge.storage.huggingface), and
  the discovery here surfaces only data/ files when *listing*.
- Eval artifacts (queries, brute-force results) live under
  ``{root}/eval/...``:
    s3://bucket/prefix/eval/queries_1000.parquet
    hf://datasets/repo/eval/queries_1000.parquet      (NOT data/eval/ —
                                                       that would make
                                                       load_dataset() try
                                                       to read evals as rows)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pyarrow.fs


EVAL_SUBDIR = "eval"
HF_DATA_SUBDIR = "data"
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
class HfDestination:
    repo_id: str  # "namespace/dataset-name"
    subdir: str = ""  # path within the data/ root (empty = the whole data/ tree)

    @property
    def scheme(self) -> str:
        return "hf"

    @property
    def root_uri(self) -> str:
        if self.subdir:
            return f"hf://datasets/{self.repo_id}/{HF_DATA_SUBDIR}/{self.subdir}".rstrip("/")
        return f"hf://datasets/{self.repo_id}"

    def child_uri(self, sub: str) -> str:
        """URI for a corpus parquet (lives under data/)."""
        sub = sub.lstrip("/")
        if self.subdir:
            return f"hf://datasets/{self.repo_id}/{HF_DATA_SUBDIR}/{self.subdir}/{sub}"
        return f"hf://datasets/{self.repo_id}/{HF_DATA_SUBDIR}/{sub}"

    def eval_uri(self, filename: str) -> str:
        # eval lives at repo root, sibling of data/ — so load_dataset() won't
        # try to read it as rows.
        return f"hf://datasets/{self.repo_id}/{EVAL_SUBDIR}/{filename}"


Destination = Union[S3Destination, HfDestination]

def parse_destination(uri: str) -> Destination:
    """
    Parse an s3:// or hf://datasets/ URI into a Destination.

    Accepts:
      s3://bucket
      s3://bucket/prefix/...
      hf://datasets/namespace/name
      hf://datasets/namespace/name/subdir/...

    For hf:// any segments past ``namespace/name`` are treated as a *data/*
    sub-tree (i.e. they reference paths under data/, not the repo root).
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"s3:// URI is missing bucket: {uri!r}")
        return S3Destination(bucket=bucket, prefix=prefix.rstrip("/"))

    if uri.startswith("hf://datasets/"):
        rest = uri[len("hf://datasets/"):]
        parts = rest.split("/", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "hf:// URI must be hf://datasets/{namespace}/{name}[/{subdir}], "
                f"got {uri!r}"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        subdir = parts[2].rstrip("/") if len(parts) == 3 else ""
        return HfDestination(repo_id=repo_id, subdir=subdir)

    raise ValueError(
        f"Unknown URI scheme in {uri!r}. Supported: s3://, hf://datasets/"
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


def _discover_hf(dest: HfDestination) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    paths = api.list_repo_files(dest.repo_id, repo_type="dataset")

    data_prefix = f"{HF_DATA_SUBDIR}/"
    if dest.subdir:
        data_prefix = f"{HF_DATA_SUBDIR}/{dest.subdir.rstrip('/')}/"

    uris: list[str] = []
    for path in paths:
        if not path.endswith(".parquet"):
            continue
        if not path.startswith(data_prefix):
            # eval/ at repo root, README files, etc. — skip
            continue
        uris.append(f"hf://datasets/{dest.repo_id}/{path}")
    return sorted(uris)


def filesystem_for_uri(uri: str):
    """
    Return a pyarrow-compatible filesystem object suitable for
    ``pq.read_table(..., filesystem=fs)``.

    For S3, this is ``pyarrow.fs.S3FileSystem``. For HF, this is
    ``huggingface_hub.HfFileSystem`` (fsspec-based; pyarrow accepts
    fsspec filesystems via duck-typing in ParquetFile / read_table).
    """
    if uri.startswith("s3://"):
        import pyarrow.fs as pafs
        return pafs.S3FileSystem()
    if uri.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        return HfFileSystem()
    raise ValueError(f"Unknown URI scheme in {uri!r}")


def fs_path_for_uri(uri: str) -> str:
    """
    Strip the scheme from a URI to get the path the filesystem expects.

    pyarrow.fs.S3FileSystem expects:    bucket/key/...
    huggingface_hub.HfFileSystem expects: datasets/repo/path
    """
    if uri.startswith("s3://"):
        return uri[len("s3://"):]
    if uri.startswith("hf://"):
        return uri[len("hf://"):]
    raise ValueError(f"Unknown URI scheme in {uri!r}")


def bare_key_for_uri(uri: str) -> str:
    """
    The "bare key" used as the per-file identifier in
    ``make_point_id(bare_key, row_index)``.

    Strips the scheme + bucket-or-repo identifier; keeps everything else.
    Brute-force, generate-queries, and the loader's vf_point_id macro must
    all agree on this form so IDs match across pipelines.

      s3://bucket/prefix/file.parquet      -> prefix/file.parquet
      hf://datasets/ns/name/data/file.pq   -> data/file.pq

    --- ID space anchoring (intentional design) ---
    The anchor is the top-level container: the S3 bucket, or the HF dataset
    repo. Two consequences fall out of this choice:

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
        rest = uri[len("s3://"):]
        _, _, key = rest.partition("/")
        return key
    if uri.startswith("hf://datasets/"):
        rest = uri[len("hf://datasets/"):]
        # Skip namespace/name; keep everything after.
        parts = rest.split("/", 2)
        return parts[2] if len(parts) == 3 else ""
    raise ValueError(f"Unknown URI scheme in {uri!r}")


def list_parquets_under(prefix_uri: str) -> list[str]:
    """
    List all .parquet files under a URI prefix. Does NOT filter eval/, since
    this is used to enumerate eval artifacts (queries, brute-force partials)
    that live under {root}/eval/.
    """
    if prefix_uri.startswith("s3://"):
        import boto3
        rest = prefix_uri[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        uris: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    uris.append(f"s3://{bucket}/{obj['Key']}")
        return sorted(uris)

    if prefix_uri.startswith("hf://datasets/"):
        from huggingface_hub import HfApi
        rest = prefix_uri[len("hf://datasets/"):]
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"Bad hf:// prefix: {prefix_uri!r}")
        repo_id = f"{parts[0]}/{parts[1]}"
        in_repo_prefix = parts[2] if len(parts) == 3 else ""
        api = HfApi()
        paths = api.list_repo_files(repo_id, repo_type="dataset")
        uris: list[str] = []
        for path in paths:
            if not path.endswith(".parquet"):
                continue
            if in_repo_prefix and not path.startswith(in_repo_prefix):
                continue
            uris.append(f"hf://datasets/{repo_id}/{path}")
        return sorted(uris)

    raise ValueError(f"Unknown URI scheme in {prefix_uri!r}")


def upload_file_to_uri(local_path: str, dest_uri: str) -> None:
    """
    Upload ``local_path`` to the given destination URI. Synchronous, intended
    for one-shot eval artifact writes (queries, brute-force partials/final).
    """
    if dest_uri.startswith("s3://"):
        import boto3
        rest = dest_uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        boto3.client("s3").upload_file(local_path, bucket, key)
        return

    if dest_uri.startswith("hf://datasets/"):
        from huggingface_hub import HfApi
        rest = dest_uri[len("hf://datasets/"):]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"hf:// upload URI needs in-repo path: {dest_uri!r}")
        repo_id = f"{parts[0]}/{parts[1]}"
        path_in_repo = parts[2]
        HfApi().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        return

    raise ValueError(f"Unknown URI scheme in {dest_uri!r}")


def upload_bytes_to_uri(data: bytes, dest_uri: str) -> None:
    """Same as upload_file_to_uri but for in-memory bytes."""
    if dest_uri.startswith("s3://"):
        import boto3
        rest = dest_uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
        return

    if dest_uri.startswith("hf://datasets/"):
        from huggingface_hub import HfApi
        rest = dest_uri[len("hf://datasets/"):]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"hf:// upload URI needs in-repo path: {dest_uri!r}")
        repo_id = f"{parts[0]}/{parts[1]}"
        path_in_repo = parts[2]
        HfApi().upload_file(
            path_or_fileobj=data,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
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
            bucket=ds_cfg["s3_bucket"],
            prefix=ds_cfg.get("s3_prefix", "").rstrip("/"),
        )
    if ds_type == "huggingface":
        return HfDestination(
            repo_id=ds_cfg["repo_id"],
            subdir=ds_cfg.get("subdir") or "",
        )
    raise ValueError(
        f"Unknown datasource.type for discovery: {ds_type!r}. "
        f"Supported: 's3', 'huggingface'."
    )
