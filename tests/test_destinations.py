"""
Tests for vectorforge.destinations — URI parsing, scheme dispatch, and the
bare-key derivation that brute-force / generate-queries / loader macros
must agree on.
"""

import pytest

from vectorforge.destinations import (
    HfDestination,
    S3Destination,
    bare_key_for_uri,
    fs_path_for_uri,
    parse_destination,
)


# ---------------------------------------------------------------------------
# parse_destination
# ---------------------------------------------------------------------------


def test_parse_s3_with_prefix():
    dest = parse_destination("s3://my-bucket/some/prefix")
    assert isinstance(dest, S3Destination)
    assert dest.bucket == "my-bucket"
    assert dest.prefix == "some/prefix"
    assert dest.root_uri == "s3://my-bucket/some/prefix"


def test_parse_s3_without_prefix():
    dest = parse_destination("s3://just-bucket")
    assert isinstance(dest, S3Destination)
    assert dest.bucket == "just-bucket"
    assert dest.prefix == ""
    assert dest.root_uri == "s3://just-bucket"


def test_parse_s3_strips_trailing_slash():
    assert parse_destination("s3://b/p/").prefix == "p"


def test_parse_hf_dataset_uri():
    dest = parse_destination("hf://datasets/qdrant/fineweb-bge-large")
    assert isinstance(dest, HfDestination)
    assert dest.repo_id == "qdrant/fineweb-bge-large"
    assert dest.subdir == ""
    assert dest.root_uri == "hf://datasets/qdrant/fineweb-bge-large"


def test_parse_hf_dataset_with_subdir():
    dest = parse_destination("hf://datasets/ns/name/cc-main-2025-26")
    assert isinstance(dest, HfDestination)
    assert dest.repo_id == "ns/name"
    assert dest.subdir == "cc-main-2025-26"
    # Note: subdir lives under data/ for HF
    assert dest.root_uri == "hf://datasets/ns/name/data/cc-main-2025-26"


def test_parse_hf_rejects_missing_namespace_or_name():
    with pytest.raises(ValueError, match="hf://"):
        parse_destination("hf://datasets/just-namespace")


def test_parse_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown URI scheme"):
        parse_destination("bb://bucket/key")


# ---------------------------------------------------------------------------
# eval_uri composition — eval/ at root for HF, under prefix for S3
# ---------------------------------------------------------------------------


def test_s3_eval_uri():
    dest = S3Destination(bucket="b", prefix="p/q")
    assert dest.eval_uri("queries_1000.parquet") == "s3://b/p/q/eval/queries_1000.parquet"


def test_hf_eval_uri_lives_at_repo_root_not_under_data():
    dest = HfDestination(repo_id="ns/name")
    # MUST NOT be under data/, otherwise load_dataset() would treat eval
    # artifacts as dataset rows.
    assert dest.eval_uri("queries_1000.parquet") == "hf://datasets/ns/name/eval/queries_1000.parquet"


def test_hf_eval_uri_ignores_subdir():
    # subdir applies to the corpus tree under data/; eval is a sibling.
    dest = HfDestination(repo_id="ns/name", subdir="cc-2025")
    assert dest.eval_uri("queries.parquet") == "hf://datasets/ns/name/eval/queries.parquet"


def test_hf_child_uri_goes_under_data():
    dest = HfDestination(repo_id="ns/name")
    assert dest.child_uri("rank00/batch_0.parquet") == "hf://datasets/ns/name/data/rank00/batch_0.parquet"


# ---------------------------------------------------------------------------
# bare_key_for_uri — consistency between sides that compute make_point_id
# ---------------------------------------------------------------------------


def test_bare_key_s3():
    assert bare_key_for_uri("s3://bucket/prefix/file.parquet") == "prefix/file.parquet"


def test_bare_key_s3_no_prefix():
    assert bare_key_for_uri("s3://bucket/file.parquet") == "file.parquet"


def test_bare_key_hf():
    # The HF bare key includes the data/ component, mirroring how S3 keys
    # include their prefix. Both sides (loader macro + brute-force) must
    # produce this exact form for IDs to match.
    assert bare_key_for_uri("hf://datasets/ns/name/data/file.parquet") == "data/file.parquet"


def test_bare_key_hf_nested():
    assert bare_key_for_uri(
        "hf://datasets/ns/name/data/cc-2025/rank00/batch_0.parquet"
    ) == "data/cc-2025/rank00/batch_0.parquet"


def test_bare_key_unknown_scheme():
    with pytest.raises(ValueError):
        bare_key_for_uri("file:///local/path.parquet")


# ---------------------------------------------------------------------------
# fs_path_for_uri — what pyarrow.fs / fsspec want
# ---------------------------------------------------------------------------


def test_fs_path_s3():
    assert fs_path_for_uri("s3://bucket/key/file.parquet") == "bucket/key/file.parquet"


def test_fs_path_hf():
    assert fs_path_for_uri("hf://datasets/ns/name/data/file.parquet") == "datasets/ns/name/data/file.parquet"