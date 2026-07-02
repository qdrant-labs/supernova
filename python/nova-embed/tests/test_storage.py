"""Tests for the generic object-store backend.

Exercises the parts that don't need a cloud account: local (`file://`) round-trip
of upload_file/upload_bytes, scheme detection, config promotion of endpoint/region
(the S3-compatible knobs), and registry wiring under both `object_store` and the
`s3` name. Constructing an S3 backend is lazy, so these run offline with no
credentials.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("obstore")  # storage's object-store backend (install nova-embed[embed])

import nova_embed.storage  # noqa: F401  — import side effect registers the backends
from nova_embed.registry import STORAGE
from nova_embed.storage.object_store import ObjectStoreBackend


def test_local_path_roundtrip(tmp_path):
    root = tmp_path / "out"  # doesn't exist yet — ensure_ready must create it
    be = ObjectStoreBackend.from_config({"path": f"file://{root}"})
    assert be.destination == f"file://{root}"

    staging = tmp_path / "staging.parquet"
    staging.write_bytes(b"PAR1-batch")
    asyncio.run(be.upload_file(str(staging), "sub/dir/batch_0.parquet"))
    assert not staging.exists()  # staging copy is consumed
    assert (root / "sub/dir/batch_0.parquet").read_bytes() == b"PAR1-batch"

    asyncio.run(be.upload_bytes(b'{"ok":true}', "manifest.json"))
    assert (root / "manifest.json").read_bytes() == b'{"ok":true}'


def test_upload_file_defaults_to_basename(tmp_path):
    root = tmp_path / "out"
    be = ObjectStoreBackend.from_config({"path": f"file://{root}"})
    staging = tmp_path / "abc.parquet"
    staging.write_bytes(b"x")
    asyncio.run(be.upload_file(str(staging)))  # no remote_subpath → basename
    assert (root / "abc.parquet").read_bytes() == b"x"


def test_endpoint_and_region_promoted_into_config():
    be = ObjectStoreBackend.from_config(
        {"path": "s3://b/p", "endpoint": "https://acct.r2.cloudflarestorage.com", "region": "auto"}
    )
    assert be._config["endpoint"].endswith("r2.cloudflarestorage.com")
    assert be._config["region"] == "auto"


def test_scheme_selects_store_kind():
    assert ObjectStoreBackend.from_config({"path": "gs://bucket/pre"}).scheme == "gs"
    assert ObjectStoreBackend.from_config({"path": "az://container/pre"}).scheme == "az"


def test_registry_object_store_and_s3_alias(tmp_path):
    a = STORAGE.build({"type": "object_store", "path": f"file://{tmp_path}"})
    b = STORAGE.build({"type": "s3", "path": "s3://bkt/pre"})
    assert isinstance(a, ObjectStoreBackend)
    assert isinstance(b, ObjectStoreBackend) and b.path == "s3://bkt/pre"


def test_missing_path_errors():
    with pytest.raises(ValueError):
        ObjectStoreBackend.from_config({})


def test_path_without_scheme_errors():
    with pytest.raises(ValueError):
        ObjectStoreBackend(path="/no/scheme/path")
