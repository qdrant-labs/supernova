"""JsonlSource: file-granular sharding + streaming/parse behavior.

No network: HF API calls happen only in __init__, which these tests bypass
(instances built via __new__, like test_hf_source.py). Streaming is exercised
against real local temp files by stubbing _file_providers to yield them.
"""

from __future__ import annotations

import gzip
import json

import pytest

from nova_embed.sources.base import (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    apply_record_projection,
    files_for_shard,
)
from nova_embed.sources.jsonl import JsonlSource


# --- files_for_shard: the core sharding contract ---------------------------


def test_files_for_shard_covers_all_without_overlap():
    paths = [f"chunk/{i:04d}.jsonl" for i in range(10)]
    shards = [files_for_shard(paths, r, 4) for r in range(4)]
    # every file assigned exactly once, order preserved
    assert sum(shards, []) == paths
    # counts differ by at most one (balanced): 10 over 4 -> 3,3,2,2
    assert [len(s) for s in shards] == [3, 3, 2, 2]


def test_files_for_shard_is_contiguous():
    paths = [f"{i}.jsonl" for i in range(7)]
    assert files_for_shard(paths, 0, 3) == ["0.jsonl", "1.jsonl", "2.jsonl"]
    assert files_for_shard(paths, 1, 3) == ["3.jsonl", "4.jsonl"]
    assert files_for_shard(paths, 2, 3) == ["5.jsonl", "6.jsonl"]


def test_files_for_shard_exact_division():
    paths = [f"{i}.jsonl" for i in range(6)]
    assert [len(files_for_shard(paths, r, 3)) for r in range(3)] == [2, 2, 2]


def test_files_for_shard_more_jobs_than_files_leaves_empty_ranks():
    paths = ["a.jsonl", "b.jsonl"]
    assert files_for_shard(paths, 0, 5) == ["a.jsonl"]
    assert files_for_shard(paths, 1, 5) == ["b.jsonl"]
    assert files_for_shard(paths, 2, 5) == []
    assert files_for_shard(paths, 4, 5) == []


def test_files_for_shard_single_job_gets_everything():
    paths = [f"{i}.jsonl" for i in range(5)]
    assert files_for_shard(paths, 0, 1) == paths


def test_files_for_shard_rejects_bad_args():
    paths = ["a.jsonl"]
    with pytest.raises(ValueError):
        files_for_shard(paths, 0, 0)
    with pytest.raises(ValueError):
        files_for_shard(paths, 3, 3)  # rank out of range
    with pytest.raises(ValueError):
        files_for_shard(paths, -1, 3)


# --- set_file_shard on the source ------------------------------------------


def _bare_source(paths, **attrs):
    """A JsonlSource with attributes set directly (no HF __init__)."""
    src = JsonlSource.__new__(JsonlSource)
    src.dataset_name = "org/data"
    src.revision = None
    src.render_columns = {}
    src.exclude_columns = set()
    src._paths = list(paths)
    src._my_files = list(paths)
    src._include_provenance = False
    src._on_bad_line = "skip"
    for k, v in attrs.items():
        setattr(src, k, v)
    return src


def test_set_file_shard_narrows_to_rank_subset():
    paths = [f"chunk/{i:04d}.jsonl" for i in range(8)]
    src = _bare_source(paths)
    src.set_file_shard(1, 4)
    assert src._my_files == ["chunk/0002.jsonl", "chunk/0003.jsonl"]


def test_list_files_and_files_for_rank():
    paths = [f"{i}.jsonl" for i in range(4)]
    src = _bare_source(paths)
    assert src.list_files() == [(p, None) for p in paths]
    assert src.files_for_rank(0, 2) == ["0.jsonl", "1.jsonl"]


def test_get_total_rows_unsupported():
    src = _bare_source(["a.jsonl"])
    with pytest.raises(NotImplementedError):
        src.get_total_rows()


# --- streaming / parsing / provenance --------------------------------------


def _write_jsonl(path, rows, gz=False):
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _stub_providers(src, files):
    """Make stream() read the given local (path_label, local_path) files."""
    def providers():
        for label, local in files:
            yield label, local, lambda: None
    src._file_providers = providers


def test_stream_reads_only_assigned_files(tmp_path):
    f0 = tmp_path / "0.jsonl"
    f1 = tmp_path / "1.jsonl"
    _write_jsonl(f0, [{"contents": "a"}, {"contents": "b"}])
    _write_jsonl(f1, [{"contents": "c"}])
    src = _bare_source(["0.jsonl", "1.jsonl"])
    _stub_providers(src, [("0.jsonl", str(f0)), ("1.jsonl", str(f1))])
    rows = list(src.stream())
    assert [r["contents"] for r in rows] == ["a", "b", "c"]


def test_only_jsonl_suffixes_supported():
    # F1: bare .json is NOT treated as JSON Lines (would silently yield 0 rows).
    from nova_embed.sources.jsonl import _JSONL_SUFFIXES

    assert _JSONL_SUFFIXES == (".jsonl", ".jsonl.gz")
    assert "corpus.jsonl".endswith(_JSONL_SUFFIXES)
    assert "corpus.jsonl.gz".endswith(_JSONL_SUFFIXES)
    assert not "corpus.json".endswith(_JSONL_SUFFIXES)
    assert not "corpus.json.gz".endswith(_JSONL_SUFFIXES)


def test_provenance_is_record_ordinal_not_physical_line(tmp_path):
    # F2: source_row_number counts only EMITTED rows, so blank/bad lines don't
    # create gaps — parity with the parquet source's data-row index.
    f = tmp_path / "x.jsonl"
    f.write_text(
        '{"contents": "a"}\n'  # physical line 0 -> record 0
        "\n"                     # physical line 1 (blank, skipped)
        "{bad json}\n"           # physical line 2 (bad, skipped)
        '{"contents": "b"}\n',   # physical line 3 -> record 1
        encoding="utf-8",
    )
    src = _bare_source(["x.jsonl"], _include_provenance=True, _on_bad_line="skip")
    _stub_providers(src, [("x.jsonl", str(f))])
    rows = list(src.stream())
    assert [r["contents"] for r in rows] == ["a", "b"]
    assert [r[SOURCE_ROW_COLUMN] for r in rows] == [0, 1]  # not [0, 3]


def test_stream_stamps_provenance(tmp_path):
    f = tmp_path / "chunk/x.jsonl"
    f.parent.mkdir(parents=True)
    _write_jsonl(f, [{"contents": "a"}, {"contents": "b"}])
    src = _bare_source(["chunk/x.jsonl"], _include_provenance=True)
    _stub_providers(src, [("chunk/x.jsonl", str(f))])
    rows = list(src.stream())
    assert rows[0][SOURCE_FILE_COLUMN] == "chunk/x.jsonl"
    assert rows[0][SOURCE_ROW_COLUMN] == 0
    assert rows[1][SOURCE_ROW_COLUMN] == 1


def test_stream_skips_blank_lines(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"contents": "a"}\n\n{"contents": "b"}\n', encoding="utf-8")
    src = _bare_source(["x.jsonl"])
    _stub_providers(src, [("x.jsonl", str(f))])
    assert [r["contents"] for r in src.stream()] == ["a", "b"]


def test_stream_bad_line_skip_vs_error(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"contents": "a"}\n{bad json}\n{"contents": "b"}\n', encoding="utf-8")

    src_skip = _bare_source(["x.jsonl"], _on_bad_line="skip")
    _stub_providers(src_skip, [("x.jsonl", str(f))])
    assert [r["contents"] for r in src_skip.stream()] == ["a", "b"]

    src_err = _bare_source(["x.jsonl"], _on_bad_line="error")
    _stub_providers(src_err, [("x.jsonl", str(f))])
    with pytest.raises(ValueError):
        list(src_err.stream())


def test_stream_reads_gzip(tmp_path):
    f = tmp_path / "x.jsonl.gz"
    _write_jsonl(f, [{"contents": "a"}, {"contents": "b"}], gz=True)
    src = _bare_source(["x.jsonl.gz"])
    _stub_providers(src, [("x.jsonl.gz", str(f))])
    assert [r["contents"] for r in src.stream()] == ["a", "b"]


def test_open_lines_binary_handle_plain():
    # Exercises the remote (non-prefetch) branch: a binary file handle rather
    # than a local path string. HfFileSystem.open(...) returns such a handle.
    import io as _io

    src = _bare_source(["x.jsonl"])
    fh = _io.BytesIO(b'{"contents": "a"}\n{"contents": "b"}\n')
    lines = list(src._open_lines(fh, "x.jsonl"))
    assert [json.loads(l)["contents"] for l in lines] == ["a", "b"]


def test_open_lines_binary_handle_gzip():
    import io as _io

    buf = _io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b'{"contents": "a"}\n{"contents": "b"}\n')
    buf.seek(0)
    src = _bare_source(["x.jsonl.gz"])
    lines = list(src._open_lines(buf, "x.jsonl.gz"))
    assert [json.loads(l)["contents"] for l in lines] == ["a", "b"]


def test_stream_via_binary_handles(tmp_path):
    # stream() through the binary-handle path end to end (not just _open_lines).
    import io as _io

    def providers():
        yield "0.jsonl", _io.BytesIO(b'{"contents": "a"}\n'), lambda: None
        yield "1.jsonl", _io.BytesIO(b'{"contents": "b"}\n'), lambda: None

    src = _bare_source(["0.jsonl", "1.jsonl"], _include_provenance=True)
    src._file_providers = providers
    rows = list(src.stream())
    assert [r["contents"] for r in rows] == ["a", "b"]
    assert [r[SOURCE_FILE_COLUMN] for r in rows] == ["0.jsonl", "1.jsonl"]


# --- record projection (shared helper) -------------------------------------


def test_format_record_renders_then_excludes():
    src = _bare_source(
        ["x.jsonl"],
        render_columns={"combined": "{title}: {content}"},
        exclude_columns={"content"},
    )
    rec = src.format_record({"title": "T", "content": "C", "keep": 1})
    assert rec.row == {"title": "T", "keep": 1, "combined": "T: C"}


def test_apply_record_projection_direct():
    rec = apply_record_projection(
        {"a": 1, "b": 2}, render_columns={}, exclude_columns={"b"}
    )
    assert rec.row == {"a": 1}
