"""Row assembly (iter_chunks + empty-input policy), the parquet writer, and a
mini end-to-end run through run_embedder with fake backends + local storage."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("obstore")

import fake_backends  # noqa: F401  — registers the fake backends

from nova_embed.chunkers import build_chunker
from nova_embed.config import EmbedderEntry
from nova_embed.embedders.engine import OutputSpec, build_engine
from nova_embed.embedders.runner import run_embedder
from nova_embed.media import Modality
from nova_embed.models import (
    EmbeddedRecord,
    MultiVectorEmbedding,
    OutputKind,
    Record,
    SparseEmbedding,
)
from nova_embed.registry import STORAGE
from nova_embed.sources.base import DatasetSource, EmptyInputStats, iter_chunks
from nova_embed.storage.writer import write_batch


class ListSource(DatasetSource):
    def __init__(self, rows: list[dict], name: str = "test-source"):
        self._rows = rows
        self._name = name

    def stream(self) -> Iterator[dict]:
        yield from self._rows

    def format_record(self, row: dict) -> Record:
        return Record(row=row)

    @property
    def source_name(self) -> str:
        return self._name

    def get_total_rows(self) -> int:
        return len(self._rows)


TEXT_GROUPS = [{"text": Modality.TEXT}]


def collect(rows, input_groups=TEXT_GROUPS, **kwargs):
    stats = kwargs.pop("stats", EmptyInputStats())
    chunks = list(
        iter_chunks(ListSource(rows), input_groups, chunk_size=10, stats=stats, **kwargs)
    )
    records = [r for _, chunk in chunks for r in chunk]
    return records, stats


# ------------------------------------------------------- empty-input policy

def test_skip_drops_and_counts():
    records, stats = collect(
        [{"text": "a"}, {"text": ""}, {"text": None}, {"text": "b"}],
        on_empty_input="skip",
    )
    assert [r.row["text"] for r in records] == ["a", "b"]
    assert stats.rows_skipped == 2


TWO_ENTRY_GROUPS = [{"text": Modality.TEXT}, {"title": Modality.TEXT}]


def test_null_keeps_partial_rows():
    records, stats = collect(
        [{"text": "a", "title": ""}, {"text": "", "title": "t"}],
        input_groups=TWO_ENTRY_GROUPS,
        on_empty_input="null",
    )
    assert len(records) == 2
    assert stats.rows_skipped == 0


def test_null_still_skips_all_empty_rows():
    records, stats = collect(
        [{"text": "", "title": ""}, {"text": "a", "title": "t"}],
        input_groups=TWO_ENTRY_GROUPS,
        on_empty_input="null",
    )
    assert len(records) == 1
    assert stats.rows_skipped == 1


# ------------------------------------------- multimodal input groups

# One multimodal entry = ONE group spanning both columns: empty only when
# every part is empty, so partial rows are valid inputs, not policy events.
MM_GROUP = [{"text": Modality.TEXT, "image": Modality.IMAGE}]


def test_multimodal_group_keeps_partial_rows_under_skip():
    records, stats = collect(
        [
            {"text": "a", "image": b"img"},
            {"text": "b", "image": None},   # text-only: valid
            {"text": "", "image": b"img"},  # image-only: valid
            {"text": "", "image": None},    # all parts empty: skipped
        ],
        input_groups=MM_GROUP,
        on_empty_input="skip",
    )
    assert [r.row["text"] for r in records] == ["a", "b", ""]
    assert stats.rows_skipped == 1


def test_multimodal_group_error_only_when_all_parts_empty():
    records, _ = collect(
        [{"text": "a", "image": None}],
        input_groups=MM_GROUP,
        on_empty_input="error",
    )
    assert len(records) == 1
    with pytest.raises(ValueError, match="on_empty_input=error"):
        collect(
            [{"text": "", "image": None}],
            input_groups=MM_GROUP,
            on_empty_input="error",
        )


def test_multimodal_group_beside_plain_entry():
    # the plain entry's empty column is still a policy event even though the
    # multimodal group is satisfied
    records, stats = collect(
        [{"text": "a", "image": b"img", "title": ""}],
        input_groups=MM_GROUP + [{"title": Modality.TEXT}],
        on_empty_input="skip",
    )
    assert records == []
    assert stats.rows_skipped == 1


def test_error_policy_raises():
    with pytest.raises(ValueError, match="on_empty_input=error"):
        collect([{"text": "a"}, {"text": ""}], on_empty_input="error")


def test_wrong_input_column_dies_on_first_row():
    with pytest.raises(ValueError, match="Available columns"):
        collect([{"body": "a"}])


# ------------------------------------------------------- chunking

def test_chunker_splits_column_and_replicates_rest():
    chunker = build_chunker({"strategy": "fixed_char", "chunk_chars": 3})
    records, _ = collect(
        [{"text": "abcdefgh", "id": 7}],
        chunker=chunker,
        split_column="text",
    )
    assert [r.row["text"] for r in records] == ["abc", "def", "gh"]
    assert all(r.row["id"] == 7 for r in records)


def test_chunker_requires_split_column():
    chunker = build_chunker({"strategy": "fixed_char", "chunk_chars": 3})
    with pytest.raises(ValueError, match="together"):
        collect([{"text": "abc"}], chunker=chunker)


def test_batching_respects_chunk_size():
    rows = [{"text": f"t{i}"} for i in range(25)]
    chunks = list(
        iter_chunks(ListSource(rows), TEXT_GROUPS, chunk_size=10)
    )
    assert [len(c) for _, c in chunks] == [10, 10, 5]
    assert [cid for cid, _ in chunks] == [0, 1, 2]


# ------------------------------------------------------- writer

def spec(name, column, kind):
    return OutputSpec(
        name=name,
        column=column,
        kind=kind,
        model_name="m",
        dimensions=2,
        max_tokens=None,
        input_column="text",
        modality=Modality.TEXT,
    )


def test_writer_all_kinds_with_nulls(tmp_path):
    records = [
        EmbeddedRecord(
            row={"text": "a", "id": 1},
            embeddings={
                "d": [1.0, 2.0],
                "s": SparseEmbedding(indices=[3], values=[0.5]),
                "m": MultiVectorEmbedding(vectors=[[1.0, 0.0]]),
            },
        ),
        EmbeddedRecord(
            row={"text": "", "id": 2},
            embeddings={"d": None, "s": None, "m": None},
        ),
    ]
    path = write_batch(
        records,
        str(tmp_path),
        0,
        output_specs=[
            spec("d", "dense_col", OutputKind.DENSE),
            spec("s", "sparse_col", OutputKind.SPARSE),
            spec("m", "mv_col", OutputKind.MULTIVECTOR),
        ],
    )
    table = pq.read_table(path)
    assert table.column_names == ["text", "id", "dense_col", "sparse_col", "mv_col"]
    assert table.column("dense_col").to_pylist() == [[1.0, 2.0], None]
    assert table.column("sparse_col").to_pylist() == [
        {"indices": [3], "values": [0.5]},
        None,
    ]
    assert table.column("mv_col").to_pylist() == [[[1.0, 0.0]], None]
    dense_type = table.schema.field("dense_col").type
    assert pa.types.is_list(dense_type) and dense_type.value_type == pa.float32()


def test_writer_accepts_ndarray_dense_rows(tmp_path):
    # ST backends keep dense rows as float32 ndarrays (4B/elem vs ~32B for
    # Python floats in the flush buffer); the writer must take them as-is.
    import numpy as np

    records = [
        EmbeddedRecord(row={"id": 1}, embeddings={"d": np.array([1.0, 2.0], dtype=np.float32)}),
        EmbeddedRecord(row={"id": 2}, embeddings={"d": None}),
    ]
    path = write_batch(
        records, str(tmp_path), 0, output_specs=[spec("d", "dense_col", OutputKind.DENSE)]
    )
    assert pq.read_table(path).column("dense_col").to_pylist() == [[1.0, 2.0], None]


def test_writer_schema_stable_when_batch_all_null(tmp_path):
    records = [EmbeddedRecord(row={"text": ""}, embeddings={"d": None})]
    path = write_batch(
        records, str(tmp_path), 1, output_specs=[spec("d", "dense_col", OutputKind.DENSE)]
    )
    table = pq.read_table(path)
    # typed column even when every value is null — schema identical across batches
    dense_type = table.schema.field("dense_col").type
    assert pa.types.is_list(dense_type) and dense_type.value_type == pa.float32()


_UUID_PARQUET = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.parquet"
)


def test_writer_content_addressed_name_is_deterministic(tmp_path):
    records = [EmbeddedRecord(row={"text": "a"}, embeddings={"d": [1.0, 2.0]})]
    specs = [spec("d", "dense_col", OutputKind.DENSE)]
    p1 = write_batch(
        records, str(tmp_path / "one"), 0, output_specs=specs, content_addressed=True
    )
    p2 = write_batch(
        records,
        str(tmp_path / "two"),
        7,  # different batch id and rank prefix: the CONTENT names the file
        output_specs=specs,
        filename_prefix="rank03_",
        content_addressed=True,
    )
    assert _UUID_PARQUET.fullmatch(os.path.basename(p1))
    assert os.path.basename(p1) == os.path.basename(p2)


def test_writer_shard_buckets_keep_counter_names(tmp_path):
    specs = [spec("d", "dense_col", OutputKind.DENSE)]
    rels = []
    for i in range(8):
        records = [
            EmbeddedRecord(row={"text": f"t{i}"}, embeddings={"d": [float(i)]})
        ]
        p = write_batch(records, str(tmp_path), i, output_specs=specs, shard_buckets=4)
        rels.append(os.path.relpath(p, tmp_path))
    for rel in rels:
        bucket, name = rel.split(os.sep)
        assert re.fullmatch(r"00[0-3]", bucket)
        assert name.startswith("batch_")  # counters kept without content_addressed


def test_writer_shard_bucket_derived_from_content_uuid(tmp_path):
    records = [EmbeddedRecord(row={"text": "a"}, embeddings={"d": [1.0]})]
    p = write_batch(
        records,
        str(tmp_path),
        0,
        output_specs=[spec("d", "dense_col", OutputKind.DENSE)],
        content_addressed=True,
        shard_buckets=16,
    )
    bucket, name = os.path.relpath(p, tmp_path).split(os.sep)
    u = uuid.UUID(name.removesuffix(".parquet"))
    assert int(bucket) == int.from_bytes(u.bytes[:8], "big") % 16


def test_writer_rejects_collision_with_source_column(tmp_path):
    records = [EmbeddedRecord(row={"text": "a"}, embeddings={"d": [1.0]})]
    with pytest.raises(ValueError, match="collide with source columns"):
        write_batch(
            records, str(tmp_path), 0, output_specs=[spec("d", "text", OutputKind.DENSE)]
        )


# ------------------------------------------------------- end to end

def test_run_embedder_end_to_end(tmp_path):
    rows = [
        {"text": "hello", "id": 0},
        {"text": "", "id": 1},  # skipped by default policy
        {"text": "world!!", "id": 2},
    ]
    entries = [
        EmbedderEntry.model_validate(
            {
                "name": "dense_a",
                "kind": "dense",
                "type": "fake",
                "input_column": "text",
                "modality": "text",
            }
        ),
        EmbedderEntry.model_validate(
            {
                "name": "sparse_a",
                "kind": "sparse",
                "type": "fake",
                "input_column": "text",
                "modality": "text",
            }
        ),
    ]
    engine = build_engine(entries)
    storage = STORAGE.build({"type": "local", "output_dir": str(tmp_path)})

    asyncio.run(
        run_embedder(
            source=ListSource(rows),
            engine=engine,
            storage=storage,
            chunk_size=2,
            num_workers=2,
            flush_threshold=100,
            output_dir=str(tmp_path),
        )
    )

    files = sorted(tmp_path.glob("batch_*.parquet"))
    assert files
    table = pq.read_table(files[0])
    assert table.num_rows == 2  # empty row skipped
    assert table.column("dense_a_embedding").to_pylist() == [
        [5.0, 5.0],
        [7.0, 7.0],
    ]
    assert table.column("id").to_pylist() == [0, 2]

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["total_records"] == 2
    assert manifest["rows_skipped_empty_input"] == 1
    assert manifest["on_empty_input"] == "skip"
    names = {e["name"]: e for e in manifest["embedders"]}
    assert names["dense_a"]["kind"] == "dense"
    assert names["dense_a"]["column"] == "dense_a_embedding"
    assert names["sparse_a"]["modality"] == "text"


def dense_entry(**overrides):
    data = {
        "name": "dense_a",
        "kind": "dense",
        "type": "fake",
        "input_column": "text",
        "modality": "text",
    }
    data.update(overrides)
    return EmbedderEntry.model_validate(data)


def run(tmp_path, rows, entries, **kwargs):
    engine = build_engine(entries)
    storage = STORAGE.build({"type": "local", "output_dir": str(tmp_path)})
    asyncio.run(
        run_embedder(
            source=ListSource(rows),
            engine=engine,
            storage=storage,
            chunk_size=2,
            num_workers=1,
            flush_threshold=100,
            output_dir=str(tmp_path),
            **kwargs,
        )
    )
    return pq.read_table(sorted(tmp_path.glob("batch_*.parquet"))[0])


def test_drop_columns_removes_input_column_from_output(tmp_path):
    table = run(
        tmp_path,
        [{"text": "hello", "id": 0}],
        [dense_entry()],
        drop_columns=["text"],  # embed it, don't carry it
    )
    assert table.column_names == ["id", "dense_a_embedding"]
    assert table.column("dense_a_embedding").to_pylist() == [[5.0, 5.0]]

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["drop_columns"] == ["text"]


def test_drop_columns_typo_dies_on_first_chunk(tmp_path):
    with pytest.raises(ValueError, match="drop_columns \\['imge'\\] not found"):
        run(tmp_path, [{"text": "hello"}], [dense_entry()], drop_columns=["imge"])


def test_run_embedder_content_addressed_sharded_output(tmp_path):
    rows = [{"text": f"text number {i}", "id": i} for i in range(5)]
    engine = build_engine([dense_entry()])
    storage = STORAGE.build({"type": "local", "output_dir": str(tmp_path)})
    asyncio.run(
        run_embedder(
            source=ListSource(rows),
            engine=engine,
            storage=storage,
            chunk_size=2,
            num_workers=1,
            flush_threshold=2,  # several flushes -> several files
            output_dir=str(tmp_path),
            content_addressed_files=True,
            shard_output_buckets=8,
        )
    )

    files = sorted(tmp_path.glob("*/*.parquet"))
    assert files
    for f in files:
        assert re.fullmatch(r"00[0-7]", f.parent.name)
        assert _UUID_PARQUET.fullmatch(f.name)

    # every row lands exactly once across the sharded files
    ids = [i for f in files for i in pq.read_table(f).column("id").to_pylist()]
    assert sorted(ids) == list(range(5))

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["content_addressed_files"] is True
    assert manifest["shard_output_buckets"] == 8
    # the manifest is the record of which files this rank wrote
    assert sorted(manifest["output_files"]) == sorted(
        str(f.relative_to(tmp_path)) for f in files
    )


def test_run_embedder_multimodal_end_to_end(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (2, 2))
    rows = [
        {"text": "hello", "image": img, "id": 0},  # both parts
        {"text": "hey", "image": None, "id": 1},   # text-only — valid input
        {"text": "", "image": img, "id": 2},       # image-only — valid input
        {"text": "", "image": None, "id": 3},      # all parts empty — skipped
    ]
    entries = [
        EmbedderEntry.model_validate(
            {
                "name": "mm",
                "kind": "dense",
                "type": "fake_mm",
                "modality": "multimodal",
                "input_columns": {"text": "text", "image": "image"},
                "instruction": "Represent the user's input.",
            }
        )
    ]
    # drop the image column: a PIL object can't be written to parquet anyway
    table = run(tmp_path, rows, entries, drop_columns=["image"])
    assert table.column("id").to_pylist() == [0, 1, 2]
    # fake_mm encodes [len(text), has_image]
    assert table.column("mm_embedding").to_pylist() == [
        [5.0, 1.0],
        [3.0, 0.0],
        [0.0, 1.0],
    ]

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["rows_skipped_empty_input"] == 1
    (spec,) = manifest["embedders"]
    assert spec["modality"] == "multimodal"
    assert spec["input_column"] == "text=text,image=image"
    # the instruction is part of the embedding space — the query side needs it
    assert spec["instruction"] == "Represent the user's input."
