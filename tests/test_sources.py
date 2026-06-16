from supernova.sources.base import DatasetSource
from supernova.chunkers import Chunker, PassthroughChunker
from supernova.models import Record


class FakeSource(DatasetSource):
    """In-memory source for testing."""

    def __init__(self, rows: list[dict], text_field: str = "text"):
        self._rows = rows
        self._text_field = text_field

    @property
    def source_name(self) -> str:
        return "fake"

    def get_total_rows(self) -> int:
        return len(self._rows)

    def stream(self):
        yield from self._rows

    def format_record(self, row: dict) -> Record:
        return Record(text=row[self._text_field], columns=dict(row))


class FakeSplitChunker(Chunker):
    """Splits text on whitespace — one piece per word — to exercise fan-out."""

    def chunk(self, text: str) -> list[str]:
        return text.split()


# Default: one piece per row (the no-op chunker). get_chunks takes a Chunker,
# not an embedder — splitting is owned by the chunkers module (issue #12).
chunker = PassthroughChunker()


def test_get_chunks_single_chunk():
    rows = [{"text": f"row {i}"} for i in range(5)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunker, chunk_size=10))
    assert len(chunks) == 1
    chunk_id, records = chunks[0]
    assert chunk_id == 0
    assert len(records) == 5
    assert records[0].text == "row 0"
    assert records[4].text == "row 4"


def test_get_chunks_multiple_chunks():
    rows = [{"text": f"row {i}"} for i in range(25)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunker, chunk_size=10))
    assert len(chunks) == 3
    assert chunks[0][0] == 0
    assert len(chunks[0][1]) == 10
    assert chunks[1][0] == 1
    assert len(chunks[1][1]) == 10
    assert chunks[2][0] == 2
    assert len(chunks[2][1]) == 5


def test_get_chunks_exact_boundary():
    rows = [{"text": f"row {i}"} for i in range(20)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunker, chunk_size=10))
    assert len(chunks) == 2
    assert len(chunks[0][1]) == 10
    assert len(chunks[1][1]) == 10


def test_chunk_index_for_short_texts():
    rows = [{"text": "short"} for _ in range(3)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunker, chunk_size=10))
    assert len(chunks) == 1
    _, records = chunks[0]
    assert len(records) == 3


def test_chunker_fans_one_row_into_multiple_records():
    # A splitting chunker turns one row's text into several pieces; each piece
    # becomes its own Record, and the batch packing counts pieces, not rows.
    rows = [{"text": "alpha beta gamma"}, {"text": "delta epsilon"}]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(FakeSplitChunker(), chunk_size=10))
    assert len(chunks) == 1
    _, records = chunks[0]
    assert [r.text for r in records] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
    ]
    # original row columns are preserved on every piece
    assert records[0].columns["text"] == "alpha beta gamma"


def test_chunker_pieces_respect_batch_size():
    # 4 words across 2 rows, chunk_size=3 -> pieces split across two batches.
    rows = [{"text": "a b c"}, {"text": "d"}]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(FakeSplitChunker(), chunk_size=3))
    assert [len(records) for _, records in chunks] == [3, 1]
    assert [cid for cid, _ in chunks] == [0, 1]


def test_blank_rows_are_skipped():
    rows = [{"text": "keep"}, {"text": "   "}, {"text": ""}, {"text": "also"}]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunker, chunk_size=10))
    _, records = chunks[0]
    assert [r.text for r in records] == ["keep", "also"]
