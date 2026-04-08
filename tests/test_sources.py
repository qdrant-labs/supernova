from vectorforge.sources.base import DatasetSource
from vectorforge.embedders.base import Embedder
from vectorforge.models import Record


class FakeEmbedder(Embedder):
    """Embedder stub for testing source chunking."""

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def max_tokens(self) -> int:
        return 1000

    def split_text(self, text: str) -> list[str]:
        # No splitting for tests — just return as-is
        return [text]

    async def embed(self, texts):
        return [[0.0]] * len(texts)


class FakeSource(DatasetSource):
    """In-memory source for testing."""

    def __init__(self, rows: list[dict], text_field: str = "text"):
        self._rows = rows
        self._text_field = text_field

    @property
    def source_name(self) -> str:
        return "fake"

    def stream(self):
        yield from self._rows

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        return Record(
            row_id=row_id,
            source_row_id=0,
            chunk_id=chunk_id,
            chunk_index=0,
            text=row[self._text_field],
            columns=dict(row),
        )


embedder = FakeEmbedder()


def test_get_chunks_single_chunk():
    rows = [{"text": f"row {i}"} for i in range(5)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(embedder, chunk_size=10))
    assert len(chunks) == 1
    chunk_id, records = chunks[0]
    assert chunk_id == 0
    assert len(records) == 5
    assert records[0].text == "row 0"
    assert records[4].row_id == 4
    assert records[0].source_row_id == 0
    assert records[4].source_row_id == 4


def test_get_chunks_multiple_chunks():
    rows = [{"text": f"row {i}"} for i in range(25)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(embedder, chunk_size=10))
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
    chunks = list(source.get_chunks(embedder, chunk_size=10))
    assert len(chunks) == 2
    assert len(chunks[0][1]) == 10
    assert len(chunks[1][1]) == 10


def test_chunk_index_for_short_texts():
    """Short texts should all have chunk_index=0."""
    rows = [{"text": "short"} for _ in range(3)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(embedder, chunk_size=10))
    for _, records in chunks:
        for r in records:
            assert r.chunk_index == 0