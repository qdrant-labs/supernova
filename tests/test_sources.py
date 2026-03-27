from vectorforge.sources.base import DatasetSource
from vectorforge.models import Record


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
            chunk_id=chunk_id,
            text=row[self._text_field],
            source=self.source_name,
        )


def test_get_chunks_single_chunk():
    rows = [{"text": f"row {i}"} for i in range(5)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunk_size=10))
    assert len(chunks) == 1
    chunk_id, records = chunks[0]
    assert chunk_id == 0
    assert len(records) == 5
    assert records[0].text == "row 0"
    assert records[4].row_id == 4


def test_get_chunks_multiple_chunks():
    rows = [{"text": f"row {i}"} for i in range(25)]
    source = FakeSource(rows)
    chunks = list(source.get_chunks(chunk_size=10))
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
    chunks = list(source.get_chunks(chunk_size=10))
    assert len(chunks) == 2
    assert len(chunks[0][1]) == 10
    assert len(chunks[1][1]) == 10
