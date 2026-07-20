"""HuggingFaceSource footer-index behavior: the sweep is the dominant HF
request cost at fleet scale, so these tests lock in HOW MANY footers get read,
not just what the index contains. No network: _fetch_count is stubbed.
"""

from __future__ import annotations

from nova_embed.sources.huggingface import HuggingFaceSource


def make_source(counts: list[int], offset: int = 0, limit: int | None = None):
    """Bare instance (no HF API calls) over files of the given row counts.

    Returns (source, fetched) where `fetched` records every footer read.
    """
    src = HuggingFaceSource.__new__(HuggingFaceSource)
    paths = [f"data/{i:04d}.parquet" for i in range(len(counts))]
    by_path = dict(zip(paths, counts))
    src.dataset_name = "org/data"
    src._parquet_paths = paths
    src._files_with_counts = []
    src._next_path_idx = 0
    src._counts_complete = False
    src._metadata_workers = 2
    src._total_rows_override = None
    src._offset = offset
    src._limit = limit
    src._local_paths = {}

    fetched: list[str] = []

    def fetch(path):
        fetched.append(path)
        return path, by_path[path]

    src._fetch_count = fetch
    return src, fetched


def test_total_rows_reads_every_footer():
    src, fetched = make_source([10] * 30)
    assert src.get_total_rows() == 300
    assert len(fetched) == 30


def test_window_stops_footer_sweep_early():
    # rank window [0, 100) over 100 files x 10 rows: only the first batch of
    # footers is needed, not all 100
    src, fetched = make_source([10] * 100, offset=0, limit=100)
    window = src._window_files()
    assert [p for p, _, _ in window] == [f"data/{i:04d}.parquet" for i in range(10)]
    assert len(fetched) < 100  # early stop: never indexed past the window
    # batched sweep may read slightly past the window boundary, never before it
    assert len(fetched) >= 10


def test_total_rows_override_skips_sweep_entirely():
    src, fetched = make_source([10] * 50)
    src._total_rows_override = 500
    assert src.get_total_rows() == 500
    assert fetched == []


def test_index_extends_without_rereading():
    src, fetched = make_source([10] * 100, offset=0, limit=100)
    src._window_files()
    first_pass = len(fetched)
    src.get_total_rows()  # broader ask: completes the index
    assert len(fetched) == 100  # continued where it stopped
    assert len(set(fetched)) == len(fetched)  # no footer read twice
    assert first_pass < 100


def test_set_window_rescopes_and_reuses_index():
    src, fetched = make_source([10] * 40)
    assert src.get_total_rows() == 400  # full sweep (the counting pass)
    src.set_window(200, 100)
    window = src._window_files()
    assert [p for p, _, _ in window] == [
        f"data/{i:04d}.parquet" for i in range(20, 30)
    ]
    assert len(fetched) == 40  # window change reused the index: zero new reads


def test_window_files_carry_correct_offsets():
    src, _ = make_source([5, 10, 20], offset=7, limit=10)
    window = src._window_files()
    # rows 7..17 live in file 1 (rows 5..15) and file 2 (rows 15..35)
    assert window == [
        ("data/0001.parquet", 10, 5),
        ("data/0002.parquet", 20, 15),
    ]
