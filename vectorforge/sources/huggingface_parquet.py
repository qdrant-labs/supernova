"""
File-level sharded reader for HuggingFace datasets stored as native parquet shards.

Use this when the dataset is huge (≥5GB) and HF's `IterableDataset.skip()` is
broken — typical for native-parquet datasets like HuggingFaceTB/dclm-edu where
skip(N > ~1.5M) silently no-ops and yields from offset 0 regardless. Reproduced:

    ds = load_dataset("HuggingFaceTB/dclm-edu", split="train", streaming=True)
    ds.skip(0)            -> "Books..."  (different doc)
    ds.skip(300_000_000)  -> "Car-sharing firm..."
    ds.skip(1_500_000)    -> "Car-sharing firm..." (same as 300M — skip ignored)

Mechanism:
  1. List all parquet files in the dataset's HF repo (filtered to the chosen split).
  2. Read each file's parquet footer to get its row count.
  3. stream() maps (offset, limit) to a contiguous file range and yields rows from
     those files in order, applying intra-file offsets at the boundaries.

This exposes the same DatasetSource interface as HuggingFaceSource so the rest
of the pipeline (chunker, runner, writer) is untouched.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from vectorforge.sources.base import DatasetSource
from vectorforge.sources.huggingface import _build_text_extractor
from vectorforge.models import Record

logger = logging.getLogger(__name__)


class HuggingFaceParquetSource(DatasetSource):
    def __init__(
        self,
        dataset_name: str,
        config: str | None = None,
        split: str = "train",
        text_field: str | None = "text",
        text_template: str | None = None,
        exclude_columns: list[str] | None = None,
        offset: int | None = None,
        limit: int | None = None,
        total_rows_override: int | None = None,
        path_filter: str | None = None,
        metadata_workers: int = 32,
    ):
        """
        Args:
            dataset_name: HF Hub repo id, e.g. "HuggingFaceTB/dclm-edu".
            config: HF config name (only used for filtering when relevant — many
                native-parquet datasets have a single default config).
            split: HF split name. The path_filter (or default split-name match)
                determines which files are read.
            offset / limit: applied as a row-window across all selected files.
            total_rows_override: skip the metadata sweep at construction time by
                trusting this number; metadata is still read per-file as we go.
            path_filter: substring filter on parquet file paths (e.g. "train/").
                Defaults to filtering by the split name when present in paths.
            metadata_workers: parallelism for the per-file footer fetches.
        """
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.text_field = text_field
        self.text_template = text_template
        self.exclude_columns = set(exclude_columns or [])
        self._offset = offset or 0
        self._limit = limit
        self._total_rows_override = total_rows_override
        self._metadata_workers = metadata_workers
        self._extract_text = _build_text_extractor(text_field, text_template)

        from huggingface_hub import HfApi, HfFileSystem
        self._fs = HfFileSystem()
        api = HfApi()
        all_files = api.list_repo_files(dataset_name, repo_type="dataset")

        parquet_paths = sorted(f for f in all_files if f.endswith(".parquet"))
        # apply explicit filter if given, otherwise filter by split name when it
        # naturally appears in paths (e.g. "train/0.parquet" or "data/train-...")
        if path_filter is not None:
            parquet_paths = [p for p in parquet_paths if path_filter in p]
        elif any(split in p for p in parquet_paths):
            parquet_paths = [p for p in parquet_paths if split in p]

        if not parquet_paths:
            raise ValueError(
                f"No parquet files found in {dataset_name} matching split={split!r} / "
                f"path_filter={path_filter!r}"
            )
        self._parquet_paths = parquet_paths
        # lazy: list of (path, num_rows) -- populated on first use
        self._files_with_counts: list[tuple[str, int]] | None = None

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def _ensure_counts(self) -> None:
        if self._files_with_counts is not None:
            return

        import pyarrow.parquet as pq

        def fetch(path: str) -> tuple[str, int]:
            try:
                pf = pq.ParquetFile(f"datasets/{self.dataset_name}/{path}", filesystem=self._fs)
                return path, pf.metadata.num_rows
            except Exception as e:
                logger.warning("Failed to read row count for %s: %s", path, e)
                return path, 0

        logger.info(
            "Reading parquet footers for %d files (parallel=%d)...",
            len(self._parquet_paths), self._metadata_workers,
        )
        with ThreadPoolExecutor(max_workers=self._metadata_workers) as ex:
            results = list(ex.map(fetch, self._parquet_paths))
        # filter out failed / zero-row files so they don't break offset math
        results = [(p, n) for p, n in results if n > 0]
        self._files_with_counts = results
        logger.info(
            "Indexed %d parquet files, %d total rows",
            len(results), sum(n for _, n in results),
        )

    def get_total_rows(self) -> int:
        if self._total_rows_override is not None:
            return self._total_rows_override
        self._ensure_counts()
        return sum(n for _, n in self._files_with_counts)

    def stream(self) -> Iterator[dict]:
        import pyarrow.parquet as pq

        self._ensure_counts()
        offset = self._offset
        limit = self._limit
        rows_yielded = 0

        cumulative = 0
        for path, num_rows in self._files_with_counts:
            file_start = cumulative
            file_end = cumulative + num_rows
            cumulative = file_end

            # this file ends before our window starts -> skip entirely
            if file_end <= offset:
                continue
            # we've satisfied limit -> done
            if limit is not None and rows_yielded >= limit:
                return

            # offset within this file (0 if we're past the start of our window)
            intra_offset = max(0, offset - file_start)
            available_in_file = num_rows - intra_offset
            want_from_file = (
                min(available_in_file, limit - rows_yielded) if limit is not None
                else available_in_file
            )
            if want_from_file <= 0:
                continue

            url = f"datasets/{self.dataset_name}/{path}"
            pf = pq.ParquetFile(url, filesystem=self._fs)
            taken_from_file = 0

            for batch in pf.iter_batches(batch_size=10_000):
                batch_len = len(batch)
                # batch is entirely before the intra-file offset
                if intra_offset >= batch_len:
                    intra_offset -= batch_len
                    continue

                rows = batch.to_pylist()
                if intra_offset > 0:
                    rows = rows[intra_offset:]
                    intra_offset = 0

                remaining_in_file = want_from_file - taken_from_file
                if len(rows) > remaining_in_file:
                    rows = rows[:remaining_in_file]

                for row in rows:
                    yield row
                taken_from_file += len(rows)
                rows_yielded += len(rows)

                if taken_from_file >= want_from_file:
                    break

            logger.debug(
                "%s: yielded %d rows (file rows %d..%d, intra-file want %d)",
                path, taken_from_file, file_start, file_end, want_from_file,
            )

    def extract_text(self, row: dict) -> str:
        return self._extract_text(row)

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        columns = {k: v for k, v in row.items() if k not in self.exclude_columns}
        return Record(
            row_id=row_id,
            source_row_id=0,
            chunk_id=chunk_id,
            chunk_index=0,
            text=self.extract_text(row),
            columns=columns,
        )
