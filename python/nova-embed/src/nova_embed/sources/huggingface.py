"""
File-level sharded reader for HuggingFace datasets stored as native parquet shards.

Lists every parquet file in an HF dataset repo, reads each file's footer to
get its row count, and yields rows in a contiguous (offset, limit) window.
Because it operates at the parquet level (not the `datasets` library level)
it avoids the `IterableDataset.skip(N).take(M)` silent-no-op bug that bites
native-parquet datasets like HuggingFaceTB/dclm-edu (skip > ~1.5M restarts
from offset 0).
"""

from __future__ import annotations

import fnmatch
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from nova_embed.sources.base import DatasetSource
from nova_embed.models import Record
from nova_embed.registry import SOURCES


def _build_text_extractor(text_field: str | None, text_template: str | None):
    """
    Returns a function that extracts text from a row.
    - text_template: format string like "{title}: {abstract}"
    - text_field: single field name (fallback)
    """
    if text_template:

        def extract(row: dict) -> str:
            return text_template.format(**row)

        return extract

    if text_field:

        def extract(row: dict) -> str:
            val = row.get(text_field)
            if val is None:
                raise ValueError(f"Row is missing text field '{text_field}'")
            return val

        return extract

    raise ValueError("Must specify either text_field or text_template")


def _filter_paths(paths: list[str], pattern: str | list[str] | None) -> list[str]:
    """
    Apply a path filter to a list of HF parquet paths.

    Patterns:
      - None: pass-through.
      - "regex:<expr>": treat <expr> as a Python regex (re.search).
      - any other string: glob (fnmatch).
      - list of patterns: union (a path matches if it matches any pattern).
    """
    if pattern is None:
        return list(paths)
    if isinstance(pattern, list):
        out: list[str] = []
        seen: set[str] = set()
        for sub in pattern:
            for p in _filter_paths(paths, sub):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out
    if pattern.startswith("regex:"):
        rx = re.compile(pattern[len("regex:") :])
        return [p for p in paths if rx.search(p)]
    return fnmatch.filter(paths, pattern)


logger = logging.getLogger(__name__)


# "huggingface_parquet" is a legacy alias kept so existing configs keep working.
@SOURCES.register("huggingface", "huggingface_parquet")
class HuggingFaceSource(DatasetSource):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str | None = "text",
        text_template: str | None = None,
        exclude_columns: list[str] | None = None,
        offset: int | None = None,
        limit: int | None = None,
        total_rows_override: int | None = None,
        path_filter: str | None = None,
        metadata_workers: int = 4,
        prefetch: bool = False,
        prefetch_dir: str = "/tmp/nova_embed_parquet",
    ):
        """
        Args:
            dataset_name: HF Hub repo id, e.g. "HuggingFaceTB/dclm-edu".
            split: HF split name. The path_filter (or default split-name match)
                determines which files are read.
            offset / limit: applied as a row-window across all selected files.
            total_rows_override: skip the metadata sweep at construction time by
                trusting this number; metadata is still read per-file as we go.
            path_filter: substring filter on parquet file paths (e.g. "train/").
                Defaults to filtering by the split name when present in paths.
            metadata_workers: parallelism for the per-file footer fetches. Keep
                this low (default 4) to avoid bursting HF resolver rate limits
                when many jobs start simultaneously.
            prefetch: download parquet files for this rank's window to local disk
                before streaming. Eliminates per-batch HTTP range requests (422
                row groups × 14 columns = thousands of requests per file) at the
                cost of a one-time sequential download (~2.5 GB/file). Strongly
                recommended for multi-job runs.
            prefetch_dir: local directory for downloaded parquet files.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.text_template = text_template
        self.exclude_columns = set(exclude_columns or [])
        self._offset = offset or 0
        self._limit = limit
        self._total_rows_override = total_rows_override
        self._metadata_workers = metadata_workers
        self._prefetch = prefetch
        self._prefetch_dir = prefetch_dir
        self._local_paths: dict[str, str] = {}
        self._extract_text = _build_text_extractor(text_field, text_template)

        from huggingface_hub import HfApi, HfFileSystem

        self._fs = HfFileSystem()
        api = HfApi()
        all_files = api.list_repo_files(dataset_name, repo_type="dataset")

        all_parquets = sorted(f for f in all_files if f.endswith(".parquet"))
        # explicit path_filter wins (glob; "regex:..." for regex; list = union of patterns).
        # otherwise fall back to filtering by split name when it appears in paths
        # (e.g. "train/0.parquet" or "data/train-...").
        if path_filter is not None:
            parquet_paths = _filter_paths(all_parquets, path_filter)
            if not parquet_paths:
                raise ValueError(
                    f"path_filter={path_filter!r} matched 0 files in {dataset_name}. "
                    f"Sample of available paths: {all_parquets[:5]}"
                )
        elif any(split in p for p in all_parquets):
            parquet_paths = [p for p in all_parquets if split in p]
        else:
            parquet_paths = all_parquets

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

        def fetch(path: str) -> tuple[str, int | None]:
            url = f"datasets/{self.dataset_name}/{path}"
            last_err: Exception | None = None
            for attempt in range(6):
                try:
                    pf = pq.ParquetFile(url, filesystem=self._fs)
                    return path, pf.metadata.num_rows
                except Exception as e:
                    last_err = e
                    is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
                    if is_rate_limit:
                        # HF rate-limit windows are minutes-long, so back off harder.
                        wait = min(5 * 2**attempt, 120)
                        reason = "rate limited"
                    else:
                        # Generic flake (5xx, connection reset, DNS hiccup, etc.).
                        # These are usually one-shot — a couple of seconds is enough.
                        wait = min(2 * 2**attempt, 30)
                        reason = f"transient error ({type(e).__name__})"
                    logger.warning(
                        "%s reading footer for %s (attempt %d/6), retrying in %ds: %s",
                        reason,
                        path,
                        attempt + 1,
                        wait,
                        e,
                    )
                    time.sleep(wait)
            logger.warning(
                "Giving up on footer read for %s after 6 retries: %s", path, last_err
            )
            return path, None

        logger.info(
            "Reading parquet footers for %d files (parallel=%d)...",
            len(self._parquet_paths),
            self._metadata_workers,
        )
        with ThreadPoolExecutor(max_workers=self._metadata_workers) as ex:
            results = list(ex.map(fetch, self._parquet_paths))

        failed = [p for p, n in results if n is None]
        if failed:
            # Silently dropping files would corrupt the offset table -- offsets are
            # derived from cumulative sum of file row counts. Better to fail loud
            # so the user knows their slice is incomplete.
            raise RuntimeError(
                f"Footer read failed for {len(failed)}/{len(results)} parquet files in "
                f"{self.dataset_name}. First failures: {failed[:5]}. "
                "Retry, or pass a tighter path_filter to skip them explicitly."
            )

        # zero-row files are real and ok (just empty), but we drop them from the
        # offset table to keep the math clean.
        self._files_with_counts = [(p, n) for p, n in results if n > 0]
        logger.info(
            "Indexed %d parquet files, %d total rows",
            len(self._files_with_counts),
            sum(n for _, n in self._files_with_counts),
        )

    def list_files(self) -> list[tuple[str, int]]:
        """Public accessor for (path, row_count) pairs. Used by `--list-files`."""
        self._ensure_counts()
        return list(self._files_with_counts)

    def get_total_rows(self) -> int:
        if self._total_rows_override is not None:
            return self._total_rows_override
        self._ensure_counts()
        return sum(n for _, n in self._files_with_counts)

    def _prefetch_files(self) -> None:
        """Download only the parquet files overlapping this rank's window to local disk.

        Uses ``huggingface_hub.hf_hub_download`` rather than ``HfFileSystem.get``
        because the former honours ``HF_HUB_ENABLE_HF_TRANSFER=1`` and falls
        through to the Rust hf_transfer client for multi-part parallel
        downloads — typically 5-10x faster on EC2 than fsspec's single-stream
        HTTP. With the env unset it uses standard requests at the same speed
        as before.
        """
        import os
        from pathlib import Path
        from huggingface_hub import hf_hub_download

        self._ensure_counts()
        offset = self._offset
        limit = self._limit

        cumulative = 0
        to_download = []
        for path, num_rows in self._files_with_counts:
            file_end = cumulative + num_rows
            if file_end > offset and (limit is None or cumulative < offset + limit):
                to_download.append(path)
            cumulative = file_end

        Path(self._prefetch_dir).mkdir(parents=True, exist_ok=True)

        for path in to_download:
            # hf_hub_download with local_dir writes to `{local_dir}/{path}` —
            # preserves the repo path structure, so e.g.
            # "data/train-00046.parquet" lands at
            # "{prefetch_dir}/data/train-00046.parquet".
            expected_local = os.path.join(self._prefetch_dir, path)
            if os.path.exists(expected_local):
                logger.info("Already cached: %s", path)
                local_path = expected_local
            else:
                logger.info(
                    "Downloading datasets/%s/%s -> %s",
                    self.dataset_name, path, expected_local,
                )
                local_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=path,
                    repo_type="dataset",
                    local_dir=self._prefetch_dir,
                )
                logger.info(
                    "Downloaded %s (%.1f GB)",
                    path, os.path.getsize(local_path) / 1e9,
                )
            self._local_paths[path] = local_path

    def stream(self) -> Iterator[dict]:
        import pyarrow.parquet as pq

        self._ensure_counts()
        if self._prefetch and not self._local_paths:
            self._prefetch_files()
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
                min(available_in_file, limit - rows_yielded)
                if limit is not None
                else available_in_file
            )
            if want_from_file <= 0:
                continue

            if path in self._local_paths:
                pf = pq.ParquetFile(self._local_paths[path])
            else:
                pf = pq.ParquetFile(
                    f"datasets/{self.dataset_name}/{path}", filesystem=self._fs
                )
            taken_from_file = 0

            for batch in pf.iter_batches(batch_size=10_000):
                batch_len = len(batch)

                # batch is entirely before the intra-file offset
                if intra_offset >= batch_len:
                    intra_offset -= batch_len
                    continue

                remaining_in_file = want_from_file - taken_from_file
                slice_len = min(batch_len - intra_offset, remaining_in_file)

                # 1. Zero-copy Arrow slicing before converting to Python list
                rows = batch.slice(intra_offset, slice_len).to_pylist()

                # Reset intra_offset since we've now entered the window
                intra_offset = 0

                yield from rows

                taken_from_file += len(rows)
                rows_yielded += len(rows)

                if taken_from_file >= want_from_file:
                    break

            logger.debug(
                "%s: yielded %d rows (file rows %d..%d, intra-file want %d)",
                path,
                taken_from_file,
                file_start,
                file_end,
                want_from_file,
            )

    def extract_text(self, row: dict) -> str:
        return self._extract_text(row)

    def format_record(self, row: dict) -> Record:
        columns = {k: v for k, v in row.items() if k not in self.exclude_columns}
        return Record(text=self.extract_text(row), columns=columns)
