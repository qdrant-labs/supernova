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

from nova_embed.sources.base import DatasetSource, files_in_window
from nova_embed.models import Record
from nova_embed.registry import SOURCES

# Provenance columns stamped onto each row when include_provenance=True.
SOURCE_FILE_COLUMN = "source_file_name"
SOURCE_ROW_COLUMN = "source_row_number"


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
        prefetch_window: int | None = None,
        include_provenance: bool = False,
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
        # None → download the whole window up front (cached; fast re-runs, but needs
        # disk for every file at once). An int N → rolling: keep at most N files
        # downloaded *ahead* of the reader and delete each once consumed, so on-disk
        # staging stays ~N+1 files (bounds disk for corpora too big to fully cache,
        # and overlaps the next download with the current file's embedding). Only
        # files THIS run downloads are deleted; pre-existing cached files are kept.
        self._prefetch_window = prefetch_window
        self._include_provenance = include_provenance
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
        to_download = [
            path
            for path, _ in files_in_window(
                self._files_with_counts, self._offset, self._limit
            )
        ]

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

    def _window_files(self) -> list[tuple[str, int, int]]:
        """``[(path, num_rows, file_start)]`` for files overlapping this rank's window.

        Same membership as ``files_in_window`` (the sharding contract), plus each
        file's global start offset so the reader can compute its intra-file slice.
        """
        self._ensure_counts()
        out: list[tuple[str, int, int]] = []
        cumulative = 0
        for path, num_rows in self._files_with_counts:
            file_start = cumulative
            file_end = cumulative + num_rows
            cumulative = file_end
            if file_end > self._offset and (
                self._limit is None or file_start < self._offset + self._limit
            ):
                out.append((path, num_rows, file_start))
        return out

    def _read_columns(self, schema_names: list[str]) -> list[str] | None:
        """Columns to actually READ from each file (parquet column projection).

        Skipping excluded columns at READ time — not just dropping them from the
        output in ``format_record`` — is what keeps a fat column (e.g. ms_marco's
        ``passages``, which dwarfs the ``query`` we embed) off the wire and out of
        the Arrow→Python conversion. Returns None (read everything) when a
        ``text_template`` is used, since the template may reference arbitrary fields
        we can't safely drop.
        """
        if self.text_template is not None:
            return None
        cols = [c for c in schema_names if c not in self.exclude_columns]
        if self.text_field and self.text_field not in cols and self.text_field in schema_names:
            cols.append(self.text_field)  # never drop the field we embed
        return cols

    def _emit_file_rows(self, pf, path, intra_offset, want_from_file, start_in_file):
        """Yield up to ``want_from_file`` rows from ``pf``, starting ``intra_offset`` rows in."""
        read_cols = self._read_columns(pf.schema_arrow.names)
        taken = 0
        for batch in pf.iter_batches(batch_size=10_000, columns=read_cols):
            batch_len = len(batch)
            if intra_offset >= batch_len:  # batch entirely before our start
                intra_offset -= batch_len
                continue
            slice_len = min(batch_len - intra_offset, want_from_file - taken)
            rows = batch.slice(intra_offset, slice_len).to_pylist()
            intra_offset = 0  # we've entered the window
            if self._include_provenance:
                base = start_in_file + taken  # file-local index of this slice's first row
                for j, row in enumerate(rows):
                    row[SOURCE_FILE_COLUMN] = path
                    row[SOURCE_ROW_COLUMN] = base + j
            yield from rows
            taken += len(rows)
            if taken >= want_from_file:
                break

    def _file_providers(self, window):
        """Yield ``(path, num_rows, file_start, ParquetFile, done_fn)`` per window file.

        Three strategies, chosen from the prefetch config:
          - rolling (prefetch + prefetch_window): a daemon thread downloads window
            files in order, at most ``prefetch_window`` ahead (the bounded queue is
            the backpressure); the reader calls ``done_fn`` to delete each file it
            downloaded once consumed → on-disk staging stays ~window+1 files, and the
            next download overlaps the current file's embedding.
          - eager (prefetch, no window): download the whole window up front, read local.
          - none: open each file remotely over HfFileSystem (the slow range-request path).
        """
        import pyarrow.parquet as pq

        if self._prefetch and self._prefetch_window:
            import os
            import queue
            import threading

            from huggingface_hub import hf_hub_download

            os.makedirs(self._prefetch_dir, exist_ok=True)
            ready: queue.Queue = queue.Queue(maxsize=self._prefetch_window)
            stop = threading.Event()

            def downloader():
                for path, _, _ in window:
                    if stop.is_set():
                        return
                    expected = os.path.join(self._prefetch_dir, path)
                    cached = os.path.exists(expected)
                    if cached:
                        local = expected
                    else:
                        logger.info("Downloading datasets/%s/%s", self.dataset_name, path)
                        local = hf_hub_download(
                            repo_id=self.dataset_name,
                            filename=path,
                            repo_type="dataset",
                            local_dir=self._prefetch_dir,
                        )
                    ready.put((path, local, cached))  # blocks when window full → bounds disk
                ready.put(None)

            threading.Thread(target=downloader, daemon=True).start()
            info = {p: (n, s) for p, n, s in window}
            try:
                while True:
                    item = ready.get()
                    if item is None:
                        return
                    path, local, cached = item
                    num_rows, file_start = info[path]

                    def done(local=local, cached=cached):
                        if not cached:  # only delete what we fetched this run; keep prior caches
                            try:
                                os.remove(local)
                            except OSError:
                                pass

                    yield path, num_rows, file_start, pq.ParquetFile(local), done
            finally:
                stop.set()

        if self._prefetch:
            if not self._local_paths:
                self._prefetch_files()
            for path, num_rows, file_start in window:
                yield path, num_rows, file_start, pq.ParquetFile(self._local_paths[path]), lambda: None
            return

        for path, num_rows, file_start in window:
            pf = pq.ParquetFile(f"datasets/{self.dataset_name}/{path}", filesystem=self._fs)
            yield path, num_rows, file_start, pf, lambda: None

    def stream(self) -> Iterator[dict]:
        offset = self._offset
        limit = self._limit
        rows_yielded = 0

        for path, num_rows, file_start, pf, done in self._file_providers(self._window_files()):
            if limit is not None and rows_yielded >= limit:
                done()
                break
            intra_offset = max(0, offset - file_start)
            available = num_rows - intra_offset
            want = (
                min(available, limit - rows_yielded) if limit is not None else available
            )
            if want <= 0:
                done()
                continue
            # start_in_file == intra_offset: file-local index where our window begins
            for row in self._emit_file_rows(pf, path, intra_offset, want, intra_offset):
                yield row
                rows_yielded += 1
            done()

    def extract_text(self, row: dict) -> str:
        return self._extract_text(row)

    def format_record(self, row: dict) -> Record:
        columns = {k: v for k, v in row.items() if k not in self.exclude_columns}
        return Record(text=self.extract_text(row), columns=columns)
