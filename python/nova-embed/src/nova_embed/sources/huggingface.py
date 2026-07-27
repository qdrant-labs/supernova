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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from nova_embed.sources.base import (
    DatasetSource,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    apply_record_projection,
    files_in_window,
    filter_paths,
)
from nova_embed.models import Record
from nova_embed.registry import SOURCES

logger = logging.getLogger(__name__)


# "huggingface_parquet" is a legacy alias kept so existing configs keep working.
@SOURCES.register("huggingface", "huggingface_parquet")
class HuggingFaceSource(DatasetSource):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        render_columns: dict[str, str] | None = None,
        required_columns: list[str] | None = None,
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
            render_columns: derived columns composed from other fields via
                format templates, e.g. {"combined": "{title}: {abstract}"}.
                Rendered per row BEFORE exclude_columns filtering, so embedder
                entries can use them as input_column and they land in the
                output parquet like any other column.
            required_columns: columns that must never be dropped by the parquet
                read projection, whatever exclude_columns says. The CLI injects
                the configured input columns here so the field being embedded
                always survives.
            offset / limit: applied as a row-window across all selected files.
            total_rows_override: trust this number instead of summing every
                file's footer. THE knob for fleet runs: with it set, a rank
                never does the full-dataset footer sweep — it reads footers in
                path order only until its own (offset, limit) window is covered
                (see _ensure_counts), cutting HF requests per rank from
                O(all files) to O(files before the window's end). The total is
                printed in every run's "Indexed ... total rows" log line.
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
        self.render_columns = dict(render_columns or {})
        self.required_columns = set(required_columns or [])
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

        from huggingface_hub import HfApi, HfFileSystem

        self._fs = HfFileSystem()
        api = HfApi()
        all_files = api.list_repo_files(dataset_name, repo_type="dataset")

        all_parquets = sorted(f for f in all_files if f.endswith(".parquet"))
        # explicit path_filter wins (glob; "regex:..." for regex; list = union of patterns).
        # otherwise fall back to filtering by split name when it appears in paths
        # (e.g. "train/0.parquet" or "data/train-...").
        if path_filter is not None:
            parquet_paths = filter_paths(all_parquets, path_filter)
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
        # incremental footer index: (path, num_rows) in path order, extended on
        # demand — see _ensure_counts. _next_path_idx is the first unread path.
        self._files_with_counts: list[tuple[str, int]] = []
        self._next_path_idx = 0
        self._counts_complete = False

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def set_window(self, offset: int, limit: int | None) -> None:
        """Re-scope this source to a row window, keeping the footer index.

        Lets the CLI reuse the instance it built for `--num-jobs` row counting
        as the pipeline source, instead of building a second instance that
        re-reads every footer — at fleet scale the footer sweep is the dominant
        HF request cost, so it must happen at most once per process.
        """
        self._offset = offset or 0
        self._limit = limit
        self._local_paths = {}  # prefetch staging is window-scoped

    def _fetch_count(self, path: str) -> tuple[str, int | None]:
        import pyarrow.parquet as pq

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

    def _ensure_counts(self, through: int | None = None) -> None:
        """Extend the footer index far enough to cover global row `through`.

        ``through=None`` indexes every file. Otherwise footers are read in path
        order, in small parallel batches, and reading STOPS once the cumulative
        row count reaches ``through`` — a rank whose window ends early never
        pays for footers past it. Each footer is ~2-3 HTTP requests against
        HF's rate-limited resolve endpoint, and N ranks × all files is how a
        fleet burns through a request quota in minutes, so never read more
        than the caller needs. Results accumulate: a later, broader call
        continues where this one stopped.
        """
        if self._counts_complete:
            return
        cumulative = sum(n for _, n in self._files_with_counts)
        if through is not None and cumulative >= through:
            return

        remaining = len(self._parquet_paths) - self._next_path_idx
        logger.info(
            "Reading parquet footers (%d of %d files unread, parallel=%d%s)...",
            remaining,
            len(self._parquet_paths),
            self._metadata_workers,
            "" if through is None else f", stopping at row {through:,}",
        )
        batch_size = self._metadata_workers * 8
        with ThreadPoolExecutor(max_workers=self._metadata_workers) as ex:
            while self._next_path_idx < len(self._parquet_paths):
                batch = self._parquet_paths[
                    self._next_path_idx : self._next_path_idx + batch_size
                ]
                results = list(ex.map(self._fetch_count, batch))
                failed = [p for p, n in results if n is None]
                if failed:
                    # Silently dropping files would corrupt the offset table --
                    # offsets are derived from the cumulative sum of file row
                    # counts. Better to fail loud so the user knows their slice
                    # is incomplete.
                    raise RuntimeError(
                        f"Footer read failed for {len(failed)}/{len(results)} parquet "
                        f"files in {self.dataset_name}. First failures: {failed[:5]}. "
                        "Retry, or pass a tighter path_filter to skip them explicitly."
                    )
                self._next_path_idx += len(batch)
                for p, n in results:
                    # zero-row files are real and ok (just empty), but we drop
                    # them from the offset table to keep the math clean.
                    if n > 0:
                        self._files_with_counts.append((p, n))
                        cumulative += n
                if through is not None and cumulative >= through:
                    break

        if self._next_path_idx >= len(self._parquet_paths):
            self._counts_complete = True
        logger.info(
            "Indexed %d parquet files, %d rows%s",
            len(self._files_with_counts),
            cumulative,
            "" if self._counts_complete else " (partial index, window-bounded)",
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

    @property
    def _window_end(self) -> int | None:
        """Global row index just past this rank's window (None = unbounded)."""
        return None if self._limit is None else self._offset + self._limit

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

        self._ensure_counts(self._window_end)
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
        self._ensure_counts(self._window_end)
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
        the Arrow→Python conversion. Returns None (read everything) when
        ``render_columns`` is used, since a template may reference arbitrary
        fields we can't safely drop.
        """
        if self.render_columns:
            return None
        cols = [c for c in schema_names if c not in self.exclude_columns]
        for required in self.required_columns:
            if required not in cols and required in schema_names:
                cols.append(required)  # never drop a field we embed
        return cols

    def _emit_file_rows(self, pf, path, intra_offset, want_from_file, start_in_file):
        """Yield up to ``want_from_file`` rows from ``pf``, starting ``intra_offset`` rows in.

        Reads per ROW GROUP (a ``Table``), not via ``iter_batches`` (which emits
        ``RecordBatch``es): a fat binary column — e.g. an image struct — can
        overflow a single Arrow array and come back chunked, which a Table
        tolerates but a RecordBatch cannot ("Nested data conversions not
        implemented for chunked array outputs"). Rows are still converted in
        ≤10k-row slices to bound the Arrow→Python memory spike.
        """
        read_cols = self._read_columns(pf.schema_arrow.names)
        taken = 0
        for rg_idx in range(pf.num_row_groups):
            if taken >= want_from_file:
                break
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            if intra_offset >= rg_rows:  # row group entirely before our start
                intra_offset -= rg_rows
                continue
            table = pf.read_row_group(rg_idx, columns=read_cols)
            pos = intra_offset
            intra_offset = 0  # we've entered the window
            while pos < rg_rows and taken < want_from_file:
                slice_len = min(10_000, rg_rows - pos, want_from_file - taken)
                rows = table.slice(pos, slice_len).to_pylist()
                if self._include_provenance:
                    base = start_in_file + taken  # file-local index of this slice's first row
                    for j, row in enumerate(rows):
                        row[SOURCE_FILE_COLUMN] = path
                        row[SOURCE_ROW_COLUMN] = base + j
                yield from rows
                taken += len(rows)
                pos += slice_len

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

    def format_record(self, row: dict) -> Record:
        return apply_record_projection(row, self.render_columns, self.exclude_columns)
