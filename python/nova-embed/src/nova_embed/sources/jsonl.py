"""
File-sharded reader for HuggingFace datasets stored as native JSON Lines shards.

Unlike the parquet source (which exploits parquet footers to map a global row
window to files with a handful of cheap HTTP requests), jsonl files carry NO
footer and NO row index: the only way to count a file's rows is to read the
whole file. So this source does not do row-window sharding at all — it shards by
*file*. Each rank is assigned a contiguous block of whole files
(`files_for_shard`) and streams them start to finish. This is what lets a
dataset like MedRAG/pubmed — whose full corpus lives as 1,166 `.jsonl` chunks on
`main`, with only a truncated auto-parquet preview elsewhere — be embedded
end to end without a conversion step.

Trade-offs versus the parquet source, by design:
  - No column projection: jsonl has no columnar layout, so every field of every
    row comes off the wire (exclude_columns trims the OUTPUT, not the transfer).
  - File-granular sharding: rank runtime is only as balanced as file sizes, and
    there is no mid-file resume (a re-run redoes whole files — pair with
    pipeline.content_addressed_files for idempotent output).
  - get_total_rows() is intentionally unsupported (would require scanning the
    corpus); the CLI's file-shard path never calls it.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import queue
import threading
from typing import Iterator

from nova_embed.sources.base import (
    DatasetSource,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    apply_record_projection,
    files_for_shard,
    filter_paths,
)
from nova_embed.models import Record
from nova_embed.registry import SOURCES

logger = logging.getLogger(__name__)

# Extensions we treat as JSON Lines (one JSON object per line). Only .jsonl —
# NOT bare .json, which is typically a single array/object, not line-delimited:
# reading such a file line-by-line would fail every json.loads and, under the
# default on_bad_line="skip", produce zero rows with no error (silent success).
# `.jsonl.gz` is decompressed transparently by the reader.
_JSONL_SUFFIXES = (".jsonl", ".jsonl.gz")


@SOURCES.register("jsonl", "huggingface_jsonl")
class JsonlSource(DatasetSource):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        revision: str | None = None,
        render_columns: dict[str, str] | None = None,
        required_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        path_filter: str | None = None,
        prefetch: bool = True,
        prefetch_dir: str = "/tmp/nova_embed_jsonl",
        prefetch_window: int | None = 2,
        include_provenance: bool = False,
        on_bad_line: str = "skip",
    ):
        """
        Args:
            dataset_name: HF Hub repo id, e.g. "MedRAG/pubmed".
            split: HF split name. Used to filter file paths when they carry the
                split name and no explicit path_filter is given.
            revision: git revision (branch/tag/commit SHA) to read. Defaults to
                the repo's default branch. Pin a commit SHA for reproducibility.
            render_columns: derived columns composed via format templates, e.g.
                {"combined": "{title}: {content}"}. Rendered before
                exclude_columns, so an embedder entry can name them as
                input_column and they land in the output.
            required_columns: columns that must survive projection. jsonl has no
                read-time projection, so this is informational only (the whole
                row is read regardless); kept for interface symmetry.
            exclude_columns: columns dropped from the OUTPUT record. Does not
                reduce transfer — jsonl reads every field.
            path_filter: glob / "regex:..." / list-of-patterns over file paths
                (see filter_paths). Defaults to filtering by split name when it
                appears in paths.
            prefetch: download each file to local disk before reading (uses
                hf_hub_download, honoring HF_HUB_ENABLE_HF_TRANSFER). Strongly
                recommended; the alternative streams over HfFileSystem.
            prefetch_dir: local staging directory for downloaded files.
            prefetch_window: with prefetch, keep at most N files downloaded ahead
                of the reader, deleting each once consumed (bounds disk, overlaps
                the next download with the current file's embedding). None =
                download this rank's whole assignment up front.
            include_provenance: stamp SOURCE_FILE_COLUMN (repo path) and
                SOURCE_ROW_COLUMN (0-based file-local record ordinal, counting
                only emitted rows — parity with the parquet source) onto each row.
            on_bad_line: "skip" (warn and drop) or "error" (raise) on a line
                that is not valid JSON.
        """
        if on_bad_line not in ("skip", "error"):
            raise ValueError(
                f"on_bad_line must be 'skip' or 'error', got {on_bad_line!r}"
            )
        self.dataset_name = dataset_name
        self.split = split
        self.revision = revision
        self.render_columns = dict(render_columns or {})
        self.required_columns = set(required_columns or [])
        self.exclude_columns = set(exclude_columns or [])
        self._prefetch = prefetch
        self._prefetch_dir = prefetch_dir
        self._prefetch_window = prefetch_window
        self._include_provenance = include_provenance
        self._on_bad_line = on_bad_line

        from huggingface_hub import HfApi, HfFileSystem

        self._fs = HfFileSystem()
        api = HfApi()
        all_files = api.list_repo_files(
            dataset_name, repo_type="dataset", revision=revision
        )
        all_jsonl = sorted(f for f in all_files if f.endswith(_JSONL_SUFFIXES))

        if path_filter is not None:
            paths = filter_paths(all_jsonl, path_filter)
            if not paths:
                raise ValueError(
                    f"path_filter={path_filter!r} matched 0 jsonl files in "
                    f"{dataset_name}. Sample of available paths: {all_jsonl[:5]}"
                )
        elif any(split in p for p in all_jsonl):
            paths = [p for p in all_jsonl if split in p]
        else:
            paths = all_jsonl

        if not paths:
            raise ValueError(
                f"No .jsonl files found in {dataset_name} matching split={split!r} / "
                f"path_filter={path_filter!r}. This source reads JSON Lines only — "
                f"for parquet datasets use type: huggingface."
            )
        self._paths = paths
        # Files this rank owns. Defaults to ALL files (single-process run); the
        # CLI narrows it via set_file_shard for a fleet.
        self._my_files = list(paths)

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def set_file_shard(self, job_rank: int, num_jobs: int) -> None:
        """Assign this rank a contiguous block of whole files.

        The presence of this method is how the CLI detects a file-sharded source
        and skips the row-window math entirely (jsonl has no cheap row index).
        """
        self._my_files = files_for_shard(self._paths, job_rank, num_jobs)
        logger.info(
            "Rank %d/%d owns %d of %d jsonl files",
            job_rank,
            num_jobs,
            len(self._my_files),
            len(self._paths),
        )

    def list_files(self) -> list[tuple[str, int | None]]:
        """(path, row_count) pairs; counts are None (unknown without scanning).

        Used by the dry-run planner to show the file partition.
        """
        return [(p, None) for p in self._paths]

    def files_for_rank(self, job_rank: int, num_jobs: int) -> list[str]:
        """Which files a given rank would own — for the dry-run planner."""
        return files_for_shard(self._paths, job_rank, num_jobs)

    def get_total_rows(self) -> int:
        raise NotImplementedError(
            "JsonlSource shards by file, not by row window: an exact row total "
            "would require scanning the whole corpus (jsonl has no footer). The "
            "CLI's file-shard path does not call this."
        )

    # --- file providers (mirrors the HF source's three strategies) ---

    def _open_lines(self, local_or_fh, path: str) -> Iterator[str]:
        """Yield decoded text lines from a local path or open binary handle,
        decompressing transparently for .gz paths."""
        gz = path.endswith(".gz")
        if isinstance(local_or_fh, str):
            if gz:
                fh = gzip.open(local_or_fh, "rt", encoding="utf-8")
            else:
                fh = open(local_or_fh, "rt", encoding="utf-8")
        else:
            raw = gzip.GzipFile(fileobj=local_or_fh) if gz else local_or_fh
            fh = io.TextIOWrapper(raw, encoding="utf-8")
        with fh:
            yield from fh

    def _file_providers(self) -> Iterator[tuple[str, object, callable]]:
        """Yield (path, local_path_or_binary_fh, done_fn) for each owned file.

        Three strategies from the prefetch config, matching the parquet source:
          - rolling (prefetch + prefetch_window): a daemon downloads ahead, at
            most `prefetch_window` files; done_fn deletes each once consumed.
          - eager (prefetch, no window): download each file, then read local.
          - none: open remotely over HfFileSystem (slow range-request path).
        """
        if self._prefetch and self._prefetch_window:
            from huggingface_hub import hf_hub_download

            os.makedirs(self._prefetch_dir, exist_ok=True)
            ready: queue.Queue = queue.Queue(maxsize=self._prefetch_window)
            stop = threading.Event()

            def downloader():
                for path in self._my_files:
                    if stop.is_set():
                        return
                    expected = os.path.join(self._prefetch_dir, path)
                    cached = os.path.exists(expected)
                    if cached:
                        local = expected
                    else:
                        logger.info(
                            "Downloading datasets/%s/%s", self.dataset_name, path
                        )
                        local = hf_hub_download(
                            repo_id=self.dataset_name,
                            filename=path,
                            repo_type="dataset",
                            revision=self.revision,
                            local_dir=self._prefetch_dir,
                        )
                    ready.put((path, local, cached))  # blocks when full → bounds disk
                ready.put(None)

            threading.Thread(target=downloader, daemon=True).start()
            try:
                while True:
                    item = ready.get()
                    if item is None:
                        return
                    path, local, cached = item

                    def done(local=local, cached=cached):
                        if not cached:  # only delete what we fetched this run
                            try:
                                os.remove(local)
                            except OSError:
                                pass

                    yield path, local, done
            finally:
                stop.set()
            return

        if self._prefetch:
            from huggingface_hub import hf_hub_download

            os.makedirs(self._prefetch_dir, exist_ok=True)
            for path in self._my_files:
                expected = os.path.join(self._prefetch_dir, path)
                if os.path.exists(expected):
                    local = expected
                else:
                    logger.info("Downloading datasets/%s/%s", self.dataset_name, path)
                    local = hf_hub_download(
                        repo_id=self.dataset_name,
                        filename=path,
                        repo_type="dataset",
                        revision=self.revision,
                        local_dir=self._prefetch_dir,
                    )
                yield path, local, lambda: None
            return

        for path in self._my_files:
            url = f"datasets/{self.dataset_name}/{path}"
            if self.revision:
                url = f"datasets/{self.dataset_name}@{self.revision}/{path}"
            fh = self._fs.open(url, "rb")
            yield path, fh, lambda: None

    def stream(self) -> Iterator[dict]:
        for path, handle, done in self._file_providers():
            # record_idx counts only EMITTED rows (skipping blank/bad lines), so
            # SOURCE_ROW_COLUMN is the data-record ordinal — same semantics as
            # the parquet source's row number. line_no is the physical line,
            # kept for error/warning messages where it's easier to locate.
            record_idx = 0
            try:
                for line_no, line in enumerate(self._open_lines(handle, path)):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        if self._on_bad_line == "error":
                            raise ValueError(
                                f"invalid JSON at {path}:{line_no}: {e}"
                            ) from e
                        logger.warning("skipping bad JSON line %s:%d: %s", path, line_no, e)
                        continue
                    if self._include_provenance:
                        row[SOURCE_FILE_COLUMN] = path
                        row[SOURCE_ROW_COLUMN] = record_idx
                    record_idx += 1
                    yield row
            finally:
                done()

    def format_record(self, row: dict) -> Record:
        return apply_record_projection(row, self.render_columns, self.exclude_columns)
