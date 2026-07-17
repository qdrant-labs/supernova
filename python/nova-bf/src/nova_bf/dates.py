"""Datetime payload support for filters.

A corpus/queries column is treated as a datetime ONLY when explicitly declared
in `corpus.date_fields` / `queries.date_fields` (see config.py) — there is no
type sniffing, so a plain string column is never silently reinterpreted as a
date. Every declared date field is normalized to **int64 epoch microseconds**
at load time:

  - corpus/query columns are converted right after each file is read
    (`compute.py`),
  - static `range` bound literals in the YAML are converted at config load
    (`config.py`),

so all downstream range logic — the CPU `filters.py` path AND the GPU-native
range path in `compute.py` — operates on plain numbers, unchanged. Epoch
microseconds is also exactly Qdrant's own internal datetime representation
(i64 µs since the Unix epoch), so a `range` over a declared date field is
value-for-value comparable to a Qdrant `DatetimeRange` — range parity for free.

Supported per-field `format`s:
  - "rfc3339" (default): ISO-8601 / RFC-3339 timestamps parsed as UTC, e.g.
    "2013-05-18T05:48:54Z" (also accepts an explicit offset). Naive strings
    (no `Z`/offset) are interpreted as UTC.
  - "epoch_s" / "epoch_ms" / "epoch_us": the column is already numeric epoch
    at that scale; values are rescaled to microseconds.
  - any `strptime` pattern containing "%": parsed with that pattern (naive →
    UTC).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc

DEFAULT_FORMAT = "rfc3339"
_EPOCH_MULT = {"epoch_s": 1_000_000, "epoch_ms": 1_000, "epoch_us": 1}


def normalize_date_fields(raw) -> dict[str, str]:
    """`list[str] | dict[str, str | None] | None` -> `{field: format}`.

    Bare list entries, and dict entries with an empty/None format, default to
    `rfc3339`. This is the single place the two accepted YAML shapes collapse
    to one canonical mapping the rest of the code uses."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {k: (v or DEFAULT_FORMAT) for k, v in raw.items()}
    return {k: DEFAULT_FORMAT for k in raw}


def parse_scalar_epoch_us(value, fmt: str = DEFAULT_FORMAT) -> int:
    """One datetime literal (a static YAML `range` bound) -> int64 epoch µs.

    Raises `ValueError` on an unparseable value — a bad date literal in a
    config is a hard error at load, never a silent non-match at runtime."""
    if fmt in _EPOCH_MULT:
        return int(round(float(value) * _EPOCH_MULT[fmt]))
    s = str(value)
    if fmt == "rfc3339":
        # `fromisoformat` handles offsets natively; normalize a trailing `Z`
        # (only accepted directly on 3.11+) so behavior is version-stable.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(s, fmt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def to_epoch_us_array(col, fmt: str = DEFAULT_FORMAT) -> pa.Array:
    """A datetime column (arrow string / timestamp / numeric) -> an int64
    epoch-µs arrow array, nulls preserved.

    Fail-fast: a value that doesn't parse raises (via pyarrow) rather than
    becoming null — an unparseable payload date is a data error worth
    surfacing, and the reader thread in `compute.py` already turns such an
    exception into a loud run failure."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if fmt in _EPOCH_MULT:
        scaled = pc.multiply(pc.cast(col, pa.float64()), float(_EPOCH_MULT[fmt]))
        return pc.cast(pc.round(scaled), pa.int64())
    if fmt == "rfc3339":
        # Cast handles both ISO-8601 strings (parsed as UTC) and an already
        # native timestamp column; the int64 of a timestamp is µs-since-epoch
        # regardless of any tz annotation.
        ts = pc.cast(col, pa.timestamp("us", tz="UTC"))
    else:
        ts = pc.strptime(col, format=fmt, unit="us")
    return pc.cast(ts, pa.int64())


def convert_table_date_columns(table: pa.Table, date_formats: dict[str, str]) -> pa.Table:
    """Return `table` with each PRESENT declared date column replaced by its
    int64 epoch-µs form. A declared field absent from this table's schema is
    left untouched (the caller's existing missing-field handling still applies
    to filter fields); a non-date filter column is never touched."""
    if not date_formats:
        return table
    names = set(table.column_names)
    for name, fmt in date_formats.items():
        if name in names:
            idx = table.schema.get_field_index(name)
            table = table.set_column(idx, name, to_epoch_us_array(table[name], fmt))
    return table
