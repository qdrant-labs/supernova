#!/usr/bin/env python
"""Differential harness for nova-bf's native parquet reader (`rust_fetch`).

Contract under test: for every input file the native path either returns
EXACTLY what pyarrow returns (same dtype, shape and bytes), or declines --
`UnsupportedLayout` / `CorruptParquet` from the crate, `None` from nova-bf's
`Store` -- so the caller falls back to pyarrow. Anything else is a FAIL:
different bytes, native output where pyarrow raises, an unexpected exception
type, a crash/panic, a hang (per-case timeout) or a memory blow-up.

Sections (all on by default, pick with --sections):

  matrix   pyarrow-written files over codec x page version x dictionary x
           dtype x list kind x shape, with randomized writer options
           (statistics, page index, checksums, BSS, versions, batch sizes,
           page row limits, CDC/bloom where the pyarrow version has them).
  mutate   seeded mutants of small matrix (and crafted) files: thrift-aware
           footer and page-header field edits, list dup/drop/swap, binary
           and ARROW:schema edits, page drop/dup/swap, truncation, cuts,
           splices from other files, raw byte flips.
  crafted  hand-written pages (tests/_pqw.py) for codec framing and empty
           page placements that random mutation essentially never reaches;
           what pyarrow does with each goes to <out>/crafted.tsv.
  fetch    NOVA_BF_RUST_FETCH_COLUMNS: Store.read_columns_fetched (canary-filled
           gaps, tiny tail, no padding/coalescing) vs Store.read_columns on
           mixed side-column tables, plus footer/page mutants of them.

Every dense file goes through BOTH the crate (`rust_fetch.read_dense` or
`dense_plan` + `read_dense_into` into a canary-filled buffer, local Client)
and nova-bf's `Store.read_dense` with NOVA_BF_RUST_FETCH_DENSE=1. References:
crate vs `dense_to_2d(pq.read_table(path, columns=[col])[col])`; Store vs
`dense_to_2d(Store.read_columns(path, [col])[col])` and, when the whole file
is readable, vs the dense column of a read of ALL columns (what nova-bf's
fallback actually decodes).

Verdicts, per case and check:
  identical            native == pyarrow, bit for bit
  fallback             native declined (UnsupportedLayout / CorruptParquet / None)
  guarded              crate-level only: the crate returned data pyarrow would not,
                       but nova-bf's documented `read_schema` cross-check (the crate
                       ignores ARROW:schema) declines the file -- not a failure
  fallback_unexpected  matrix only: a supported-looking pyarrow-written file was
                       declined (allowed; a coverage signal, see unexpected_fallbacks.jsonl)
  REF_ABORT            the pyarrow reference itself ran out of memory or died; the
                       pyarrow-only read is re-run in a fresh process to tell
  FAIL                 different bytes; native output where pyarrow raises; an
                       unexpected exception; a crash/hang/memory blow-up in a native
                       stage that the pyarrow-only read does not reproduce
  ERROR                a bug in this harness

Each FAIL saves the file + a JSON sidecar to <out>/fails/ (at most
--max-saved-per-sig files per FAIL signature; all are counted) and prints a
one-line repro (`--repro FILE`, also collected in fails/REPRO.txt). Also
written: <out>/crafted.tsv (pyarrow vs native per crafted case). The last
line of output is `SUMMARY {json}`; exit 0 = clean, 1 = FAIL(s),
2 = harness errors, 3 = aborted (MemAvailable below --min-avail-gb).

Run under a memory cap, e.g.:
  ulimit -v 16000000; python tests/fuzz_differential.py --size local --workers 4 --out DIR
"""

from __future__ import annotations

import argparse
import base64
import collections
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import struct
import sys
import time
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_NOVA_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from _pqw import (  # noqa: E402
    FLBA, FLOAT, I32, OPT, REP, S, STD_SCHEMA, def_levels_rle, packed_run,
    page_header, rep_levels_rle, rle_run, selem, uvar, write_file,
)

# --------------------------------------------------------------------------
# sizes
# --------------------------------------------------------------------------

SIZES = {
    "smoke": dict(matrix="smoke", mutants=150, bases=16, fetch=16, fetch_mutants=40, fetch_bases=16, timeout=60),
    "local": dict(matrix="local", mutants=40000, bases=300, fetch=1500, fetch_mutants=6000, fetch_bases=400,
                  timeout=60),
    "node": dict(matrix="node", mutants=2000000, bases=3000, fetch=30000, fetch_mutants=300000, fetch_bases=3000,
                 timeout=120),
}
ALL_SECTIONS = ("matrix", "mutate", "crafted", "fetch")
_BLOCK = 100   # consecutive mutants sharing one base file

# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def case_rng(seed, *parts) -> random.Random:
    return random.Random(_h(seed, *parts))


def np_rng(seed, *parts) -> np.random.Generator:
    return np.random.default_rng(_h(seed, "np", *parts))


def _short(e: BaseException, n=160) -> str:
    return f"{type(e).__name__}: {str(e)[:n]}"


def _so_md5() -> str:
    try:
        import rust_fetch._rust_fetch as m
        return hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    except Exception as e:  # pragma: no cover
        return f"? ({e})"


def versions() -> dict:
    import rust_fetch
    return {"python": sys.version.split()[0], "pyarrow": pa.__version__, "numpy": np.__version__,
            "rust_fetch": rust_fetch.__version__, "rust_fetch_file": rust_fetch.__file__,
            "rust_fetch_so_md5": _so_md5()}


def _writer_params() -> set:
    import inspect
    return set(inspect.signature(pq.ParquetWriter.__init__).parameters)


# --------------------------------------------------------------------------
# dense vectors / pyarrow-written files
# --------------------------------------------------------------------------

CODECS = ["none", "snappy", "gzip", "zstd", "lz4", "brotli"]   # "lz4" = LZ4_RAW on disk
KINDS = ["list", "large", "fixed"]
DTYPES = ["f32", "f16"]

_SPECIAL32 = np.array([0x80000000, 0x7FC00000, 0x7FC00001, 0xFFC00000, 0x7F800001, 0x7F800000,
                       0xFF800000, 0x00000001, 0x007FFFFF, 0x7F7FFFFF, 0x3F800000], np.uint32)
_SPECIAL16 = np.array([0x8000, 0x7E00, 0x7E01, 0xFE00, 0x7C01, 0x7C00, 0xFC00, 0x0001, 0x03FF,
                       0x7BFF, 0x3C00], np.uint16)


def make_vectors(rng: np.random.Generator, n, dim, dtype, style):
    npt = np.float16 if dtype == "f16" else np.float32
    if style == "pool":
        k = int(rng.choice([1, 2, 3, 7, 50]))
        pool = rng.standard_normal(k).astype(npt)
        v = pool[rng.integers(0, k, (n, dim))]
    elif style == "int":
        v = rng.integers(-3, 4, (n, dim)).astype(npt)
    else:
        v = rng.standard_normal((n, dim)).astype(npt)
    if style == "special" and v.size:
        bits = v.reshape(-1).view(np.uint16 if dtype == "f16" else np.uint32)
        sp = _SPECIAL16 if dtype == "f16" else _SPECIAL32
        idx = rng.integers(0, bits.size, min(bits.size, 3 * len(sp)))
        bits[idx] = sp[rng.integers(0, len(sp), idx.size)]
    return np.ascontiguousarray(v)


def dense_column(v, kind):
    n, dim = v.shape
    vt = pa.float16() if v.dtype == np.float16 else pa.float32()
    flat = pa.array(v.reshape(-1), type=vt)
    if kind == "fixed":
        return pa.FixedSizeListArray.from_arrays(flat, dim)
    offsets = np.arange(n + 1, dtype=np.int64) * dim
    if kind == "large":
        return pa.LargeListArray.from_arrays(pa.array(offsets, pa.int64()), flat)
    return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32), pa.int32()), flat)


_FIXED_SHAPES = [  # n, dim, row_group_size, data_page_size
    (0, 4, None, None), (1, 1, None, None), (1, 7, 1, 1), (2, 1, 1, 1), (3, 3, 2, 64),
    (7, 5, 3, 40), (13, 1, 5, 1), (64, 129, 10, 512), (100, 8, 7, 1), (257, 17, 64, 4096),
    (513, 8, 512, 2048), (1000, 24, 300, 2048), (2000, 2, 2000, 1 << 20), (37, 5, 13, 40),
    (5, 768, 2, 1024), (999, 3, None, 128),
]


def _random_shape(r: random.Random, small=False):
    if small:
        n = r.choice([0, 1, 2, 3, 5, 8, 13, 31, 64, 120])
        dim = r.choice([1, 2, 3, 4, 5, 7, 8, 12])
    else:
        n = r.choice([0, 1, 2, 3, 5, 9, 31, 100, 257, 1024, 3000, 10007])
        dim = r.choice([1, 2, 3, 5, 8, 15, 16, 33, 64, 127, 256, 384, 768])
        while n * dim > (4 << 20):
            n //= 2
    rg = r.choice([None, 1, 2, 3, 7, 64, 500, max(1, n // 2), max(1, n - 1)])
    dps = r.choice([None, 1, 16, 64, 100, 512, 4096, 65536])
    return (n, dim, rg, dps)


def matrix_params(size, seed):
    """Deterministic list of matrix cases (dicts)."""
    combos = [dict(codec=c, pv=pv, dic=d, dtype=dt, kind=k)
              for c in CODECS for pv in ("1.0", "2.0") for d in (False, True) for dt in DTYPES for k in KINDS]
    out = []
    extra = {"smoke": 0, "local": 0, "node": 200}[size]
    for ci, combo in enumerate(combos):
        shapes = list(_FIXED_SHAPES)
        r = case_rng(seed, "matrix-shapes", ci)
        shapes += [_random_shape(r) for _ in range(extra)]
        for si, sh in enumerate(shapes):
            out.append(dict(combo, shape=list(sh), mi=len(out)))
    if size == "smoke":
        r = case_rng(seed, "matrix-smoke")
        # every combo's first two shapes would be 288; take a stratified sample
        out = [m for m in out if m["shape"][0] <= 257]
        out = sorted(r.sample(out, 60), key=lambda m: m["mi"])
    return out


def mutation_bases(n_bases, seed):
    out = []
    for i in range(n_bases):
        r = case_rng(seed, "base", i)
        out.append(dict(codec=r.choice(CODECS[:5] + ["none", "snappy", "zstd"]), pv=r.choice(["1.0", "2.0"]),
                        dic=r.random() < 0.4, dtype=r.choice(DTYPES), kind=r.choice(KINDS),
                        shape=list(_random_shape(r, small=True)), mi=f"base{i}"))
    return out


def write_matrix_file(path, m, seed, wparams):
    """Write one matrix case; returns (resolved writer options, expect_native, column)."""
    r = case_rng(seed, "matrix-opts", m["mi"])
    g = np_rng(seed, "matrix-vals", m["mi"])
    n, dim, rg, dps = m["shape"]
    style = r.choice(["normal", "normal", "special", "pool", "int"])
    if m["dic"] and r.random() < 0.5:
        style = "pool"
    v = make_vectors(g, n, dim, m["dtype"], style)
    cols = {}
    side = r.choice(["none", "before", "after", "both", "prefix"])
    if side in ("before", "both"):
        cols["id"] = pa.array(np.arange(n, dtype=np.int64))
    if side == "prefix":
        cols["emb"] = dense_column(make_vectors(g, n, max(1, dim - 1), m["dtype"], "normal"), "list")
    cols["e"] = dense_column(v, m["kind"])
    if side in ("after", "both"):
        cols["txt"] = pa.array([f"row-{i}" for i in range(n)], pa.string())
    if side == "prefix":
        cols["e_tail"] = pa.array(np.ones(n))
    t = pa.table(cols)
    o = dict(compression=m["codec"], use_dictionary=m["dic"], data_page_version=m["pv"])
    if rg is not None:
        o["row_group_size"] = rg
    if dps is not None:
        o["data_page_size"] = dps
    o["write_statistics"] = r.random() < 0.7
    if r.random() < 0.3:
        o["write_page_index"] = True
    if r.random() < 0.2:
        o["write_page_checksum"] = True
    if r.random() < 0.15:
        o["store_schema"] = False
    if r.random() < 0.3:
        o["version"] = r.choice(["1.0", "2.4", "2.6"])
    if r.random() < 0.15:
        o["use_compliant_nested_type"] = False
    if r.random() < 0.25:
        o["write_batch_size"] = r.choice([1, 3, 7, 64, 1024])
    if m["dic"] and r.random() < 0.3:
        o["dictionary_pagesize_limit"] = r.choice([1, 64, 512, 1 << 20])
    if "max_rows_per_page" in wparams and r.random() < 0.2:
        o["max_rows_per_page"] = r.choice([1, 2, 3, 10, 100])
    if m["codec"] in ("gzip", "zstd", "brotli") and r.random() < 0.3:
        o["compression_level"] = {"gzip": r.choice([1, 9]), "zstd": r.choice([1, 3, 19]),
                                  "brotli": r.choice([1, 11])}[m["codec"]]
    bss = False
    if not m["dic"] and r.random() < 0.1:
        bss = True
        if r.random() < 0.5:
            o["use_byte_stream_split"] = ["e"] if r.random() < 0.5 else True
        else:
            o["column_encoding"] = {"e": "BYTE_STREAM_SPLIT"}
    if "use_content_defined_chunking" in wparams and r.random() < 0.1:
        o["use_content_defined_chunking"] = True
    if "bloom_filter_options" in wparams and "id" in cols and r.random() < 0.1:
        o["bloom_filter_options"] = {"id": {"ndv": max(1, n)}}
    try:
        pq.write_table(t, path, **o)
    except Exception as e:
        # an extra this pyarrow refuses for this combo: drop the extras, keep the core
        dropped = [k for k in ("bloom_filter_options", "use_content_defined_chunking", "max_rows_per_page",
                               "use_byte_stream_split", "column_encoding") if k in o]
        if not dropped:
            raise
        for k in dropped:
            o.pop(k)
        o["_dropped"] = f"{dropped}: {_short(e, 80)}"
        bss = False
        pq.write_table(t, path, **{k: v for k, v in o.items() if not k.startswith("_")})
    expect = m["codec"] != "brotli" and not bss
    return o, expect


# --------------------------------------------------------------------------
# crafted files (hand-written pages)
# --------------------------------------------------------------------------

DIM = 4
ROWS = 3
NV = ROWS * DIM
_VALS32 = np.arange(1, NV + 1, dtype=np.float32)
_VALS32.view(np.uint32)[2] = 0x7FC00001     # NaN payload
_VALS32[5] = -0.0
F16_SCHEMA = [selem("schema", nchild=1), selem("e", rep=OPT, nchild=1, conv=3, logical=3),
              selem("list", rep=REP, nchild=1), selem("element", type=FLBA, tlen=2, rep=OPT, logical=15)]
CODEC_ID = {"none": 0, "snappy": 1, "gzip": 2, "lzo": 3, "brotli": 4, "lz4h": 5, "zstd": 6, "lz4": 7}


def _levels(start, n, dim=DIM, maxdef=3):
    """rep/def levels for values [start, start+n) of fixed-width rows."""
    if start % dim == 0 and n % dim == 0:
        rep = rep_levels_rle(n, dim)
    else:
        rep = b"".join(rle_run(0 if (start + i) % dim == 0 else 1, 1, 1) for i in range(n))
    return rep, def_levels_rle(n, maxdef) if n else b""


def _v1_plain(vb, start, n, maxdef=3):
    rep, dfl = _levels(start, n, maxdef=maxdef)
    return struct.pack("<I", len(rep)) + rep + struct.pack("<I", len(dfl)) + dfl + vb


def pg_v1(vb, start, n, comp=None, usize=None, enc=0, csize=None):
    body = _v1_plain(vb, start, n)
    c = comp(body) if comp else body
    sub = S((1, I32, n), (2, I32, enc), (3, I32, 3), (4, I32, 3))
    return page_header(0, len(body) if usize is None else usize, len(c) if csize is None else csize, sub, 5) + c


def pg_v2(vb, start, n, rows, comp=None, is_compressed=None, usize=None, enc=0):
    rep, dfl = _levels(start, n)
    c = comp(vb) if comp else vb
    body = rep + dfl + c
    f = [(1, I32, n), (2, I32, 0), (3, I32, rows), (4, I32, enc), (5, I32, len(dfl)), (6, I32, len(rep))]
    if is_compressed is not None:
        f.append((7, "bool", is_compressed))
    us = len(rep) + len(dfl) + len(vb) if usize is None else usize
    return page_header(3, us, len(body), S(*f), 8) + body


def pg_dict(vb, n, comp=None, usize=None):
    c = comp(vb) if comp else vb
    return page_header(2, len(vb) if usize is None else usize, len(c), S((1, I32, n), (2, I32, 0)), 7) + c


def _f32b(a):
    return np.asarray(a, np.float32).tobytes()


def _gz(b):
    return gzip.compress(b, mtime=0)


def _zl(b):
    return zlib.compress(b)


def _rawdef(b):
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return co.compress(b) + co.flush()


def _gz_custom(flg=0, extra=b"", hcrc=None, method=8, bad_crc=False, bad_isize=False):
    def f(b):
        h = bytes([0x1F, 0x8B, method, flg]) + struct.pack("<I", 0) + bytes([0, 255]) + extra
        if flg & 2:
            h += struct.pack("<H", (zlib.crc32(h) & 0xFFFF) if hcrc is None else hcrc)
        crc = zlib.crc32(b) ^ (1 if bad_crc else 0)
        return h + _rawdef(b) + struct.pack("<II", crc & 0xFFFFFFFF, (len(b) + (1 if bad_isize else 0)) & 0xFFFFFFFF)
    return f


def _zl_custom(cmf=0x78, fdict=False, bad_adler=False):
    def f(b):
        fl = (0x20 if fdict else 0) | 0x80
        fl += (31 - ((cmf * 256 + fl) % 31)) % 31
        out = bytes([cmf, fl & 0xFF]) + (b"\0\0\0\1" if fdict else b"") + _rawdef(b)
        return out + struct.pack(">I", zlib.adler32(b) ^ (1 if bad_adler else 0))
    return f


def _pa(codec):
    return lambda b: pa.Codec(codec).compress(b, asbytes=True)


def _zstd_raw_frame(data, with_size=True, size_lie=0):
    if with_size:
        sz = len(data) + size_lie
        if sz < 256:
            hdr = struct.pack("<I", 0xFD2FB528) + bytes([0x20]) + bytes([sz])
        else:
            hdr = struct.pack("<I", 0xFD2FB528) + bytes([0x60]) + struct.pack("<H", sz - 256)
    else:
        hdr = struct.pack("<I", 0xFD2FB528) + bytes([0x00, 0x50])
    return hdr + struct.pack("<I", (len(data) << 3) | 1)[:3] + data


_ZSKIP = struct.pack("<II", 0x184D2A50, 4) + b"skip"


def _lz4_seq(lits, mlen=None, off=None):
    ll = len(lits)
    ml = None if mlen is None else mlen - 4
    out = bytearray([min(ll, 15) << 4 | (0 if ml is None else min(ml, 15))])
    if ll >= 15:
        rr = ll - 15
        while rr >= 255:
            out.append(255)
            rr -= 255
        out.append(rr)
    out += lits
    if mlen is not None:
        out += struct.pack("<H", off)
        if ml >= 15:
            rr = ml - 15
            while rr >= 255:
                out.append(255)
                rr -= 255
            out.append(rr)
    return bytes(out)


def _lz4_lit(b):
    return _lz4_seq(b)


def _snappy_lit(data):
    out = bytearray()
    for i in range(0, len(data), 60):
        part = data[i:i + 60]
        out += bytes([(len(part) - 1) << 2]) + part
    return bytes(out)


def _with_dup_tail(raw, k=4):
    """raw with its last k bytes replaced by the k before them (so a match can produce them)."""
    v = bytearray(raw)
    v[-k:] = v[-2 * k:-k]
    return bytes(v)


def _crafted_cases():
    """name -> builder(path). Builders write a file with column 'e' (3 rows x 4)."""
    C = {}
    vb = _VALS32.tobytes()

    def one(pages, codec="none", schema=STD_SCHEMA, leaf=FLOAT, chunks=None, has_dict=False, dict_len=0):
        def b(p):
            write_file(p, schema, chunks or [dict(pages=pages() if callable(pages) else pages, n_rows=ROWS,
                                                  num_values=NV, has_dict=has_dict, dict_len=dict_len)],
                       codec=CODEC_ID[codec], leaf_type=leaf)
        return b

    raw = _v1_plain(vb, 0, NV)

    def v1c(comp_bytes, codec, usize=None):
        sub = S((1, I32, NV), (2, I32, 0), (3, I32, 3), (4, I32, 3))
        return one(page_header(0, len(raw) if usize is None else usize, len(comp_bytes), sub, 5) + comp_bytes, codec)

    def v2c(comp_vals, codec, is_compressed=None):
        rep, dfl = _levels(0, NV)
        body = rep + dfl + comp_vals
        f = [(1, I32, NV), (2, I32, 0), (3, I32, ROWS), (4, I32, 0), (5, I32, len(dfl)), (6, I32, len(rep))]
        if is_compressed is not None:
            f.append((7, "bool", is_compressed))
        return one(page_header(3, len(rep) + len(dfl) + len(vb), len(body), S(*f), 8) + body, codec)

    # ---- controls
    for codec, fn in [("none", None), ("snappy", _pa("snappy")), ("gzip", _gz), ("zstd", _pa("zstd")),
                      ("lz4", _pa("lz4_raw"))]:
        C[f"ctl_{codec}_v1"] = one(pg_v1(vb, 0, NV, fn), codec)
        C[f"ctl_{codec}_v2"] = one(pg_v2(vb, 0, NV, ROWS, fn), codec)
        C[f"ctl_{codec}_v2_uncompressed_flag"] = one(pg_v2(vb, 0, NV, ROWS, None, is_compressed=False), codec)
        C[f"ctl_{codec}_split_pages"] = one(pg_v1(vb[:24], 0, 6, fn) + pg_v1(vb[24:], 6, 6, fn), codec)
    C["ctl_f16_v1"] = one(pg_v1(np.arange(NV, dtype=np.float16).tobytes(), 0, NV), schema=F16_SCHEMA, leaf=FLBA)

    # ---- gzip (codec 2) framing
    gz_cases = {
        "gzip_two_members": lambda b: _gz(b[:len(b) // 2]) + _gz(b[len(b) // 2:]),
        "gzip_member_plus_empty_member": lambda b: _gz(b) + _gz(b""),
        "gzip_trailing_zeros": lambda b: _gz(b) + b"\0" * 8,
        "gzip_trailing_garbage": lambda b: _gz(b) + b"garbage!",
        "gzip_bad_crc": _gz_custom(bad_crc=True),
        "gzip_bad_isize": _gz_custom(bad_isize=True),
        "gzip_truncated_trailer": lambda b: _gz(b)[:-8],
        "gzip_fhcrc_good": _gz_custom(2),
        "gzip_fhcrc_bad": _gz_custom(2, hcrc=0x1234),
        "gzip_fextra": _gz_custom(4, extra=struct.pack("<H", 4) + b"abcd"),
        "gzip_fname": _gz_custom(8, extra=b"name\0"),
        "gzip_reserved_flag": _gz_custom(0x20),
        "gzip_method7": _gz_custom(method=7),
        "zlib_single": _zl,
        "zlib_trailing_garbage": lambda b: _zl(b) + b"garbage!",
        "zlib_two_streams": lambda b: _zl(b[:len(b) // 2]) + _zl(b[len(b) // 2:]),
        "zlib_bad_adler": _zl_custom(bad_adler=True),
        "zlib_truncated_adler": lambda b: _zl(b)[:-4],
        "zlib_fdict": _zl_custom(fdict=True),
        "zlib_cinfo8": _zl_custom(cmf=0x88),
        "zlib_cinfo0": _zl_custom(cmf=0x08),
        "raw_deflate": _rawdef,
        "gzip_then_zlib": lambda b: _gz(b[:len(b) // 2]) + _zl(b[len(b) // 2:]),
    }
    for name, fn in gz_cases.items():
        C[f"{name}_v1"] = v1c(fn(raw), "gzip")
        C[f"{name}_v2"] = v2c(fn(vb), "gzip")
    C["gzip_usize_lie_big_v1"] = v1c(_gz(raw), "gzip", usize=len(raw) + 7)
    C["gzip_usize_lie_small_v1"] = v1c(_gz(raw), "gzip", usize=len(raw) - 4)

    # ---- zstd (codec 6) framing
    zr = _pa("zstd")
    z_cases = {
        "zstd_two_frames": lambda b: zr(b[:len(b) // 2]) + zr(b[len(b) // 2:]),
        "zstd_skippable_before": lambda b: _ZSKIP + zr(b),
        "zstd_skippable_after": lambda b: zr(b) + _ZSKIP,
        "zstd_only_skippable": lambda b: _ZSKIP,
        "zstd_empty_frame_last": lambda b: zr(b) + zr(b""),
        "zstd_trailing_zero": lambda b: zr(b) + b"\0",
        "zstd_trailing_garbage": lambda b: zr(b) + b"xx",
        "zstd_trailing_partial_magic": lambda b: zr(b) + b"\x28\xb5",
        "zstd_raw_with_size": lambda b: _zstd_raw_frame(b),
        "zstd_raw_no_size": lambda b: _zstd_raw_frame(b, False),
        "zstd_raw_no_size_two": lambda b: _zstd_raw_frame(b[:9], False) + _zstd_raw_frame(b[9:], False),
        "zstd_raw_size_lie_big": lambda b: _zstd_raw_frame(b, size_lie=4),
        "zstd_raw_size_lie_small": lambda b: _zstd_raw_frame(b, size_lie=-4),
        "zstd_truncated": lambda b: zr(b)[:-3],
    }
    for name, fn in z_cases.items():
        C[f"{name}_v1"] = v1c(fn(raw), "zstd")
        C[f"{name}_v2"] = v2c(fn(vb), "zstd")

    # ---- LZ4_RAW (codec 7) block endings
    raw_m = _with_dup_tail(raw)
    raw5 = bytearray(raw)
    raw5[-5:-1] = raw5[-9:-5]
    raw5 = bytes(raw5)
    raw12 = bytearray(raw)
    raw12[-10:-6] = raw12[-14:-10]
    raw12 = bytes(raw12)
    raw0 = bytearray(raw)
    raw0[-16:-12] = raw0[-20:-16]
    raw0 = bytes(raw0)

    def lz(name, comp, data):
        sub = S((1, I32, NV), (2, I32, 0), (3, I32, 3), (4, I32, 3))
        C[name] = one(page_header(0, len(data), len(comp), sub, 5) + comp, "lz4")
    lz("lz4_literal_only", _lz4_lit(raw), raw)
    lz("lz4_ends_with_match", _lz4_seq(raw_m[:-4], 4, 4), raw_m)
    lz("lz4_match_in_last5", _lz4_seq(raw5[:-5], 4, 4) + _lz4_seq(raw5[-1:]), raw5)
    lz("lz4_match_start_within_last12", _lz4_seq(raw12[:-10], 4, 4) + _lz4_seq(raw12[-6:]), raw12)
    lz("lz4_trailing_empty_token", _lz4_lit(raw) + b"\x00", raw)
    lz("lz4_trailing_extra_bytes", _lz4_lit(raw) + b"\x10Z", raw)
    lz("lz4_offset0", _lz4_seq(raw0[:-16], 4, 0) + _lz4_seq(raw0[-12:]), raw0)
    lz("lz4_manual_ok", _lz4_seq(raw0[:-16], 4, 4) + _lz4_seq(raw0[-12:]), raw0)
    raw_ov = raw[:20] + bytes([raw[19]]) * 4 + raw[24:]
    lz("lz4_overlapping_match_off1", _lz4_seq(raw[:20], 4, 1) + _lz4_seq(raw[24:]), raw_ov)
    lz("lz4_truncated_block", _lz4_lit(raw)[:-3], raw)
    lz("lz4_offset_past_start", _lz4_seq(raw0[:-16], 4, 4000) + _lz4_seq(raw0[-12:]), raw0)
    lz("lz4_two_literal_runs", _lz4_seq(raw[:30]) + _lz4_seq(raw[30:]), raw)
    lz("lz4_long_match_then_empty_literals", _lz4_seq(raw[:20] + b"\0" * 4, 30, 4) + _lz4_seq(b""),
       raw[:20] + b"\0" * 34)
    C["lz4_frame_under_raw_codec_v1"] = v1c(_pa("lz4")(raw), "lz4")
    # LZ4 (hadoop, codec 5)
    hadoop = struct.pack(">II", len(raw), len(_lz4_lit(raw))) + _lz4_lit(raw)
    C["lz4hadoop_framed_v1"] = v1c(hadoop, "lz4h")
    C["lz4hadoop_raw_block_v1"] = v1c(_lz4_lit(raw), "lz4h")
    C["lz4hadoop_frame_v1"] = v1c(_pa("lz4")(raw), "lz4h")

    # ---- snappy (codec 1)
    L = len(raw)
    sn = {
        "snappy_canonical": uvar(L) + _snappy_lit(raw),
        "snappy_len_padded2": bytes([L | 0x80, 0x00]) + _snappy_lit(raw),
        "snappy_len_padded5": bytes([L | 0x80, 0x80, 0x80, 0x80, 0x00]) + _snappy_lit(raw),
        "snappy_len_padded6": bytes([L | 0x80, 0x80, 0x80, 0x80, 0x80, 0x00]) + _snappy_lit(raw),
        "snappy_len_too_small": uvar(L - 4) + _snappy_lit(raw),
        "snappy_len_too_big": uvar(L + 4) + _snappy_lit(raw),
        "snappy_len_huge": uvar(1 << 31) + _snappy_lit(raw),
        "snappy_lit_long4": uvar(L) + bytes([63 << 2]) + (L - 1).to_bytes(4, "little") + raw,
        "snappy_trailing_garbage": pa.Codec("snappy").compress(raw, asbytes=True) + b"\x00",
        "snappy_truncated": pa.Codec("snappy").compress(raw, asbytes=True)[:-2],
    }
    for name, comp in sn.items():
        C[f"{name}_v1"] = v1c(comp, "snappy")
    raw_s = _with_dup_tail(raw)
    for nm, tag, off in [("copy2_off4", 2, struct.pack("<H", 4)), ("copy4_off4", 3, struct.pack("<I", 4)),
                         ("copy_off0", 2, struct.pack("<H", 0)), ("copy_off_past", 2, struct.pack("<H", 999))]:
        comp = uvar(L) + _snappy_lit(raw_s[:-4]) + bytes([((4 - 1) << 2) | tag]) + off
        sub = S((1, I32, NV), (2, I32, 0), (3, I32, 3), (4, I32, 3))
        C[f"snappy_{nm}_v1"] = one(page_header(0, L, len(comp), sub, 5) + comp, "snappy")

    # ---- codecs native must refuse
    C["brotli_v1"] = v1c(_pa("brotli")(raw), "brotli")
    C["lzo_codec_v1"] = v1c(raw, "lzo")
    for cid in (8, 99):
        def mk(cid=cid):
            def b(p):
                write_file(p, STD_SCHEMA, [dict(pages=pg_v1(vb, 0, NV), n_rows=ROWS, num_values=NV)], codec=cid)
            return b
        C[f"unknown_codec_{cid}"] = mk()
    C["none_usize_ne_csize"] = one(pg_v1(vb, 0, NV, usize=len(raw) + 4))

    # ---- empty data pages
    def E1(comp=None):
        return pg_v1(b"", 0, 0, comp)

    def E2():
        return pg_v2(b"", 0, 0, 0)

    rows = lambda s, n, comp=None: pg_v1(vb[4 * s:4 * (s + n)], s, n, comp)  # noqa: E731
    rows2 = lambda s, n: pg_v2(vb[4 * s:4 * (s + n)], s, n, n // DIM)  # noqa: E731
    emp = {
        "empty_first": lambda: E1() + rows(0, NV),
        "empty_first_v2": lambda: E2() + rows2(0, NV),
        "empty_two_first": lambda: E1() + E1() + rows(0, NV),
        "empty_mid_row_boundary": lambda: rows(0, 4) + E1() + rows(4, 8),
        "empty_mid_row_boundary_v2": lambda: rows2(0, 4) + E2() + rows2(4, 8),
        "empty_mid_after_v1_then_v2": lambda: rows(0, 4) + E2() + rows(4, 8),
        "empty_two_mid": lambda: rows(0, 4) + E1() + E1() + rows(4, 8),
        "empty_mid_row": lambda: rows(0, 6) + E1() + rows(6, 6),
        "empty_two_mid_row": lambda: rows(0, 6) + E1() + E1() + rows(6, 6),
        "empty_last": lambda: rows(0, NV) + E1(),
        "empty_last_two": lambda: rows(0, NV) + E1() + E1(),
        "empty_index_then_empty": lambda: rows(0, 4) + page_header(1, 0, 0) + E1() + rows(4, 8),
        "empty_after_index_first": lambda: page_header(1, 0, 0) + E1() + rows(0, NV),
        "empty_index_garbage": lambda: rows(0, 4) + page_header(1, 4, 4) + b"\xff" * 4 + rows(4, 8),
        "empty_gzip_mid": lambda: rows(0, 4, _gz) + E1(_gz) + rows(4, 8, _gz),
        "empty_v1_enc99": lambda: rows(0, 4) + pg_v1(b"", 0, 0, enc=99) + rows(4, 8),
        "empty_v1_usize_lie": lambda: rows(0, 4) + pg_v1(b"", 0, 0, usize=1000) + rows(4, 8),
        "empty_v1_garbage_body": lambda: rows(0, 4) + page_header(0, 5, 5, S((1, I32, 0), (2, I32, 0), (3, I32, 3),
                                                                              (4, I32, 3)), 5) + b"\xff" * 5 + rows(4, 8),
        "empty_v2_rows_lie": lambda: rows2(0, 4) + pg_v2(b"", 0, 0, 3) + rows2(4, 8),
        "empty_v2_lens_lie": lambda: rows2(0, 4) + page_header(3, 0, 0, S((1, I32, 0), (2, I32, 0), (3, I32, 0),
                                                                          (4, I32, 0), (5, I32, 10), (6, I32, 10)), 8)
        + rows2(4, 8),
    }
    gz_codec = {"empty_gzip_mid"}
    for name, fn in emp.items():
        C[name] = one(fn, "gzip" if name in gz_codec else "none")
    # with dictionaries
    dvals = _f32b([10, 20, 30, 40])
    dpg = pg_dict(dvals, 4)
    idx = bytes([2]) + packed_run([0, 1, 2, 3, 0, 1, 2, 3], 2) + rle_run(3, 4, 2)
    dict_data = pg_v1(idx, 0, NV, enc=8)
    C["dict_ctl"] = one(dpg + dict_data, has_dict=True, dict_len=len(dpg))
    C["dict_then_empty"] = one(dpg + E1() + dict_data, has_dict=True, dict_len=len(dpg))
    C["dict_then_empty_v2"] = one(dpg + E2() + dict_data, has_dict=True, dict_len=len(dpg))
    edp = pg_dict(b"", 0)
    C["empty_dict_then_plain"] = one(edp + rows(0, NV), has_dict=True, dict_len=len(edp))
    C["empty_dict_then_indices"] = one(edp + dict_data, has_dict=True, dict_len=len(edp))
    C["empty_dict_then_zero_bw_indices"] = one(edp + pg_v1(bytes([0]) + rle_run(0, NV, 0), 0, NV, enc=8),
                                               has_dict=True, dict_len=len(edp))
    C["dict_gzip_two_members"] = one(pg_dict(dvals, 4, lambda b: _gz(b[:6]) + _gz(b[6:])) + pg_v1(idx, 0, NV, _gz, enc=8),
                                     codec="gzip", has_dict=True,
                                     dict_len=len(pg_dict(dvals, 4, lambda b: _gz(b[:6]) + _gz(b[6:]))))
    # two row groups, empty page in the second
    C["two_rg_empty_first_in_rg2"] = one(None, chunks=[dict(pages=rows(0, NV), n_rows=ROWS, num_values=NV),
                                                       dict(pages=E1() + rows(0, NV), n_rows=ROWS, num_values=NV)])
    C["two_rg_empty_mid_in_rg2"] = one(None, chunks=[dict(pages=rows(0, NV), n_rows=ROWS, num_values=NV),
                                                     dict(pages=rows(0, 4) + E1() + rows(4, 8), n_rows=ROWS,
                                                          num_values=NV)])
    # f16 empty pages
    hb = np.arange(NV, dtype=np.float16).tobytes()
    C["f16_empty_mid"] = one(pg_v1(hb[:8], 0, 4) + pg_v1(b"", 0, 0) + pg_v1(hb[8:], 4, 8),
                             schema=F16_SCHEMA, leaf=FLBA)
    C["f16_empty_first"] = one(pg_v1(b"", 0, 0) + pg_v1(hb, 0, NV), schema=F16_SCHEMA, leaf=FLBA)
    return C


_CRAFTED = None


def crafted_cases():
    global _CRAFTED
    if _CRAFTED is None:
        _CRAFTED = _crafted_cases()
    return _CRAFTED


# --------------------------------------------------------------------------
# side-column (NATIVE_FETCH) tables
# --------------------------------------------------------------------------


def mixed_table(g: np.random.Generator, r: random.Random, n: int):
    cols = {
        "id": pa.array(np.arange(n, dtype=np.int64) * 7 - 3),
        "sid": pa.array([f"doc-{i}" for i in range(n)]),
        "lid": pa.array([f"L{i:06d}" * (1 + i % 3) for i in range(n)], pa.large_string()),
        "text": pa.array([" ".join(g.choice(["alpha", "beta", "gamma", "δέλτα", ""], 4)) for _ in range(n)]),
        "sparse": pa.array([{"indices": [int(x) for x in g.choice(50, k, replace=False)],
                             "values": [float(x) for x in g.random(k)]}
                            for k in g.integers(0, 6, n)],
                           pa.struct([("indices", pa.list_(pa.uint32())), ("values", pa.list_(pa.float32()))])),
        "score": pa.array(g.random(n)),
        "f32": pa.array(g.standard_normal(n).astype(np.float32)),
        "f16": pa.array(g.standard_normal(n).astype(np.float16)),
        "lang": pa.array(g.choice(["en", "de", "fr"], n)).dictionary_encode(),
        "tags": pa.array([["a", "b", "c"][: i % 4] for i in range(n)], pa.list_(pa.string())),
        "kv": pa.array([[("k", i), ("z", -i)][: 1 + i % 2] for i in range(n)], pa.map_(pa.string(), pa.int64())),
        "maybe": pa.array([None if i % 5 == 0 else i for i in range(n)], pa.int32()),
        "date": pa.array(np.datetime64("2020-01-01") + np.arange(n).astype("timedelta64[D]")),
        "ts": pa.array((np.arange(n) * 3600_000).astype("datetime64[ms]")),
        "ts_tz": pa.array((np.arange(n) * 1000).astype("datetime64[ms]"), pa.timestamp("ms", tz="UTC")),
        "dstr": pa.array([f"2024-0{1 + i % 9}-1{i % 10}" for i in range(n)]),
        "dec": pa.array([None if i % 7 == 0 else i * 13 for i in range(n)], pa.decimal128(12, 2)),
        "fbin": pa.array([bytes([i % 256]) * 16 for i in range(n)], pa.binary(16)),
        "nested": pa.array([{"a": i, "b": {"c": f"x{i}", "d": [i, i + 1][: i % 3]}} for i in range(n)]),
        "lstruct": pa.array([[{"p": i, "q": "y"}] * (i % 3) for i in range(n)]),
        "ll": pa.array([[i] * (i % 3) for i in range(n)], pa.large_list(pa.int32())),
        "fsl": pa.FixedSizeListArray.from_arrays(pa.array(g.random(n * 3)), 3),
        "flag": pa.array(g.random(n) < 0.5),
        "nul": pa.nulls(n),
        "dense": dense_column(g.standard_normal((n, 6)).astype(np.float16), "list"),
        "u8": pa.array(g.integers(0, 255, n).astype(np.uint8)),
    }
    if r.random() < 0.3:
        try:
            cols["sv"] = pa.array([f"v{i}" for i in range(n)], pa.string_view())
        except Exception:
            pass
    if r.random() < 0.2:
        try:
            cols["uuid"] = pa.ExtensionArray.from_storage(pa.uuid(), pa.array([bytes([i % 256]) * 16 for i in range(n)],
                                                                                pa.binary(16)))
        except Exception:
            pass
    names = list(cols)
    r.shuffle(names)
    keep = names if r.random() < 0.3 else r.sample(names, r.randint(3, len(names)))
    return pa.table({k: cols[k] for k in keep})


def write_fetch_file(path, idx, seed, wparams):
    r = case_rng(seed, "fetch", idx)
    g = np_rng(seed, "fetch", idx)
    n = r.choice([0, 1, 2, 5, 17, 60, 200, 200, 1000])
    t = mixed_table(g, r, n)
    o = {"compression": r.choice(["none", "snappy", "gzip", "zstd", "lz4", "brotli"])}
    if r.random() < 0.3:
        o["compression"] = {c: r.choice(["none", "snappy", "zstd", "gzip"]) for c in t.column_names}
    d = r.random()
    o["use_dictionary"] = (d < 0.6) or ([c for c in t.column_names if r.random() < 0.5] if d < 0.8 else False)
    o["data_page_version"] = r.choice(["1.0", "2.0"])
    if r.random() < 0.6:
        o["row_group_size"] = r.choice([1, 5, 17, 64, 500])
    if r.random() < 0.5:
        o["data_page_size"] = r.choice([1, 64, 128, 1024, 8192])
    o["write_statistics"] = r.random() < 0.7
    for k, p in [("write_page_index", 0.3), ("write_page_checksum", 0.2)]:
        if r.random() < p:
            o[k] = True
    if r.random() < 0.15:
        o["store_schema"] = False
    if r.random() < 0.15:
        o["use_compliant_nested_type"] = False
    if r.random() < 0.2:
        o["version"] = r.choice(["1.0", "2.4", "2.6"])
    if r.random() < 0.2:
        o["write_batch_size"] = r.choice([1, 7, 64])
    if r.random() < 0.2:
        o["dictionary_pagesize_limit"] = r.choice([64, 1024])
    if "max_rows_per_page" in wparams and r.random() < 0.2:
        o["max_rows_per_page"] = r.choice([1, 3, 50])
    if o["use_dictionary"] is False and r.random() < 0.5:
        enc = {}
        for c in t.column_names:
            typ = t.schema.field(c).type
            if typ in (pa.float32(), pa.float64(), pa.float16()) and r.random() < 0.7:
                enc[c] = "BYTE_STREAM_SPLIT"
            elif typ in (pa.int64(), pa.int32()) and r.random() < 0.7:
                enc[c] = "DELTA_BINARY_PACKED"
            elif typ in (pa.string(), pa.large_string()) and r.random() < 0.7:
                enc[c] = r.choice(["DELTA_BYTE_ARRAY", "DELTA_LENGTH_BYTE_ARRAY"])
        if enc:
            o["column_encoding"] = enc
    if "use_content_defined_chunking" in wparams and r.random() < 0.1:
        o["use_content_defined_chunking"] = True
    if "bloom_filter_options" in wparams and "sid" in t.column_names and r.random() < 0.1:
        o["bloom_filter_options"] = {"sid": {"ndv": max(1, n)}}
    try:
        pq.write_table(t, path, **o)
    except Exception:
        # an option this pyarrow refuses for these types: retry plain
        o = {k: v for k, v in o.items() if k not in ("column_encoding", "use_content_defined_chunking",
                                                      "bloom_filter_options")}
        try:
            pq.write_table(t, path, **o)
        except Exception:
            t = t.drop_columns([c for c in ("sv", "uuid") if c in t.column_names])
            pq.write_table(t, path, **o)
    names = t.column_names
    pick = r.random()
    if pick < 0.15:
        cols = list(names)
    elif pick < 0.5:
        cols = [r.choice(names)]
    else:
        cols = r.sample(names, r.randint(1, min(len(names), 5)))
    if r.random() < 0.03:
        cols = cols + ["missing_col"]
    knobs = dict(tail=r.choice([64, 64, 128, 1000, 4096, 1 << 20]), pad=r.choice([0, 0, 1, 4096]),
                 coalesce=r.choice([0, 0, 8 << 10]), canary=r.random() < 0.9)
    return {k: (v if not isinstance(v, dict) else v) for k, v in o.items()}, cols, knobs


# --------------------------------------------------------------------------
# thrift-compact walker + mutation engine
# --------------------------------------------------------------------------


class Leaf:
    __slots__ = ("kind", "path", "start", "end", "value", "hdr_end", "etype", "elems", "data_start")

    def __init__(self, kind, path, start, end, value=None, **kw):
        self.kind, self.path, self.start, self.end, self.value = kind, path, start, end, value
        self.hdr_end = kw.get("hdr_end")
        self.etype = kw.get("etype")
        self.elems = kw.get("elems")
        self.data_start = kw.get("data_start")


class ThriftError(ValueError):
    pass


def _uv(b, pos):
    v = s = 0
    while True:
        if pos >= len(b) or s > 70:
            raise ThriftError("varint")
        x = b[pos]
        pos += 1
        v |= (x & 0x7F) << s
        s += 7
        if x < 0x80:
            return v, pos


def _unzz(u):
    return (u >> 1) ^ -(u & 1)


def walk_struct(b, pos, path, out, depth=0):
    if depth > 24:
        raise ThriftError("depth")
    last = 0
    while True:
        if pos >= len(b):
            raise ThriftError("eof")
        hs = pos
        h = b[pos]
        pos += 1
        if h == 0:
            return pos
        t, d = h & 15, h >> 4
        if d:
            fid = last + d
        else:
            u, pos = _uv(b, pos)
            fid = _unzz(u)
        last = fid
        pos = walk_value(b, pos, t, path + (fid,), out, depth, hs)


def walk_value(b, pos, t, path, out, depth, field_hdr=None):
    if t in (1, 2):
        if field_hdr is not None:
            out.append(Leaf("bool", path, field_hdr, field_hdr + 1, t == 1))
            return pos
        if pos >= len(b):
            raise ThriftError("eof")
        out.append(Leaf("lbool", path, pos, pos + 1, b[pos]))
        return pos + 1
    if t == 3:
        if pos >= len(b):
            raise ThriftError("eof")
        out.append(Leaf("byte", path, pos, pos + 1, b[pos]))
        return pos + 1
    if t in (4, 5, 6):
        s = pos
        u, pos = _uv(b, pos)
        out.append(Leaf("int", path, s, pos, _unzz(u)))
        return pos
    if t == 7:
        if pos + 8 > len(b):
            raise ThriftError("eof")
        return pos + 8
    if t == 8:
        s = pos
        n, p2 = _uv(b, pos)
        if p2 + n > len(b):
            raise ThriftError("bin")
        out.append(Leaf("bin", path, s, p2 + n, bytes(b[p2:p2 + n]), data_start=p2))
        return p2 + n
    if t in (9, 10):
        s = pos
        if pos >= len(b):
            raise ThriftError("eof")
        h = b[pos]
        pos += 1
        n, et = h >> 4, h & 15
        if n == 15:
            n, pos = _uv(b, pos)
        if n > len(b):
            raise ThriftError("list size")
        hdr_end = pos
        elems = []
        for i in range(n):
            es = pos
            pos = walk_value(b, pos, et, path + (i,), out, depth + 1)
            elems.append((es, pos))
        out.append(Leaf("list", path, s, pos, n, hdr_end=hdr_end, etype=et, elems=elems))
        return pos
    if t == 11:
        n, pos = _uv(b, pos)
        if n:
            if pos >= len(b):
                raise ThriftError("eof")
            kv = b[pos]
            pos += 1
            for i in range(n):
                pos = walk_value(b, pos, kv >> 4, path + (i, 0), out, depth + 1)
                pos = walk_value(b, pos, kv & 15, path + (i, 1), out, depth + 1)
        return pos
    if t == 12:
        s = pos
        pos = walk_struct(b, pos, path, out, depth + 1)
        out.append(Leaf("struct", path, s, pos))
        return pos
    raise ThriftError(f"type {t}")


def split_file(b):
    """(footer_start, footer_end) or None."""
    if len(b) < 12 or b[-4:] != b"PAR1":
        return None
    flen = struct.unpack("<I", b[-8:-4])[0]
    fs = len(b) - 8 - flen
    if fs < 4:
        return None
    return fs, len(b) - 8


def footer_leaves(b):
    sp = split_file(b)
    if sp is None:
        return None
    fs, fe = sp
    out = []
    try:
        walk_struct(b[fs:fe], 0, (), out)
    except (ThriftError, IndexError):
        return None
    return fs, fe, out


def rebuild(body, footer, keep_len=None):
    flen = len(footer) if keep_len is None else keep_len
    return bytes(body) + bytes(footer) + struct.pack("<I", flen & 0xFFFFFFFF) + b"PAR1"


def _enc_varint_padded(u, width):
    """Unsigned varint, non-canonically padded with continuation bytes to `width` (if it fits)."""
    canon = uvar(u)
    if len(canon) >= width:
        return canon
    out = bytearray(canon)
    out[-1] |= 0x80
    out += b"\x80" * (width - len(canon) - 1) + b"\x00"
    return bytes(out)


def _zz64(v):
    v = max(-(1 << 63), min((1 << 63) - 1, v))
    return ((v << 1) ^ (v >> 63)) & ((1 << 64) - 1)


I64_EDGES = [0, 1, -1, 2, 3, 4, 7, 8, 15, 16, 255, 256, 1 << 15, (1 << 31) - 1, 1 << 31, -(1 << 31), 1 << 32,
             (1 << 62), (1 << 63) - 1, -(1 << 63)]


def _new_int(r, old, pool, flen):
    c = r.random()
    if c < 0.30:
        return r.choice(I64_EDGES)
    if c < 0.55:
        return old + r.choice([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])
    if c < 0.65:
        return r.choice([old * 2, old // 2, -old, old * 3, old + 4096, old - 4096])
    if c < 0.85 and pool:
        return r.choice(pool)
    if c < 0.92:
        return r.choice([flen, flen - 8, flen - 4, flen + 1, 4])
    return old + r.randint(-300, 300)


# ColumnMetaData int fields + PageHeader int fields get extra weight
_HOT = {5, 6, 7, 9, 10, 11, 3, 1, 2, 4}


def mut_footer_int(b, r, flen):
    fl = footer_leaves(b)
    if fl is None:
        return None
    fs, fe, leaves = fl
    ints = [x for x in leaves if x.kind in ("int", "byte")]
    if not ints:
        return None
    # bias to row-group/column-chunk/meta_data fields
    rg = [x for x in ints if x.path[:1] == (4,)]
    pick = r.choice(rg if rg and r.random() < 0.75 else ints)
    pool = [x.value for x in ints]
    nv = _new_int(r, pick.value, pool, flen)
    f = bytearray(b[fs:fe])
    if pick.kind == "byte":
        new = bytes([nv & 0xFF])
    else:
        u = _zz64(nv)
        new = _enc_varint_padded(u, pick.end - pick.start) if r.random() < 0.5 else uvar(u)
    f[pick.start:pick.end] = new
    return rebuild(b[:fs], f, keep_len=(fe - fs) if r.random() < 0.05 else None), f"footer_int{pick.path}={nv}"


def mut_footer_bool(b, r, flen):
    fl = footer_leaves(b)
    if fl is None:
        return None
    fs, fe, leaves = fl
    bools = [x for x in leaves if x.kind in ("bool", "lbool")]
    if not bools:
        return None
    pick = r.choice(bools)
    f = bytearray(b[fs:fe])
    if pick.kind == "bool":
        f[pick.start] ^= 3
    else:
        f[pick.start] = 1 if f[pick.start] != 1 else 2
    return rebuild(b[:fs], f), f"footer_bool{pick.path}"


_NAMES = [b"e", b"element", b"list", b"item", b"array", b"key_value", b"e_tuple", b"bag", b"", b"ARROW:schema",
          b"schema", b"id", b"emb", b"txt", b"e.list.element"]


def mut_footer_bin(b, r, flen):
    fl = footer_leaves(b)
    if fl is None:
        return None
    fs, fe, leaves = fl
    bins = [x for x in leaves if x.kind == "bin"]
    if not bins:
        return None
    pick = r.choice(bins)
    v = bytearray(pick.value)
    c = r.random()
    if c < 0.35:
        nv = r.choice(_NAMES)
    elif c < 0.55 and v:
        v[r.randrange(len(v))] = r.randrange(256)
        nv = bytes(v)
    elif c < 0.75:
        nv = r.choice(bins).value
    elif c < 0.85:
        nv = bytes(v[:-1])
    else:
        nv = bytes(v) + bytes([r.randrange(256)])
    f = bytearray(b[fs:fe])
    f[pick.start:pick.end] = uvar(len(nv)) + nv
    return rebuild(b[:fs], f), f"footer_bin{pick.path}={nv[:24]!r}"


def _list_hdr(n, et):
    return bytes([(n << 4) | et]) if n < 15 else bytes([0xF0 | et]) + uvar(n)


def mut_footer_list(b, r, flen):
    fl = footer_leaves(b)
    if fl is None:
        return None
    fs, fe, leaves = fl
    lists = [x for x in leaves if x.kind == "list" and x.elems]
    if not lists:
        return None
    pick = r.choice(lists)
    f = bytes(b[fs:fe])
    els = [f[s:e] for s, e in pick.elems]
    op = r.choice(["dup", "drop", "swap", "count+1", "count-1", "reverse"])
    n = len(els)
    if op == "dup":
        i = r.randrange(n)
        els.insert(i, els[i])
    elif op == "drop":
        els.pop(r.randrange(n))
    elif op == "swap" and n > 1:
        i, j = r.sample(range(n), 2)
        els[i], els[j] = els[j], els[i]
    elif op == "reverse":
        els.reverse()
    cnt = len(els) + (1 if op == "count+1" else -1 if op == "count-1" else 0)
    cnt = max(cnt, 0)
    new = _list_hdr(cnt, pick.etype) + b"".join(els)
    f2 = f[:pick.start] + new + f[pick.end:]
    return rebuild(b[:fs], f2), f"footer_list{pick.path}:{op}"


def _arrow_schema_variants(field, r):
    t = field.type
    vt = getattr(t, "value_type", pa.float32())
    dim = getattr(t, "list_size", 4)
    cands = [pa.list_(vt), pa.large_list(vt), pa.list_(pa.field("element", vt, nullable=False)),
             pa.fixed_size_list(vt, dim), pa.fixed_size_list(vt, dim + 1), pa.fixed_size_list(vt, max(1, dim - 1)),
             pa.list_(pa.float16() if vt == pa.float32() else pa.float32()), pa.list_(pa.float64()),
             pa.large_list(pa.float16()), vt, pa.struct([("v", pa.list_(vt))]), pa.list_(pa.int32()),
             pa.binary(), pa.dictionary(pa.int32(), pa.list_(vt))]
    try:
        cands.append(pa.list_view(vt))
    except Exception:
        pass
    return r.choice(cands)


def mut_arrow_schema(b, r, flen):
    fl = footer_leaves(b)
    if fl is None:
        return None
    fs, fe, leaves = fl
    keys = [x for x in leaves if x.kind == "bin" and x.value == b"ARROW:schema" and x.path[:1] == (5,)]
    if not keys:
        return None
    k = keys[0]
    vals = [x for x in leaves if x.kind == "bin" and x.path == k.path[:-1] + (2,)]
    if not vals:
        return None
    v = vals[0]
    try:
        sch = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(v.value)))
        names = sch.names
        i = names.index("e") if "e" in names and r.random() < 0.8 else r.randrange(len(names))
        c = r.random()
        if c < 0.7:
            nt = _arrow_schema_variants(sch.field(i), r)
            sch = sch.set(i, pa.field(sch.field(i).name, nt, nullable=sch.field(i).nullable))
            desc = f"{names[i]}->{nt}"
        elif c < 0.8:
            sch = sch.set(i, sch.field(i).with_name(r.choice(["e", "x", names[0]])))
            desc = f"rename {names[i]}"
        elif c < 0.9:
            sch = sch.remove(i)
            desc = f"remove {names[i]}"
        else:
            sch = sch.with_metadata({b"k": b"v"})
            desc = "metadata"
        nv = base64.b64encode(sch.serialize().to_pybytes())
    except Exception:
        nv = bytes(v.value[:-3])
        desc = "garbage"
    f = bytearray(b[fs:fe])
    f[v.start:v.end] = uvar(len(nv)) + nv
    return rebuild(b[:fs], f), f"arrow_schema:{desc}"


def page_spans(b):
    """[(hdr_start, hdr_end, body_end, leaves)] for every column chunk found via the footer."""
    fl = footer_leaves(b)
    if fl is None:
        return []
    fs, fe, leaves = fl
    chunks = collections.defaultdict(dict)
    for x in leaves:
        p = x.path
        if x.kind == "int" and len(p) == 6 and p[0] == 4 and p[2] == 1 and p[4] == 3 and p[5] in (7, 9, 11):
            chunks[(p[1], p[3])][p[5]] = x.value
    out = []
    for (rg, c), m in sorted(chunks.items()):
        if 9 not in m or 7 not in m:
            continue
        start = m[9]
        if 11 in m and 0 < m[11] < start:
            start = m[11]
        end = start + m[7]
        if not (4 <= start < end <= fs):
            continue
        pos = start
        for _ in range(4096):
            if pos >= end:
                break
            hl = []
            try:
                he = walk_struct(b, pos, (), hl)
            except (ThriftError, IndexError):
                break
            cs = [x.value for x in hl if x.kind == "int" and x.path == (3,)]
            if not cs or cs[0] < 0:
                break
            be = he + cs[0]
            if be > end:
                break
            out.append((pos, he, be, hl, (rg, c)))
            pos = be
    return out


def mut_page_hdr(b, r, flen):
    pages = page_spans(b)
    if not pages:
        return None
    hs, he, be, hl, rc = r.choice(pages)
    cands = [x for x in hl if x.kind in ("int", "bool")]
    if not cands:
        return None
    pick = r.choice(cands)
    bb = bytearray(b)
    if pick.kind == "bool":
        bb[pick.start] ^= 3
        return bytes(bb), f"page_hdr_bool{rc}{pick.path}"
    nv = _new_int(r, pick.value, [x.value for x in cands if x.kind == "int"], flen)
    new = _enc_varint_padded(_zz64(nv), pick.end - pick.start)
    bb[pick.start:pick.end] = new
    return bytes(bb), f"page_hdr{rc}{pick.path}={nv}{'' if len(new) == pick.end - pick.start else ' (shift)'}"


def mut_page_body(b, r, flen):
    pages = [p for p in page_spans(b) if p[2] > p[1]]
    if not pages:
        return None
    hs, he, be, hl, rc = r.choice(pages)
    bb = bytearray(b)
    lo, hi = he, be
    if r.random() < 0.5:
        hi = min(be, he + 16)   # levels / length prefixes / codec headers
    pos = r.randrange(lo, hi)
    _byte_op(bb, pos, r)
    return bytes(bb), f"page_body{rc}@{pos - he}"


def mut_page_struct(b, r, flen):
    pages = page_spans(b)
    if not pages:
        return None
    op = r.choice(["drop", "dup", "swap_adjacent", "move_to_end_of_chunk"])
    i = r.randrange(len(pages))
    hs, he, be, hl, rc = pages[i]
    page = b[hs:be]
    if op == "drop":
        return b[:hs] + b[be:], f"page_drop{rc}#{i}"
    if op == "dup":
        return b[:be] + page + b[be:], f"page_dup{rc}#{i}"
    same = [j for j, p in enumerate(pages) if p[4] == rc]
    if op == "swap_adjacent":
        if i + 1 < len(pages) and pages[i + 1][4] == rc and pages[i + 1][0] == be:
            nxt = b[be:pages[i + 1][2]]
            return b[:hs] + nxt + page + b[pages[i + 1][2]:], f"page_swap{rc}#{i}"
        return None
    last = pages[same[-1]]
    if last[0] == hs:
        return None
    return b[:hs] + b[be:last[2]] + page + b[last[2]:], f"page_to_end{rc}#{i}"


def _byte_op(bb, pos, r):
    op = r.randrange(6)
    if op == 0:
        bb[pos] = r.randrange(256)
    elif op == 1:
        bb[pos] = (bb[pos] + 1) & 0xFF
    elif op == 2:
        bb[pos] = (bb[pos] - 1) & 0xFF
    elif op == 3:
        bb[pos] ^= 1 << r.randrange(8)
    elif op == 4:
        bb[pos] = r.choice([0, 0xFF, 0x7F, 0x80, 0x15, 0x16, 0x19, 0x1C, 0x01, 0x08])
    else:
        bb[pos] = bb[r.randrange(len(bb))]


def mut_bytes(b, r, flen):
    bb = bytearray(b)
    sp = split_file(b)
    k = r.choice([1, 1, 1, 2, 3, 8])
    where = []
    for _ in range(k):
        if sp and r.random() < 0.6:
            pos = r.randrange(sp[0], sp[1]) if sp[1] > sp[0] else r.randrange(len(bb))
        else:
            pos = r.randrange(len(bb))
        _byte_op(bb, pos, r)
        where.append(pos)
    return bytes(bb), f"bytes@{where}"


def mut_cut(b, r, flen):
    sp = split_file(b)
    c = r.random()
    if c < 0.3 or sp is None:
        k = r.randrange(0, len(b))
        return b[:k], f"truncate@{k}"
    fs, fe = sp
    if fs <= 5:
        return None
    if c < 0.55:
        a = r.randrange(4, fs)
        k = r.randint(1, min(64, fs - a))
        return b[:a] + b[a + k:], f"cut@{a}+{k}"
    if c < 0.8:
        a = r.randrange(4, fs)
        ins = bytes(r.randrange(256) for _ in range(r.randint(1, 16)))
        return b[:a] + ins + b[a:], f"insert@{a}+{len(ins)}"
    # truncate the data region but keep the footer intact
    k = r.randrange(4, fs)
    return b[:k] + b[fs:], f"data_truncate@{k}"


def mut_splice(b, r, flen, partner):
    if not partner:
        return None
    c = r.random()
    sp, psp = split_file(b), split_file(partner)
    if c < 0.4 and sp and psp:
        # partner body with our footer, or ours with partner's footer
        if r.random() < 0.5:
            return rebuild(partner[:psp[0]], b[sp[0]:sp[1]]), "splice:partner_body+our_footer"
        return rebuild(b[:sp[0]], partner[psp[0]:psp[1]]), "splice:our_body+partner_footer"
    if c < 0.7:
        mine, theirs = page_spans(b), page_spans(partner)
        if mine and theirs:
            hs, he, be, _, rc = r.choice(mine)
            ths, the, tbe, _, _ = r.choice(theirs)
            return b[:hs] + partner[ths:tbe] + b[be:], f"splice:page_from_partner{rc}"
    a = r.randrange(len(b))
    pa_ = r.randrange(len(partner))
    k = r.randint(1, 256)
    seg = partner[pa_:pa_ + k]
    bb = bytearray(b)
    bb[a:a + len(seg)] = seg
    return bytes(bb[:len(b)]), f"splice:overwrite@{a}+{len(seg)}"


MUTATORS = [  # (weight, fn)
    (25, mut_footer_int), (15, mut_bytes), (12, mut_page_hdr), (8, mut_page_body), (6, mut_footer_list),
    (5, mut_footer_bin), (3, mut_footer_bool), (5, mut_arrow_schema), (6, mut_page_struct), (6, mut_cut),
    (7, "splice"),
]
_MW = [w for w, _ in MUTATORS]


def mutate(base: bytes, r: random.Random, partner: bytes | None):
    b = base
    log = []
    nops = r.choices([1, 2, 3], [60, 25, 15])[0]
    for _ in range(nops):
        res = None
        for _try in range(6):
            fn = r.choices([f for _, f in MUTATORS], _MW)[0]
            try:
                res = mut_splice(b, r, len(b), partner) if fn == "splice" else fn(b, r, len(b))
            except (ThriftError, IndexError, ValueError):
                res = None
            if res is not None:
                break
        if res is None:
            res = mut_bytes(b, r, len(b))
        b, desc = res
        log.append(desc)
        if len(b) < 12:
            break
    return b, log


# --------------------------------------------------------------------------
# checks (run in workers, and by --repro)
# --------------------------------------------------------------------------

ALLOWED_CRATE = None  # (UnsupportedLayout, CorruptParquet), set at init
_STAGE_CODES = {"idle": 0, "gen": 1, "crate": 2, "store": 3, "ref": 4, "fetch": 5, "fetch_ref": 6}
_STAGE_NAMES = {v: k for k, v in _STAGE_CODES.items()}
NATIVE_STAGES = {"crate", "store", "fetch"}


class Ctx:
    def __init__(self, stage_arr=None, slot=0):
        import rust_fetch
        from nova_bf import io as nio
        global ALLOWED_CRATE
        self.rf = rust_fetch
        self.nio = nio
        ALLOWED_CRATE = (rust_fetch.UnsupportedLayout, rust_fetch.CorruptParquet)
        self.clients = [rust_fetch.Client(concurrency=8, threads=2),
                        rust_fetch.Client(concurrency=4, part_size=1 << 16, threads=2)]
        self.stage_arr, self.slot = stage_arr, slot
        self.fetch_defaults = (nio._FETCH_TAIL, nio._FETCH_PAD, nio._FETCH_COALESCE)

    def stage(self, s):
        if self.stage_arr is not None:
            self.stage_arr[self.slot] = _STAGE_CODES[s]


def _ref_oom(e):
    """pyarrow failed for lack of memory (environment), not because of the file."""
    m = str(e).lower()
    return isinstance(e, (MemoryError, pa.ArrowMemoryError)) or "malloc of size" in m or "out of memory" in m


def _dense_ref(table_col):
    from nova_bf.io import dense_to_2d
    return dense_to_2d(table_col)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _describe(a):
    return f"{a.dtype}{tuple(a.shape)}"


def check_crate(ctx, path, col, opts):
    """-> (verdict, detail)"""
    rf = ctx.rf
    client = ctx.clients[opts.get("client", 0)]
    size = os.path.getsize(path) if opts.get("pass_size") else None
    ctx.stage("crate")
    try:
        if opts.get("direct"):
            plan = client.dense_plan(path, col, size)
            if plan.n_rows == 0:
                got = np.zeros((0, 0), np.float32)
            else:
                got = np.empty((plan.n_rows, plan.dim), dtype=np.dtype(plan.dtype))
                if got.nbytes <= (256 << 20):
                    got.view(np.uint8).reshape(-1).fill(0xA5)
                if plan.nbytes:
                    client.read_dense_into(plan, got.reshape(-1).view(np.uint8), opts.get("inflight", 8))
        else:
            got = rf.read_dense(client, path, col, size=size, chunks_in_flight=opts.get("inflight", 8))
    except ALLOWED_CRATE as e:
        return "fallback", _short(e)
    except MemoryError as e:
        return "fallback", "MemoryError(alloc): " + str(e)[:80]
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        return "FAIL", "crate raised unexpected " + _short(e, 300)
    ctx.stage("ref")
    try:
        ref = _dense_ref(pq.read_table(path, columns=[col])[col])
    except Exception as e:
        if _ref_oom(e):
            return "REF_ABORT", f"pyarrow reference ran out of memory: {_short(e)}"
        return _guarded(ctx, client, path, col, f"crate returned {_describe(got)} but pyarrow raised {_short(e)}")
    if not _same(got, ref):
        return _guarded(ctx, client, path, col,
                        f"crate {_describe(got)} != pyarrow {_describe(ref)}" + _first_diff(got, ref))
    return "identical", ""


def schema_guard(client, path, col):
    """nova-bf's documented cross-check (Store._read_dense_native): the crate
    does not parse ARROW:schema, so pyarrow's reading of the column type must
    be a plain list/large_list/fixed_size_list of the stored float type.
    Returns None if it passes, else the reason it declines."""
    try:
        plan = client.dense_plan(path, col, None)
    except Exception as e:
        return f"plan now fails: {_short(e)}"
    try:
        typ = pq.read_schema(path).field(col).type
    except Exception as e:
        return f"pyarrow cannot read the schema: {_short(e, 80)}"
    plain = ((pa.types.is_list(typ) or pa.types.is_large_list(typ)
              or (pa.types.is_fixed_size_list(typ) and typ.list_size >= 0
                  and (plan.n_rows == 0 or typ.list_size == plan.dim)))
             and not isinstance(typ, pa.ExtensionType)
             and typ.value_type == pa.from_numpy_dtype(np.dtype(plan.dtype)))
    return None if plain else f"pyarrow reads {col!r} as {typ}"


def _guarded(ctx, client, path, col, violation):
    """A crate-level mismatch is a FAIL unless nova-bf's schema guard declines the file."""
    why = schema_guard(client, path, col)
    if why is None:
        return "FAIL", violation + " [nova-bf schema guard passes]"
    return "guarded", f"{why} (crate alone: {violation[:100]})"


def _first_diff(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return ""
    x, y = a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)
    d = np.nonzero(x != y)[0]
    return f"; {d.size} bytes differ, first at byte {int(d[0])}" if d.size else ""


def check_store(ctx, path, col):
    nio = ctx.nio
    d = os.path.dirname(path)
    nio._NATIVE["warned"] = set()
    ctx.stage("store")
    try:
        st = nio.Store(d)
        got = st.read_dense(path, col)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        return "FAIL", "Store.read_dense raised " + _short(e, 300)
    if st._native_failed:
        return "FAIL", "Store disabled native for the run (treated a local file error as a transport failure)"
    if got is None:
        return "fallback", "Store returned None"
    ctx.stage("ref")
    st = nio.Store(d)
    try:
        ref = _dense_ref(st.read_columns(path, [col])[col])
    except Exception as e:
        if _ref_oom(e):
            return "REF_ABORT", f"pyarrow reference ran out of memory: {_short(e)}"
        return "FAIL", f"Store returned {_describe(got)} but pyarrow path read raised {_short(e)}"
    if not _same(got, ref):
        return "FAIL", f"Store {_describe(got)} != pyarrow {_describe(ref)}" + _first_diff(got, ref)
    # nova-bf's fallback decodes the dense column out of a read of all wanted
    # columns; if that read works it must agree too.
    try:
        allref = _dense_ref(st.read_columns(path, None)[col])
    except Exception:
        return "identical", ""
    if not _same(got, allref):
        return "FAIL", f"Store {_describe(got)} != dense column of an all-columns read {_describe(allref)}"
    return "identical", ""


def _bitcast_floats(arr):
    """Replace float children by their bit patterns so NaN == NaN for equality."""
    t = arr.type
    if pa.types.is_floating(t):
        it = {16: pa.uint16(), 32: pa.uint32(), 64: pa.uint64()}[t.bit_width]
        return arr.view(it)
    if pa.types.is_dictionary(t):
        return pa.DictionaryArray.from_arrays(arr.indices, _bitcast_floats(arr.dictionary))
    if pa.types.is_struct(t):
        kids = [_bitcast_floats(arr.field(i)) for i in range(t.num_fields)]
        return pa.StructArray.from_arrays(kids, fields=[pa.field(t.field(i).name, kids[i].type) for i in range(t.num_fields)],
                                          mask=arr.is_null() if arr.null_count else None)
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        cls = pa.ListArray if pa.types.is_list(t) else pa.LargeListArray
        return cls.from_arrays(arr.offsets, _bitcast_floats(arr.values),
                               mask=arr.is_null() if arr.null_count else None)
    if pa.types.is_fixed_size_list(t):
        return pa.FixedSizeListArray.from_arrays(_bitcast_floats(arr.values), t.list_size,
                                                 mask=arr.is_null() if arr.null_count else None)
    return arr


def tables_equal(a, b):
    if not a.schema.equals(b.schema, check_metadata=True):
        return False
    if a.equals(b):
        return True
    try:
        for i in range(a.num_columns):
            x = _bitcast_floats(a.column(i).combine_chunks())
            y = _bitcast_floats(b.column(i).combine_chunks())
            if not x.equals(y):
                return False
        return True
    except Exception:
        return False


def check_fetch(ctx, path, cols, knobs):
    nio = ctx.nio
    nio._NATIVE["warned"] = set()
    os.environ["NOVA_BF_RUST_FETCH_COLUMNS"] = "1"
    if knobs.get("canary", True):
        os.environ["NOVA_BF_RUST_FETCH_CANARY"] = "1"
    else:
        os.environ.pop("NOVA_BF_RUST_FETCH_CANARY", None)
    nio._FETCH_TAIL, nio._FETCH_PAD, nio._FETCH_COALESCE = knobs["tail"], knobs["pad"], knobs["coalesce"]
    d = os.path.dirname(path)
    ctx.stage("fetch")
    try:
        st = nio.Store(d)
        got = st.read_columns_fetched(path, cols)
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        return "FAIL", "read_columns_fetched raised " + _short(e, 300)
    finally:
        nio._FETCH_TAIL, nio._FETCH_PAD, nio._FETCH_COALESCE = ctx.fetch_defaults
    if st._native_failed:
        return "FAIL", "Store disabled native for the run (local file error treated as transport failure)"
    if got is None:
        return "fallback", "declined (None)"
    ctx.stage("fetch_ref")
    try:
        ref = nio.Store(d).read_columns(path, cols)
    except Exception as e:
        if _ref_oom(e):
            return "REF_ABORT", f"pyarrow reference ran out of memory: {_short(e)}"
        return "FAIL", f"fetched a {got[0].num_rows}-row table but the pyarrow path read raised {_short(e)}"
    if not tables_equal(got[0], ref):
        detail = "fetched table != pyarrow path read"
        if not got[0].schema.equals(ref.schema, check_metadata=True):
            detail += f" (schema: {got[0].schema} vs {ref.schema})"[:300]
        else:
            bad = [c for c in ref.column_names if not tables_equal(got[0].select([c]), ref.select([c]))]
            detail += f" (columns {bad}; rows {got[0].num_rows} vs {ref.num_rows})"
        return "FAIL", detail
    return "identical", ""


def run_dense_checks(ctx, path, spec):
    out = []
    if "crate" in spec["checks"]:
        out.append(("crate",) + check_crate(ctx, path, spec["col"], spec.get("crate_opts", {})))
    if "store" in spec["checks"]:
        out.append(("store",) + check_store(ctx, path, spec["col"]))
    return out


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


class Worker:
    def __init__(self, cfg, stage_arr, slot):
        self.cfg, self.slot = cfg, slot
        self.ctx = Ctx(stage_arr, slot)
        self.wparams = _writer_params()
        self.base_cache = collections.OrderedDict()
        self.fetch_meta = {}
        self.fails_dir = os.path.join(cfg["out"], "fails")

    def _crate_opts(self, r):
        return dict(client=r.choice([0, 0, 1]), direct=r.random() < 0.5, pass_size=r.random() < 0.3,
                    inflight=r.choice([1, 2, 8, 64]))

    def _base_bytes(self, spec):
        key = json.dumps(spec, sort_keys=True)
        b = self.base_cache.get(key)
        if b is not None:
            self.base_cache.move_to_end(key)
            return b
        tmp = os.path.join(self.cfg["work"], f"w{self.slot}", "base.parquet")
        if spec.get("crafted"):
            crafted_cases()[spec["crafted"]](tmp)
        elif spec.get("fetch") is not None:
            _, cols, knobs = write_fetch_file(tmp, spec["fetch"], self.cfg["seed"], self.wparams)
            self.fetch_meta[key] = (cols, knobs)
        else:
            write_matrix_file(tmp, spec, self.cfg["seed"], self.wparams)
        b = open(tmp, "rb").read()
        os.unlink(tmp)
        self.base_cache[key] = b
        if len(self.base_cache) > 256:
            self.base_cache.popitem(last=False)
        return b

    def _pre(self, path, meta):
        """Spec + provenance next to the work file BEFORE the checks run, so a
        crash/hang can still be saved with a usable repro."""
        with open(path + ".json", "w") as f:
            json.dump(meta, f, default=str)

    def run(self, task):
        sec, path = task["section"], task["path"]
        cfg = self.cfg
        self.ctx.stage("gen")
        meta = {}
        if sec == "matrix":
            m = task["m"]
            try:
                opts, expect = write_matrix_file(path, m, cfg["seed"], self.wparams)
            except Exception as e:
                return [("crate", "skip", "pyarrow cannot write: " + _short(e))], {"m": m}
            r = case_rng(cfg["seed"], "matrix-crate", m["mi"])
            spec = dict(kind="dense", col="e", checks=["crate", "store"], crate_opts=self._crate_opts(r))
            meta = {"m": m, "write_opts": opts, "expect_native": expect, "spec": spec}
            self._pre(path, meta)
            res = run_dense_checks(self.ctx, path, spec)
            if expect:
                res = [(c, "fallback_unexpected" if v == "fallback" else v, d) for c, v, d in res]
        elif sec == "crafted":
            crafted_cases()[task["name"]](path)
            spec = dict(kind="dense", col="e", checks=["crate", "store"], crate_opts={})
            meta = {"name": task["name"], "spec": spec}
            self._pre(path, meta)
            self.ctx.stage("ref")
            try:
                ref = _dense_ref(pq.read_table(path, columns=["e"])["e"])
                meta["pyarrow"] = f"ok {_describe(ref)}"
            except Exception as e:
                meta["pyarrow"] = "error " + _short(e, 100)
            res = run_dense_checks(self.ctx, path, spec)
        elif sec == "mutate":
            r = case_rng(cfg["seed"], "mutate", task["i"])
            base = self._base_bytes(task["base"])
            partner = self._base_bytes(task["partner"]) if task.get("partner") else None
            b, log = mutate(base, r, partner)
            open(path, "wb").write(b)
            spec = dict(kind="dense", col="e", checks=["crate", "store"], crate_opts=self._crate_opts(r))
            meta = {"base": task["base"], "partner": task.get("partner"), "mutations": log, "spec": spec}
            self._pre(path, meta)
            res = run_dense_checks(self.ctx, path, spec)
        elif sec == "fetch":
            if task.get("mutant") is None:
                opts, cols, knobs = write_fetch_file(path, task["i"], cfg["seed"], self.wparams)
                meta = {"write_opts": repr(opts)}
            else:
                # mutant of fetch file `fi`, read with that file's columns/knobs
                bspec = {"fetch": task["fi"]}
                base = self._base_bytes(bspec)
                cols, knobs = self.fetch_meta[json.dumps(bspec, sort_keys=True)]
                r = case_rng(cfg["seed"], "fetch-mutate", task["mutant"])
                b, log = mutate(base, r, None)
                open(path, "wb").write(b)
                meta = {"fetch_base": task["fi"], "mutations": log}
            spec = dict(kind="fetch", cols=cols, knobs=knobs)
            meta["spec"] = spec
            self._pre(path, meta)
            check = "fetch_mut" if task.get("mutant") is not None else "fetch"
            res = [(check,) + check_fetch(self.ctx, path, cols, knobs)]
        else:
            raise ValueError(sec)
        self.ctx.stage("idle")
        saved = None
        if any(v == "FAIL" for _, v, _ in res):
            saved = save_fail(self.fails_dir, task, path, meta, res, cfg)
        for q in (path, path + ".json"):
            try:
                os.unlink(q)
            except OSError:
                pass
        out_meta = {}
        if sec == "crafted":
            out_meta = {"pyarrow": meta.get("pyarrow")}
        if any(v == "fallback_unexpected" for _, v, _ in res):
            out_meta["case"] = {"write_opts": meta.get("write_opts")}
        return res, dict(out_meta, saved=saved)


def save_fail(fails_dir, task, path, meta, res, cfg):
    os.makedirs(fails_dir, exist_ok=True)
    if os.path.exists(path + ".json"):
        try:
            meta = dict(json.load(open(path + ".json")), **meta)
        except Exception:
            pass
    tag = f"{task['section']}-{task['id']}"
    dst = os.path.join(fails_dir, tag + ".parquet")
    try:
        shutil.copyfile(path, dst)
    except OSError:
        open(dst, "wb").close()
    side = dict(task=task, meta=meta, results=res, seed=cfg["seed"], size=cfg["size"], versions=versions())
    with open(dst + ".json", "w") as f:
        json.dump(side, f, indent=1, default=str)
    return dst


def _vm_gb(pid="self"):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmSize:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return 0.0


def worker_main(conn, cfg, stage_arr, slot):
    os.environ["NOVA_BF_RUST_FETCH_DENSE"] = "1"
    os.environ["NOVA_BF_RUST_FETCH_COLUMNS"] = "0"
    logging.disable(logging.CRITICAL)
    if _NOVA_SRC not in sys.path:
        sys.path.append(_NOVA_SRC)
    try:
        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
    except Exception:
        pass
    os.makedirs(os.path.join(cfg["work"], f"w{slot}"), exist_ok=True)
    w = Worker(cfg, stage_arr, slot)
    baseline, n_done = None, 0
    conn.send(("ready", slot))
    while True:
        try:
            task = conn.recv()
        except EOFError:
            return
        if task is None:
            return
        recycle = False
        try:
            res, meta = w.run(task)
        except MemoryError:
            res, meta = [("harness", "REF_ABORT", "worker out of memory/address space in harness code")], {}
            recycle = True
        except Exception:
            res, meta = [("harness", "ERROR", traceback.format_exc()[-1500:])], {}
        vm = _vm_gb()
        if baseline is None:
            baseline = vm
        limit = cfg["recycle_vm_gb"] or (baseline + 6.0)
        bye = recycle or vm > limit or (cfg["recycle_tasks"] and n_done >= cfg["recycle_tasks"])
        n_done += 1
        conn.send(("result", task["id"], res, meta, bye))
        if bye:
            return   # parent restarts this slot (pyarrow/jemalloc keep address space after huge allocs)


# --------------------------------------------------------------------------
# parent: task generation, pool, accounting
# --------------------------------------------------------------------------


def gen_tasks(cfg, sections):
    seed, size = cfg["seed"], cfg["size"]
    sz = cfg["sz"]
    tid = 0
    if "matrix" in sections:
        ms = matrix_params(sz["matrix"], seed)
        if cfg.get("matrix_limit") and cfg["matrix_limit"] < len(ms):
            step = len(ms) / cfg["matrix_limit"]
            ms = [ms[int(k * step)] for k in range(cfg["matrix_limit"])]
        for m in ms:
            tid += 1
            yield dict(id=tid, section="matrix", m=m)
    if "crafted" in sections:
        for name in crafted_cases():
            tid += 1
            yield dict(id=tid, section="crafted", name=name)
    if "mutate" in sections:
        bases = mutation_bases(sz["bases"], seed)
        cnames = list(crafted_cases())
        for i in range(sz["mutants"]):
            r = case_rng(seed, "mutate-pick", i)
            # Bases go in blocks of consecutive cases (workers take consecutive
            # tasks, so their per-process base caches hit); partners are drawn
            # near the block's base for the same reason.
            blk = i // max(1, min(_BLOCK, sz["mutants"] // max(1, len(bases))))
            if r.random() < 0.1:
                base = {"crafted": cnames[(blk + r.randrange(4)) % len(cnames)]}
            else:
                base = bases[blk % len(bases)]
            partner = bases[(blk + r.randrange(1, 9)) % len(bases)] if r.random() < 0.5 else None
            tid += 1
            yield dict(id=tid, section="mutate", i=i, base=base, partner=partner)
    if "fetch" in sections:
        for i in range(sz["fetch"]):
            tid += 1
            yield dict(id=tid, section="fetch", i=i)
        nfb = max(1, min(sz["fetch"], sz["fetch_bases"]))
        for j in range(sz["fetch_mutants"]):
            tid += 1
            yield dict(id=tid, section="fetch", i=None, mutant=j,
                       fi=(j // max(1, min(_BLOCK, sz["fetch_mutants"] // nfb))) % nfb)
    _ = size


def _mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return float("inf")


def _rss_gb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return 0.0


class Pool:
    def __init__(self, cfg, n):
        import multiprocessing as mp
        self.mp = mp.get_context("spawn")
        self.cfg, self.n = cfg, n
        self.stage = self.mp.RawArray("i", n)
        self.procs = [None] * n
        self.conns = [None] * n
        self.busy = [None] * n   # (task, deadline)
        self.ready = [False] * n
        self.startup_failures = 0
        for i in range(n):
            self._start(i)

    def _start(self, i):
        parent, child = self.mp.Pipe()
        p = self.mp.Process(target=worker_main, args=(child, self.cfg, self.stage, i), daemon=True)
        p.start()
        child.close()
        self.procs[i], self.conns[i], self.busy[i], self.ready[i] = p, parent, None, False
        self.stage[i] = 0

    def kill(self, i):
        p = self.procs[i]
        try:
            p.kill()
        except Exception:
            pass
        p.join(5)
        try:
            self.conns[i].close()
        except Exception:
            pass

    def shutdown(self):
        for i in range(self.n):
            try:
                self.conns[i].send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(5)
            if p.is_alive():
                p.kill()


def run(cfg, sections, log=print):
    from multiprocessing.connection import wait as mp_wait

    t0 = time.time()
    os.makedirs(cfg["out"], exist_ok=True)
    cfg["work"] = os.path.join(cfg["out"], "work")
    os.makedirs(cfg["work"], exist_ok=True)
    fails_dir = os.path.join(cfg["out"], "fails")
    counts = collections.defaultdict(collections.Counter)      # (section, check) -> verdict counts
    reasons = collections.defaultdict(collections.Counter)     # (section, check, verdict) -> reason counts
    fails, errors = [], []
    crafted_rows = []
    unexpected = []
    total = {s: 0 for s in sections}
    for t in gen_tasks(cfg, sections):
        total[t["section"]] += 1
    tasks = gen_tasks(cfg, sections)
    done = collections.Counter()
    pool = Pool(cfg, cfg["workers"])
    pending_iter_done = False
    timeout = cfg["timeout"]
    last_print = last_rss = time.time()
    aborted = None
    repro_lines = []
    fail_sigs = collections.Counter()

    def record(task, res, meta):
        sec = task["section"]
        done[sec] += 1
        saved = (meta or {}).get("saved")
        keep_saved = False
        for check, verdict, detail in res:
            if verdict == "FAIL":
                sig = f"{sec}.{check}: {_reason_key(detail)[:100]}"
                fail_sigs[sig] += 1
                keep_saved |= fail_sigs[sig] <= cfg["max_saved_per_sig"]
        if saved and not keep_saved and any(v == "FAIL" for _, v, _ in res):
            for q in (saved, saved + ".json"):
                try:
                    os.unlink(q)
                except OSError:
                    pass
            saved = f"(not kept: > {cfg['max_saved_per_sig']} files with this FAIL signature)"
        for check, verdict, detail in res:
            key = (sec, check)
            counts[key][verdict] += 1
            if verdict in ("fallback", "fallback_unexpected", "skip", "guarded"):
                reasons[key + (verdict,)][_reason_key(detail)] += 1
            if verdict == "fallback_unexpected" and len(unexpected) < 200:
                unexpected.append(dict(task=task, check=check, detail=detail, meta=(meta or {}).get("case")))
            if verdict == "FAIL":
                fails.append(dict(section=sec, check=check, task_id=task["id"], detail=detail, file=saved))
                if keep_saved:
                    line = f"{sys.executable} {os.path.abspath(__file__)} --repro {saved}"
                    if line not in repro_lines:
                        repro_lines.append(line)
                    log(f"FAIL [{sec}/{check}] task {task['id']}: {detail}\n  repro: {line}")
            if verdict == "ERROR":
                errors.append(dict(section=sec, task_id=task["id"], detail=detail))
                log(f"HARNESS ERROR [{sec}] task {task['id']}:\n{detail}")
            if verdict == "REF_ABORT":
                log(f"pyarrow aborted [{sec}] task {task['id']}: {detail} (saved {saved})")
        if sec == "crafted":
            crafted_rows.append((task["name"], (meta or {}).get("pyarrow"), {c: (v, d) for c, v, d in res}))

    def assign(i):
        nonlocal pending_iter_done
        if pending_iter_done:
            return False
        try:
            t = next(tasks)
        except StopIteration:
            pending_iter_done = True
            return False
        t["path"] = os.path.join(cfg["work"], f"w{i}", f"case{t['id']}.parquet")
        pool.busy[i] = (t, time.time() + timeout)
        pool.conns[i].send(t)
        return True

    def lost(i, why):
        """Worker i died or hung on its current task."""
        task = pool.busy[i][0] if pool.busy[i] else None
        stage = _STAGE_NAMES.get(pool.stage[i], "?")
        rss = _rss_gb(pool.procs[i].pid) if why != "crash" else 0
        pool.kill(i)
        if task is not None:
            native = stage in NATIVE_STAGES
            verdict = "FAIL" if native else "REF_ABORT"
            detail = {"crash": f"worker died (exit {pool.procs[i].exitcode}) in stage {stage}",
                      "hang": f"no result after {timeout}s in stage {stage}",
                      "mem": f"worker RSS {rss:.1f} GB > {cfg['max_rss_gb']} GB in stage {stage}"}[why]
            saved = None
            if os.path.exists(task["path"]):
                saved = save_fail(fails_dir, task, task["path"], {"lost": why, "stage": stage},
                                  [("worker", verdict, detail)], cfg)
                for q in (task["path"], task["path"] + ".json"):
                    if os.path.exists(q):
                        os.unlink(q)
            if saved and stage in ("store", "fetch"):
                # these stages run pyarrow too (schema read / metadata / decode):
                # does the plain pyarrow read of this file die the same way?
                ok, how = run_confirm(saved, cfg["confirm_timeout"])
                if not ok:
                    verdict = "REF_ABORT"
                    detail += f"; the pyarrow-only read dies too ({how})"
                else:
                    detail += f"; the pyarrow-only read does not ({how})"
            check = {"crate": "crate", "store": "store", "fetch": "fetch_mut" if task.get("mutant") is not None
                     else "fetch"}.get(stage, "ref" if stage in ("ref", "fetch_ref") else "gen")
            record(task, [(check, verdict, detail)], {"saved": saved, "pyarrow": f"lost ({why})"})
        pool._start(i)

    try:
        while True:
            # hand out work
            for i in range(pool.n):
                if pool.ready[i] and pool.busy[i] is None:
                    assign(i)
            if pending_iter_done and all(b is None for b in pool.busy):
                break
            ready = mp_wait([c for c in pool.conns], timeout=0.5)
            for c in ready:
                if c not in pool.conns:
                    continue   # a worker replaced earlier in this pass
                i = pool.conns.index(c)
                try:
                    msg = c.recv()
                except (EOFError, OSError):
                    if pool.busy[i] is not None:
                        lost(i, "crash")
                    else:
                        if not pool.ready[i]:
                            pool.startup_failures += 1
                            if pool.startup_failures > 3 * pool.n:
                                raise RuntimeError("workers keep dying at startup; run one case with "
                                                   "--workers 1 to see the error")
                        pool.kill(i)
                        pool._start(i)
                    continue
                if msg[0] == "ready":
                    pool.ready[i] = True
                elif msg[0] == "result":
                    task = pool.busy[i][0]
                    pool.busy[i] = None
                    record(task, msg[2], msg[3])
                    if msg[4]:   # worker is retiring: restart the slot, don't hand it work
                        pool.procs[i].join(30)
                        pool.kill(i)
                        pool._start(i)
            now = time.time()
            check_rss = now - last_rss > 0.5
            if check_rss:
                last_rss = now
            for i in range(pool.n):
                if pool.busy[i] is not None:
                    if now > pool.busy[i][1]:
                        lost(i, "hang")
                    elif check_rss and _rss_gb(pool.procs[i].pid) > cfg["max_rss_gb"]:
                        lost(i, "mem")
                elif not pool.procs[i].is_alive():
                    pool._start(i)
            if now - last_print > cfg["progress_s"]:
                last_print = now
                log(_progress(done, total, counts, t0))
                avail = _mem_available_gb()
                if avail < cfg["min_avail_gb"]:
                    aborted = f"MemAvailable {avail:.1f} GB < {cfg['min_avail_gb']} GB"
                    log("ABORT: " + aborted)
                    break
    finally:
        pool.shutdown()
    if not cfg.get("keep_work"):
        shutil.rmtree(cfg["work"], ignore_errors=True)

    if crafted_rows:
        with open(os.path.join(cfg["out"], "crafted.tsv"), "w") as f:
            f.write("case\tpyarrow\tcrate\tstore\n")
            for name, pyar, cks in sorted(crafted_rows):
                cell = lambda c: (f"{cks[c][0]}: {cks[c][1]}" if c in cks else "-").replace("\t", " ")  # noqa: E731
                f.write(f"{name}\t{pyar}\t{cell('crate')}\t{cell('store')}\n")
    if unexpected:
        with open(os.path.join(cfg["out"], "unexpected_fallbacks.jsonl"), "w") as f:
            for u in unexpected:
                f.write(json.dumps(u, default=str) + "\n")
    if repro_lines:
        with open(os.path.join(fails_dir, "REPRO.txt"), "a") as f:
            f.write("\n".join(repro_lines) + "\n")

    n_fail = sum(c["FAIL"] for c in counts.values())
    n_err = sum(c["ERROR"] for c in counts.values())
    summary = dict(
        versions=versions(), seed=cfg["seed"], size=cfg["size"], workers=cfg["workers"],
        wall_s=round(time.time() - t0, 1), sections=list(sections), aborted=aborted,
        counts={f"{s}.{c}": dict(v) for (s, c), v in sorted(counts.items())},
        fallback_reasons={f"{k[0]}.{k[1]}.{k[2]}": dict(v.most_common(8)) for k, v in sorted(reasons.items())},
        fails=n_fail, harness_errors=n_err, out=cfg["out"],
        fail_signatures=dict(fail_sigs.most_common(50)),
        fail_files=list(dict.fromkeys(f["file"] for f in fails))[:50],
    )
    log(_table(counts))
    log("SUMMARY " + json.dumps(summary, sort_keys=True, default=str))
    rc = 3 if aborted else 1 if n_fail else 2 if n_err else 0
    return rc, summary


def _reason_key(detail):
    import re
    d = str(detail)
    d = re.sub(r"/[^\s:'\"]+\.parquet", "<file>", d)
    d = re.sub(r"\d+", "N", d)
    return d[:110]


def _progress(done, total, counts, t0):
    el = time.time() - t0
    parts = []
    for s in total:
        tot = total[s]
        if not tot:
            continue
        c = collections.Counter()
        for (sec, _), v in counts.items():
            if sec == s:
                c.update(v)
        parts.append(f"{s} {done[s]}/{tot} (FAIL={c['FAIL']} ident={c['identical']} fb={c['fallback']})")
    n = sum(done.values())
    return f"[{el:7.0f}s] {n / max(el, 1e-9):6.1f} case/s | " + " | ".join(parts)


def _table(counts):
    verdicts = ["identical", "fallback", "guarded", "fallback_unexpected", "FAIL", "REF_ABORT", "skip", "ERROR"]
    rows = [f"{'section.check':<22}" + "".join(f"{v:>21}" for v in verdicts)]
    for (s, c), v in sorted(counts.items()):
        rows.append(f"{s + '.' + c:<22}" + "".join(f"{v.get(x, 0):>21}" for x in verdicts))
    return "\n".join(rows)


# --------------------------------------------------------------------------
# confirm (pyarrow-only read in a fresh process) and repro
# --------------------------------------------------------------------------


def run_confirm(saved, timeout):
    """(pyarrow_survives, description) for a saved case, in a subprocess."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--confirm", saved],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"hangs > {timeout}s"
    tail = (r.stdout.strip().splitlines() or [""])[-1][:160]
    if r.returncode == 0:
        return True, tail
    return False, f"exit {r.returncode}: {(r.stderr.strip().splitlines() or [''])[-1][:160]}"


def confirm_main(path):
    """Only the pyarrow reads nova-bf would do for this case. Exit 0 if the
    process survives (a raised exception is survival), else it dies."""
    logging.disable(logging.CRITICAL)
    if _NOVA_SRC not in sys.path:
        sys.path.append(_NOVA_SRC)
    from nova_bf.io import Store
    side = json.load(open(path + ".json"))
    spec = side["meta"].get("spec") or {"kind": "dense", "col": "e"}
    st = Store(os.path.dirname(path))
    reads = ([("path", lambda: st.read_columns(path, spec["cols"]))] if spec["kind"] == "fetch" else
             [("schema", lambda: pq.read_schema(path)),
              ("path", lambda: st.read_columns(path, [spec["col"]])),
              ("all", lambda: st.read_columns(path, None))])
    out = []
    for name, fn in reads:
        try:
            fn()
            out.append(f"{name}: ok")
        except Exception as e:
            out.append(f"{name}: raises {_short(e, 60)}")
    print("pyarrow-only: " + "; ".join(out), flush=True)
    return 0


def repro(path):
    os.environ["NOVA_BF_RUST_FETCH_DENSE"] = "1"
    logging.disable(logging.CRITICAL)
    if _NOVA_SRC not in sys.path:
        sys.path.append(_NOVA_SRC)
    side = json.load(open(path + ".json"))
    spec = side["meta"].get("spec") or {"kind": "dense", "col": "e", "checks": ["crate", "store"]}
    print("task:", json.dumps(side["task"], default=str)[:400])
    print("meta:", json.dumps({k: v for k, v in side["meta"].items() if k != "spec"}, default=str)[:600])
    print("recorded:", side["results"])
    print("versions now:", versions())
    # run on a private copy in its own directory (Store reads a directory)
    import tempfile
    d = tempfile.mkdtemp(prefix="rf_repro_", dir=os.path.dirname(path))
    p = os.path.join(d, "case.parquet")
    shutil.copyfile(path, p)
    ctx = Ctx()
    if spec["kind"] == "fetch":
        res = [("fetch",) + check_fetch(ctx, p, spec["cols"], spec["knobs"])]
    else:
        res = run_dense_checks(ctx, p, dict(spec, checks=["crate", "store"]))
        # also the other crate option combinations
        for direct in (False, True):
            for client in (0, 1):
                res.append((f"crate[direct={direct},client={client}]",)
                           + check_crate(ctx, p, spec["col"], dict(direct=direct, client=client)))
    try:
        r = _dense_ref(pq.read_table(p, columns=[spec.get("col", "e")])[spec.get("col", "e")]) \
            if spec["kind"] != "fetch" else None
        print("pyarrow:", "n/a" if r is None else f"ok {_describe(r)}")
    except Exception as e:
        print("pyarrow:", _short(e, 300))
    for r in res:
        print("  ", r)
    shutil.rmtree(d, ignore_errors=True)
    return 1 if any(v == "FAIL" for _, v, _ in res) else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_cfg(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(SIZES), default="local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--out", default=None, help="output dir (default: ./rust_fetch_fuzz_<time>)")
    ap.add_argument("--sections", default=",".join(ALL_SECTIONS),
                    help=f"comma list of {','.join(ALL_SECTIONS)}")
    ap.add_argument("--timeout", type=float, default=None, help="per-case timeout, s")
    ap.add_argument("--mutants", type=int, default=None, help="override the size's mutant count")
    ap.add_argument("--fetch", type=int, default=None, help="override the size's fetch file count")
    ap.add_argument("--fetch-mutants", type=int, default=None)
    ap.add_argument("--bases", type=int, default=None, help="mutation base-file pool size")
    ap.add_argument("--matrix-limit", type=int, default=None, help="evenly spaced subset of the matrix cases")
    ap.add_argument("--max-rss-gb", type=float, default=6.0, help="kill a worker above this RSS (counted)")
    ap.add_argument("--min-avail-gb", type=float, default=4.0, help="abort if MemAvailable drops below")
    ap.add_argument("--recycle-vm-gb", type=float, default=None,
                    help="restart a worker whose virtual size passes this (default: its size after the "
                         "first case + 6 GB; keeps clear of ulimit -v)")
    ap.add_argument("--recycle-tasks", type=int, default=20000, help="restart workers after this many cases")
    ap.add_argument("--confirm-timeout", type=float, default=120.0)
    ap.add_argument("--max-saved-per-sig", type=int, default=25,
                    help="keep at most this many saved files per FAIL signature (all are counted)")
    ap.add_argument("--progress-s", type=float, default=15.0)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--repro", default=None, help="re-run the checks on a saved FAIL file")
    ap.add_argument("--confirm", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.repro or a.confirm:
        return a, None
    sz = dict(SIZES[a.size])
    for k in ("mutants", "fetch", "bases"):
        if getattr(a, k) is not None:
            sz[k] = getattr(a, k)
    if a.fetch_mutants is not None:
        sz["fetch_mutants"] = a.fetch_mutants
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    bad = set(secs) - set(ALL_SECTIONS)
    if bad:
        ap.error(f"unknown sections {bad}")
    out = os.path.abspath(a.out or f"rust_fetch_fuzz_{time.strftime('%Y%m%d_%H%M%S')}")
    cfg = dict(size=a.size, sz=sz, seed=a.seed, workers=max(1, a.workers), out=out,
               timeout=a.timeout or sz["timeout"], max_rss_gb=a.max_rss_gb, min_avail_gb=a.min_avail_gb,
               progress_s=a.progress_s, keep_work=a.keep_work, recycle_vm_gb=a.recycle_vm_gb,
               recycle_tasks=a.recycle_tasks,
               confirm_timeout=a.confirm_timeout, matrix_limit=a.matrix_limit, max_saved_per_sig=a.max_saved_per_sig)
    return a, (cfg, tuple(s for s in ALL_SECTIONS if s in secs))


def main(argv=None, log=None):
    a, rest = build_cfg(argv)
    if a.confirm:
        return confirm_main(os.path.abspath(a.confirm))
    if a.repro:
        return repro(os.path.abspath(a.repro))
    cfg, secs = rest
    log = log or (lambda s: print(s, flush=True))
    v = versions()
    log(f"rust_fetch differential: size={cfg['size']} seed={cfg['seed']} workers={cfg['workers']} "
        f"sections={','.join(secs)} out={cfg['out']}\n  pyarrow {v['pyarrow']}, rust_fetch {v['rust_fetch']} "
        f"({v['rust_fetch_file']}, so md5 {v['rust_fetch_so_md5']})")
    rc, _ = run(cfg, secs, log)
    return rc


if __name__ == "__main__":
    sys.exit(main())
