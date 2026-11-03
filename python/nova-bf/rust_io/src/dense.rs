//! Fast path for decoding a Parquet dense-vector column directly into a
//! row-major `[n_rows, dim]` buffer without Arrow intermediates or the GIL.
//!
//! Guarantees bitwise equivalence with PyArrow's `dense_to_2d` output.
//! Unsupported layouts fall back to PyArrow.
//!
//! Supports fixed-width, non-null float32/float16 lists; Parquet v1/v2
//! data pages; plain/dictionary encoding; and uncompressed, Snappy,
//! Gzip, Zstd, and LZ4_RAW compression.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use parquet::basic::{
    Compression, ConvertedType, LogicalType, Repetition, Type as PhysicalType,
};
use parquet::file::metadata::ParquetMetaDataReader;
use tokio::task::JoinSet;

use crate::core::{Dest, FetchError, Fetcher, ReadOpts, Span};

#[derive(Debug, thiserror::Error)]
pub enum DenseError {
    /// The file is fine, but not something this fast path decodes exactly:
    /// fall back to pyarrow.
    #[error("unsupported layout: {0}")]
    Unsupported(String),
    /// The bytes contradict themselves (bad thrift, sizes that don't add up).
    #[error("corrupt parquet: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Fetch(#[from] FetchError),
}

pub type DResult<T> = std::result::Result<T, DenseError>;

fn unsupported<T>(msg: impl Into<String>) -> DResult<T> {
    Err(DenseError::Unsupported(msg.into()))
}

fn corrupt<T>(msg: impl Into<String>) -> DResult<T> {
    Err(DenseError::Corrupt(msg.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Uncompressed,
    Snappy,
    Gzip,
    Zstd,
    Lz4Raw,
}

impl Codec {
    fn from_parquet(c: Compression) -> DResult<Self> {
        Ok(match c {
            Compression::UNCOMPRESSED => Codec::Uncompressed,
            Compression::SNAPPY => Codec::Snappy,
            Compression::GZIP(_) => Codec::Gzip,
            Compression::ZSTD(_) => Codec::Zstd,
            Compression::LZ4_RAW => Codec::Lz4Raw,
            other => return unsupported(format!("codec {other:?}")),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Codec::Uncompressed => "uncompressed",
            Codec::Snappy => "snappy",
            Codec::Gzip => "gzip",
            Codec::Zstd => "zstd",
            Codec::Lz4Raw => "lz4_raw",
        }
    }

    /// Decompresses `src` into `out`, validating the expected output size
    /// and bounding allocations to prevent malformed-header overflows.
    fn decompress_vec(&self, src: &[u8], expected: usize, out: &mut Vec<u8>) -> DResult<()> {
        use std::io::Read;
        match self {
            Codec::Uncompressed | Codec::Snappy | Codec::Lz4Raw => {
                match self {
                    Codec::Snappy => {
                        let n = snap::raw::decompress_len(src)
                            .map_err(|e| DenseError::Corrupt(format!("snappy: {e}")))?;
                        // A snappy element expands at most ~21x; the preamble
                        // must agree with the header and with that bound.
                        if n != expected || n > src.len().saturating_mul(22).saturating_add(64) {
                            return corrupt(format!("snappy page of {n} bytes, header says {expected}"));
                        }
                    }
                    Codec::Lz4Raw => {
                        if expected > src.len().saturating_mul(255).saturating_add(64) {
                            return corrupt("lz4_raw page claims more than the format can expand to");
                        }
                    }
                    _ => {
                        // Uncompressed: the bytes ARE the page -- check before
                        // allocating, never trust the claimed size.
                        if src.len() != expected {
                            return corrupt(format!(
                                "uncompressed page holds {} bytes, header says {expected}",
                                src.len()
                            ));
                        }
                    }
                }
                sized(out, expected)?;
                self.decompress(src, &mut out[..])
            }
            Codec::Gzip | Codec::Zstd => {
                out.clear();
                let limit = expected as u64 + 1;
                if *self == Codec::Gzip {
                    // Arrow writes gzip framing but accepts zlib framing too.
                    let is_gzip = src.starts_with(&[0x1f, 0x8b]);
                    let mut zl = flate2::bufread::ZlibDecoder::new(src);
                    let mut gz = flate2::bufread::MultiGzDecoder::new(src);
                    let d: &mut dyn Read = if is_gzip { &mut gz } else { &mut zl };
                    d.take(limit)
                        .read_to_end(out)
                        .map_err(|e| DenseError::Corrupt(format!("gzip: {e}")))?;
                    let rest = if is_gzip { gz.into_inner() } else { zl.into_inner() };
                    if !rest.is_empty() {
                        return corrupt(format!("{} bytes after the compressed stream", rest.len()));
                    }
                } else {
                    let mut d = zstd::stream::read::Decoder::with_buffer(src)
                        .map_err(|e| DenseError::Corrupt(format!("zstd: {e}")))?;
                    (&mut d)
                        .take(limit)
                        .read_to_end(out)
                        .map_err(|e| DenseError::Corrupt(format!("zstd: {e}")))?;
                }
                if out.len() != expected {
                    return corrupt(format!(
                        "{} page decompressed to {} bytes, header says {expected}",
                        self.name(),
                        out.len()
                    ));
                }
                Ok(())
            }
        }
    }

    /// Decompress `src` into exactly `out.len()` bytes.
    fn decompress(&self, src: &[u8], out: &mut [u8]) -> DResult<()> {
        let n = match self {
            Codec::Uncompressed => {
                if src.len() != out.len() {
                    return corrupt(format!(
                        "uncompressed page holds {} bytes, header says {}",
                        src.len(),
                        out.len()
                    ));
                }
                out.copy_from_slice(src);
                out.len()
            }
            Codec::Snappy => snap::raw::Decoder::new()
                .decompress(src, out)
                .map_err(|e| DenseError::Corrupt(format!("snappy: {e}")))?,
            Codec::Gzip => {
                use std::io::Read;
                // Arrow writes gzip framing but accepts zlib framing too.
                let is_gzip = src.starts_with(&[0x1f, 0x8b]);
                let mut zl = flate2::bufread::ZlibDecoder::new(src);
                let mut gz = flate2::bufread::MultiGzDecoder::new(src);
                let d: &mut dyn Read = if is_gzip { &mut gz } else { &mut zl };
                let mut filled = 0;
                while filled < out.len() {
                    let k = d
                        .read(&mut out[filled..])
                        .map_err(|e| DenseError::Corrupt(format!("gzip: {e}")))?;
                    if k == 0 {
                        break;
                    }
                    filled += k;
                }
                // Anything left over means the header's size was wrong.
                let mut extra = [0u8; 1];
                if filled == out.len()
                    && d.read(&mut extra).map_err(|e| DenseError::Corrupt(format!("gzip: {e}")))? != 0
                {
                    return corrupt("gzip page decompresses past its declared size");
                }
                // ... and so do input bytes after the stream (pyarrow refuses them).
                let rest = if is_gzip { gz.into_inner() } else { zl.into_inner() };
                if !rest.is_empty() {
                    return corrupt(format!("{} bytes after the compressed stream", rest.len()));
                }
                filled
            }
            Codec::Zstd => zstd::bulk::Decompressor::new()
                .and_then(|mut d| d.decompress_to_buffer(src, out))
                .map_err(|e| DenseError::Corrupt(format!("zstd: {e}")))?,
            Codec::Lz4Raw => {
                lz4_block_rules(src, out.len())?;
                lz4_flex::block::decompress_into(src, out)
                    .map_err(|e| DenseError::Corrupt(format!("lz4_raw: {e}")))?
            }
        };
        if n != out.len() {
            return corrupt(format!(
                "{} page decompressed to {n} bytes, header says {}",
                self.name(),
                out.len()
            ));
        }
        Ok(())
    }
}

/// Validates LZ4 block termination rules required by PyArrow without
/// decompressing, ensuring compatible decoding behavior.
fn lz4_block_rules(src: &[u8], out_len: usize) -> DResult<()> {
    let bad = |m: &str| DenseError::Corrupt(format!("lz4_raw: {m}"));
    let mut i = 0usize;
    let mut o = 0usize; // output position
    let read_len = |i: &mut usize, mut n: usize| -> DResult<usize> {
        if n == 15 {
            loop {
                let b = *src.get(*i).ok_or_else(|| bad("truncated length"))?;
                *i += 1;
                n = n.checked_add(b as usize).ok_or_else(|| bad("length overflow"))?;
                if b != 255 {
                    break;
                }
            }
        }
        Ok(n)
    };
    while i < src.len() {
        let token = src[i];
        i += 1;
        let lit = read_len(&mut i, (token >> 4) as usize)?;
        i = i.checked_add(lit).filter(|&e| e <= src.len()).ok_or_else(|| bad("literals past input"))?;
        o = o.checked_add(lit).ok_or_else(|| bad("output overflow"))?;
        if i == src.len() {
            break; // last sequence: literals only
        }
        if i + 2 > src.len() {
            return Err(bad("truncated match offset"));
        }
        i += 2;
        let mlen = read_len(&mut i, (token & 15) as usize)?.checked_add(4).ok_or_else(|| bad("length overflow"))?;
        if o + 12 > out_len || o + mlen + 5 > out_len {
            return Err(bad("match too close to the end of the block"));
        }
        o += mlen;
    }
    Ok(())
}

/// One row group's column chunk.
#[derive(Debug, Clone)]
pub struct ChunkPlan {
    /// File byte range of the whole chunk (dictionary page included).
    pub offset: u64,
    pub len: u64,
    /// First row of this row group in the file, and its row count.
    pub row_start: u64,
    pub n_rows: u64,
    /// Number of LEVELS (== values, once nulls are ruled out) in the chunk.
    pub num_values: u64,
    pub codec: Codec,
}

/// Everything needed to decode one file's dense column. Built from the footer
/// alone (`plan_dense`), so the caller can size its destination first.
#[derive(Debug, Clone)]
pub struct DensePlan {
    pub url: String,
    pub column: String,
    pub n_rows: u64,
    pub dim: u64,
    /// Bytes per element: 4 (float32) or 2 (float16).
    pub elem: usize,
    pub max_def: i16,
    pub codec: Codec,
    pub chunks: Vec<ChunkPlan>,
    /// ETag of the object version the footer was read from; every chunk read
    /// must match it, so a file rewritten mid-read is never stitched together.
    pub etag: Option<String>,
}

impl DensePlan {
    pub fn dtype(&self) -> &'static str {
        if self.elem == 2 { "float16" } else { "float32" }
    }

    /// Size of the decoded `[n_rows, dim]` output in bytes.
    pub fn nbytes(&self) -> u64 {
        self.n_rows * self.dim * self.elem as u64
    }

    /// Bytes fetched from storage (the column's chunks only).
    pub fn column_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.len).sum()
    }
}

// ---------------------------------------------------------------------------
// Footer -> plan
// ---------------------------------------------------------------------------

/// How much of the file tail to fetch speculatively; one GET covers the
/// footer of any file we care about (Pes2o: ~10 KB).
const TAIL_GUESS: u64 = 256 << 10;

async fn fetch_vec(f: &Fetcher, url: &str, offset: u64, len: u64, opts: &ReadOpts) -> DResult<Vec<u8>> {
    let mut v = vec![0u8; len as usize];
    if len > 0 {
        let dest = Dest { addr: v.as_mut_ptr() as usize, len };
        f.read_spans_opts(url, &[Span { src: offset, len, dst: 0 }], dest, opts.clone()).await?;
    }
    Ok(v)
}

/// Read the footer of `url` and plan decoding of top-level column `column`.
pub async fn plan_dense(f: &Fetcher, url: &str, column: &str, size: Option<u64>, stop: Arc<AtomicBool>) -> DResult<DensePlan> {
    // Always HEAD: the ETag pins every later read to this object version
    // (and a caller-supplied size must agree with it).
    // Raced with `stop` (dropping a HEAD future is always safe).
    let (head_size, etag) = tokio::select! {
        r = f.head(url) => r?,
        _ = crate::core::stopped(&stop) => return Err(DenseError::Fetch(FetchError::Cancelled)),
    };
    if let Some(s) = size {
        if s != head_size {
            return Err(DenseError::Fetch(FetchError::Changed(format!(
                "{url}: expected {s} bytes, object has {head_size}"
            ))));
        }
    }
    let size = head_size;
    let opts = ReadOpts { stop, if_match: etag.clone() };
    if size < 12 {
        return corrupt(format!("{url}: {size} bytes is too small for a parquet file"));
    }
    let tail_len = TAIL_GUESS.min(size);
    let tail = fetch_vec(f, url, size - tail_len, tail_len, &opts).await?;
    let magic = &tail[tail.len() - 4..];
    if magic == b"PARE" {
        return unsupported("encrypted footer");
    }
    if magic != b"PAR1" {
        return corrupt(format!("{url}: missing PAR1 footer magic"));
    }
    let flen = u32::from_le_bytes(tail[tail.len() - 8..tail.len() - 4].try_into().unwrap()) as u64;
    if flen + 8 > size {
        return corrupt(format!("{url}: footer length {flen} exceeds file size {size}"));
    }
    let footer = if flen + 8 <= tail_len {
        tail[(tail_len - 8 - flen) as usize..(tail_len - 8) as usize].to_vec()
    } else {
        fetch_vec(f, url, size - 8 - flen, flen, &opts).await?
    };
    let mut plan = plan_from_footer(&footer, column, size)?;
    plan.url = url.to_string();
    plan.etag = etag;
    Ok(plan)
}

/// Extract column chunk offsets and sizes from a raw Parquet footer without I/O.
/// Returns (data_page_offset, dictionary_page_offset, total_compressed_size)
/// for each requested leaf in each row group, ordered by row group.
/// Validates the footer before pyarrow access, avoiding crashes on malformed metadata.
pub fn chunk_spans(footer: &[u8], leaves: &[usize]) -> DResult<Vec<(i64, Option<i64>, i64)>> {
    check_footer_thrift(footer)?;
    let md = ParquetMetaDataReader::decode_metadata(footer)
        .map_err(|e| DenseError::Corrupt(format!("footer: {e}")))?;
    let schema = md.file_metadata().schema_descr();
    let ncols = schema.num_columns();
    if let Some(&bad) = leaves.iter().find(|&&l| l >= ncols) {
        return unsupported(format!("leaf {bad} out of range ({ncols} columns)"));
    }
    let mut out = Vec::with_capacity(md.num_row_groups() * leaves.len());
    for rg in md.row_groups() {
        if rg.num_columns() != ncols {
            return unsupported("row group column count differs from the schema");
        }
        // A row group claiming more rows than a chunk holds values makes
        // pyarrow size its reads from the bogus count (the fetch fuzz saw a
        // 2^32-row claim balloon a BufferReader decode to 7.6 GB); the dense
        // planner refuses it for the same reason.
        if rg.num_rows() < 0 {
            return corrupt("negative row count in footer");
        }
        // pyarrow validates every chunk's size statistics when it builds
        // ColumnChunkMetaData; mirror plan_from_footer's checks.
        for (j, cc) in rg.columns().iter().enumerate() {
            let d = schema.column(j);
            if cc.num_values() < rg.num_rows() {
                return unsupported(format!(
                    "column chunk with {} values in a row group of {} rows",
                    cc.num_values(),
                    rg.num_rows()
                ));
            }
            if cc.unencoded_byte_array_data_bytes().is_some() && d.physical_type() != PhysicalType::BYTE_ARRAY {
                return unsupported("unencoded_byte_array_data_bytes on a non-BYTE_ARRAY column");
            }
            for (h, max, what) in [
                (cc.definition_level_histogram(), d.max_def_level(), "definition"),
                (cc.repetition_level_histogram(), d.max_rep_level(), "repetition"),
            ] {
                if let Some(h) = h {
                    if h.len() != 0 && h.len() != max as usize + 1 {
                        return unsupported(format!("{what} level histogram of {} for max level {max}", h.len()));
                    }
                }
            }
        }
        for &l in leaves {
            let cc = rg.column(l);
            out.push((cc.data_page_offset(), cc.dictionary_page_offset(), cc.compressed_size()));
        }
    }
    Ok(out)
}

pub fn plan_from_footer(footer: &[u8], column: &str, file_size: u64) -> DResult<DensePlan> {
    check_footer_thrift(footer)?;
    let md = ParquetMetaDataReader::decode_metadata(footer)
        .map_err(|e| DenseError::Corrupt(format!("footer: {e}")))?;
    let schema = md.file_metadata().schema_descr();

    // Accept only Parquet schemas PyArrow interprets as flat primitive lists;
    // reject nested, ambiguous, or incompatible layouts.
    let root = schema.root_schema();
    // pyarrow's dataset read (what nova-bf's pyarrow path uses) refuses a
    // file with ANY duplicated top-level name, not just the requested one.
    {
        let mut seen = std::collections::HashSet::new();
        if let Some(t) = root.get_fields().iter().find(|t| !seen.insert(t.name())) {
            return unsupported(format!("duplicate top-level field {:?}", t.name()));
        }
    }
    let top = root
        .get_fields()
        .iter()
        .find(|t| t.name() == column)
        .ok_or_else(|| DenseError::Unsupported(format!("no top-level column {column:?}")))?;
    let not_list = || unsupported::<()>(format!("column {column:?} is not a flat list of floats"));
    let rep_of = |t: &parquet::schema::types::Type| {
        let i = t.get_basic_info();
        if i.has_repetition() { Some(i.repetition()) } else { None }
    };
    let annotated = |t: &parquet::schema::types::Type| {
        let i = t.get_basic_info();
        i.logical_type_ref().is_some() || i.converted_type() != ConvertedType::NONE
    };
    if top.is_primitive() {
        if rep_of(top) != Some(Repetition::REPEATED) {
            not_list()?;
        }
    } else {
        let info = top.get_basic_info();
        // Ensure the column is a list with exactly one child and no conflicting type metadata.
        let logical_ok = matches!(info.logical_type_ref(), None | Some(LogicalType::List));
        let converted_ok = matches!(info.converted_type(), ConvertedType::NONE | ConvertedType::LIST);
        let is_list_annot = logical_ok
            && converted_ok
            && (matches!(info.logical_type_ref(), Some(LogicalType::List))
                || info.converted_type() == ConvertedType::LIST);
        if !is_list_annot || rep_of(top) == Some(Repetition::REPEATED) || top.get_fields().len() != 1 {
            not_list()?;
        }
        let r = &top.get_fields()[0];
        if rep_of(r) != Some(Repetition::REPEATED) {
            not_list()?;
        }
        if r.is_group() {
            if annotated(r)
                || r.name() == "array"
                || r.name() == format!("{column}_tuple")
                || r.get_fields().len() != 1
            {
                not_list()?;
            }
            let e = &r.get_fields()[0];
            if !e.is_primitive() || rep_of(e) == Some(Repetition::REPEATED) {
                not_list()?;
            }
        }
    }

    let leaves: Vec<usize> = (0..schema.num_columns())
        .filter(|&i| schema.column(i).path().parts().first().map(|s| s.as_str()) == Some(column))
        .collect();
    if leaves.len() != 1 {
        return unsupported(format!("column {column:?} has {} leaves, expected 1", leaves.len()));
    }
    let ci = leaves[0];
    let cd = schema.column(ci);
    if cd.max_rep_level() != 1 {
        return unsupported(format!("max repetition level {} (nested lists?)", cd.max_rep_level()));
    }
    let elem = match cd.physical_type() {
        PhysicalType::FLOAT if cd.logical_type_ref().is_none() => 4usize,
        PhysicalType::FIXED_LEN_BYTE_ARRAY
            if cd.type_length() == 2 && matches!(cd.logical_type_ref(), Some(LogicalType::Float16)) =>
        {
            2usize
        }
        other => {
            return unsupported(format!(
                "element type {other:?} (logical {:?}); only float32 / float16",
                cd.logical_type_ref()
            ))
        }
    };

    let ovf = || DenseError::Corrupt("row/value counts in the footer overflow".into());
    let mut chunks = Vec::with_capacity(md.num_row_groups());
    let mut codec: Option<Codec> = None;
    let mut row_start = 0u64;
    let mut total_values = 0u64;
    for rg in md.row_groups() {
        // pyarrow validates every column's size-statistics level histograms
        // (length must be max level + 1) and refuses the file otherwise;
        // parquet-rs does not -- refuse too.
        for (j, other) in rg.columns().iter().enumerate() {
            let d = schema.column(j);
            // Every column chunk holds at least one level per row. When any
            // chunk (even an unread one) claims fewer, pyarrow silently reads
            // only that many rows of the whole row group -- refuse.
            if other.num_values() < rg.num_rows() {
                return unsupported(format!(
                    "column chunk with {} values in a row group of {} rows",
                    other.num_values(),
                    rg.num_rows()
                ));
            }
            // pyarrow's SizeStatistics::Validate also refuses this field on
            // any non-BYTE_ARRAY column.
            if other.unencoded_byte_array_data_bytes().is_some()
                && d.physical_type() != PhysicalType::BYTE_ARRAY
            {
                return unsupported("unencoded_byte_array_data_bytes on a non-BYTE_ARRAY column");
            }
            for (h, max, what) in [
                (other.definition_level_histogram(), d.max_def_level(), "definition"),
                (other.repetition_level_histogram(), d.max_rep_level(), "repetition"),
            ] {
                if let Some(h) = h {
                    if h.len() != 0 && h.len() != max as usize + 1 {
                        return unsupported(format!("{what} level histogram of {} for max level {max}", h.len()));
                    }
                }
            }
        }
        let cc = rg.column(ci);
        if rg.num_rows() < 0 || cc.num_values() < 0 || cc.compressed_size() < 0 {
            return corrupt("negative count in footer");
        }
        // pyarrow builds the codec and the chunk range for EVERY row group,
        // empty ones included, so validate both before skipping anything.
        let c = Codec::from_parquet(cc.compression())?;
        {
            // The chunk starts at its dictionary page if any (as in the
            // range check below), else at its first data page.
            let start = match cc.dictionary_page_offset() {
                Some(d) if d > 0 && d < cc.data_page_offset() => d,
                _ => cc.data_page_offset(),
            };
            if start < 0
                || cc.dictionary_page_offset().is_some_and(|d| d < 0)
                || (start as u64).saturating_add(cc.compressed_size() as u64) > file_size
            {
                return corrupt("column chunk range outside the file");
            }
        }
        if rg.num_rows() == 0 {
            // Nothing to decode (pyarrow writes an empty table as one
            // zero-row group whose chunk offset is 0). Values without rows
            // would be a contradiction.
            if cc.num_values() != 0 {
                return corrupt("row group with 0 rows holds values");
            }
            continue;
        }
        codec.get_or_insert(c);
        let start = match cc.dictionary_page_offset() {
            // Some old writers record 0 for "no dictionary"; a real
            // dictionary page always precedes the first data page.
            Some(d) if d > 0 && d < cc.data_page_offset() => d,
            _ => cc.data_page_offset(),
        };
        if start < 4 {
            return corrupt(format!("column chunk starts at offset {start}"));
        }
        let (offset, len) = (start as u64, cc.compressed_size() as u64);
        if offset.checked_add(len).map_or(true, |end| end > file_size) {
            return corrupt(format!("column chunk [{offset}, +{len}) runs past the file ({file_size} bytes)"));
        }
        let n_rows = rg.num_rows() as u64;
        chunks.push(ChunkPlan { offset, len, row_start, n_rows, num_values: cc.num_values() as u64, codec: c });
        row_start = row_start.checked_add(n_rows).ok_or_else(ovf)?;
        total_values = total_values.checked_add(cc.num_values() as u64).ok_or_else(ovf)?;
    }
    let n_rows = row_start;
    let dim = if n_rows == 0 {
        0
    } else {
        if total_values % n_rows != 0 {
            return unsupported(format!("{total_values} values over {n_rows} rows: rows are ragged"));
        }
        total_values / n_rows
    };
    if n_rows > 0 && dim == 0 {
        return unsupported("zero-width vectors");
    }
    // Every byte offset used later (nbytes, row_start * row_bytes) is at most
    // this product, so checking it once here makes all of them overflow-free
    // -- and keeps every slice length within isize::MAX.
    let nbytes = n_rows
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(elem as u64))
        .filter(|&b| b <= isize::MAX as u64)
        .ok_or_else(ovf)?;
    let _ = nbytes;
    for c in &chunks {
        if Some(c.num_values) != c.n_rows.checked_mul(dim) {
            return unsupported(format!(
                "row group with {} rows holds {} values, expected {} rows x dim {dim}",
                c.n_rows, c.num_values, c.n_rows
            ));
        }
    }
    Ok(DensePlan {
        url: String::new(),
        column: column.to_string(),
        n_rows,
        dim,
        elem,
        max_def: cd.max_def_level(),
        codec: codec.unwrap_or(Codec::Uncompressed),
        chunks,
        etag: None,
    })
}

// Minimal Thrift compact decoder for Parquet page headers.
// Reject duplicate field IDs to ensure consistent behavior with PyArrow.
#[derive(Default)]
struct FieldSet {
    ids: Vec<i16>,
}

impl FieldSet {
    fn insert(&mut self, id: i16) -> DResult<()> {
        if self.ids.contains(&id) {
            return unsupported(format!("thrift field {id} repeated in one struct"));
        }
        self.ids.push(id);
        Ok(())
    }
}

/// Size limits on thrift containers / binaries (pyarrow refuses larger ones
/// in page headers; nothing legitimate in a dense column comes close).
const MAX_THRIFT_BINARY: u64 = 16 << 20;
const MAX_THRIFT_CONTAINER: u64 = 1_000_000; // pyarrow's thrift container limit
/// pyarrow's kDefaultMaxPageHeaderSize.
const MAX_PAGE_HEADER: usize = 16 << 20;
/// pyarrow's thrift string limit for the footer.
const MAX_FOOTER_BINARY: u64 = 100_000_000;

/// Expected Thrift wire types for Parquet footer fields, indexed by
/// nested field-ID paths. List elements share their parent path.
fn expected_type(path: &[i16], id: i16) -> Option<u8> {
    let (i32_, i64_, bin, list, st) = (5u8, 6u8, 8u8, 9u8, 12u8);
    match (path, id) {
        ([], 1) => Some(i32_),
        ([], 2) | ([], 4) | ([], 5) | ([], 7) => Some(list),
        ([], 3) => Some(i64_),
        ([], 6) => Some(bin),
        ([2], 1 | 2 | 3 | 5 | 6 | 7 | 8 | 9) => Some(i32_),
        ([2], 4) => Some(bin),
        ([2], 10) => Some(st),
        ([2, 10], _) => Some(st), // LogicalType union members are all structs
        ([4], 1 | 4) => Some(list),
        ([4], 2 | 3 | 5 | 6) => Some(i64_),
        ([4], 7) => Some(4), // i16
        ([4, 1], 1) => Some(bin),
        ([4, 1], 2 | 4 | 6) => Some(i64_),
        ([4, 1], 3) => Some(st),
        ([4, 1], 5 | 7) => Some(i32_),
        ([4, 1, 3], 1 | 4 | 15) => Some(i32_),
        ([4, 1, 3], 2 | 3 | 8 | 13) => Some(list),
        ([4, 1, 3], 5 | 6 | 7 | 9 | 10 | 11 | 14) => Some(i64_),
        ([4, 1, 3], 12 | 16) => Some(st),
        _ => None,
    }
}

/// Validates Parquet footer metadata, rejecting duplicate field IDs
/// and encrypted columns to ensure PyArrow-compatible behavior.
fn check_footer_thrift(footer: &[u8]) -> DResult<()> {
    /// num_children of each SchemaElement, in order (None = leaf/absent).
    struct W {
        children: Vec<Option<i64>>,
        /// SchemaElement.type (physical type) per schema element, in order.
        types: Vec<Option<i64>>,
        /// ColumnMetaData.type of every column chunk, in footer order.
        chunk_types: Vec<Option<i64>>,
        /// Byte lengths of each chunk's Statistics min/max values (fields
        /// max=1, min=2, max_value=5, min_value=6), aligned with chunk_types.
        stat_lens: Vec<Vec<u64>>,
    }
    /// i16/i32 varints: pyarrow's thrift truncates the raw varint to 32 bits
    /// BEFORE zigzag, parquet-rs zigzags 64 bits then truncates -- they
    /// disagree whenever the raw varint is >= 2^32. Refuse those.
    fn small_int(t: &mut Thrift) -> DResult<i64> {
        let raw = t.uvarint()?;
        if raw >> 32 != 0 {
            return unsupported("footer i16/i32 varint wider than 32 bits");
        }
        let raw = raw as u32;
        Ok(((raw >> 1) as i32 ^ -((raw & 1) as i32)) as i64)
    }
    fn walk_struct(t: &mut Thrift, w: &mut W, depth: u32, path: &mut Vec<i16>) -> DResult<()> {
        if depth > 64 {
            return corrupt("footer nested too deeply");
        }
        if path.as_slice() == [2] {
            if w.children.len() >= 100_000 {
                return unsupported("schema with more than 100000 elements");
            }
            w.children.push(None);
            w.types.push(None);
        }
        if path.as_slice() == [4, 1, 3] {
            w.chunk_types.push(None);
            w.stat_lens.push(Vec::new());
        }
        let mut last = 0i16;
        let mut seen = FieldSet::default();
        while let Some((id, ty)) = t.field(&mut last)? {
            seen.insert(id)?;
            // A known field carrying the wrong wire type is SKIPPED by
            // pyarrow's reader (treated as absent) but may be coerced by
            // parquet-rs: the two would see different metadata.
            if let Some(want) = expected_type(path, id) {
                let ok = if want == 1 { ty == 1 || ty == 2 } else { ty == want };
                if !ok {
                    return unsupported(format!("footer field {path:?}/{id} has thrift type {ty}, expected {want}"));
                }
            }
            if path.as_slice() == [2] && id == 5 && ty == 5 {
                let n = small_int(t)?;
                *w.children.last_mut().unwrap() = Some(n);
                continue;
            }
            if path.as_slice() == [2] && id == 1 && ty == 5 {
                let v = small_int(t)?;
                *w.types.last_mut().unwrap() = Some(v);
                continue;
            }
            if path.as_slice() == [4, 1, 3] && id == 1 && ty == 5 {
                let v = small_int(t)?;
                *w.chunk_types.last_mut().unwrap() = Some(v);
                continue;
            }
            // ColumnMetaData.statistics(12) min/max values: pyarrow decodes
            // them on read and refuses a wrong-sized fixed-width value.
            if path.as_slice() == [4, 1, 3, 12] && matches!(id, 1 | 2 | 5 | 6) && ty == 8 {
                let n = t.uvarint()?;
                if n > MAX_FOOTER_BINARY {
                    return unsupported(format!("footer binary of {n} bytes"));
                }
                t.skip_bytes(n)?;
                if let Some(v) = w.stat_lens.last_mut() {
                    v.push(n);
                }
                continue;
            }
            path.push(id);
            // FileMetaData.row_groups(4) > RowGroup.columns(1) > ColumnChunk.{8,9}
            if path.len() == 3 && path[0] == 4 && path[1] == 1 && (id == 8 || id == 9) {
                return unsupported("encrypted column chunk");
            }
            walk_value(t, w, ty, depth, path)?;
            path.pop();
        }
        Ok(())
    }
    fn walk_value(t: &mut Thrift, w: &mut W, ty: u8, depth: u32, path: &mut Vec<i16>) -> DResult<()> {
        match ty {
            12 => walk_struct(t, w, depth + 1, path),
            4 | 5 => small_int(t).map(|_| ()),
            9 | 10 => {
                let h = t.byte()?;
                let n = if h >> 4 == 15 { t.uvarint()? } else { (h >> 4) as u64 };
                if n > MAX_THRIFT_CONTAINER {
                    return unsupported(format!("footer list of {n} elements"));
                }
                let et = h & 0x0f;
                for _ in 0..n {
                    if et == 1 || et == 2 {
                        t.skip_bytes(1)?;
                    } else {
                        walk_value(t, w, et, depth + 1, path)?;
                    }
                }
                Ok(())
            }
            8 => {
                let n = t.uvarint()?;
                if n > MAX_FOOTER_BINARY {
                    return unsupported(format!("footer binary of {n} bytes"));
                }
                t.skip_bytes(n)
            }
            11 => {
                let n = t.uvarint()?;
                if n > MAX_THRIFT_CONTAINER {
                    return unsupported(format!("footer map of {n} entries"));
                }
                if n == 0 {
                    return Ok(());
                }
                let kv = t.byte()?;
                for _ in 0..n {
                    for et in [kv >> 4, kv & 0x0f] {
                        if et == 1 || et == 2 {
                            t.skip_bytes(1)?;
                        } else {
                            walk_value(t, w, et, depth + 1, path)?;
                        }
                    }
                }
                Ok(())
            }
            _ => t.skip(ty, depth.min(31)),
        }
    }
    let mut t = Thrift { b: footer, pos: 0 };
    let mut w = W { children: Vec::new(), types: Vec::new(), chunk_types: Vec::new(), stat_lens: Vec::new() };
    walk_struct(&mut t, &mut w, 0, &mut Vec::new())?;
    // pyarrow decodes column statistics when it reads a file and refuses one
    // whose fixed-width min/max has the wrong size ("Unexpected end of
    // stream"); the decoder never looks at them, so refuse too (any column).
    for (ct, lens) in w.chunk_types.iter().zip(&w.stat_lens) {
        let width = match ct {
            Some(0) => Some(1), // BOOLEAN (one PLAIN byte)
            Some(1) | Some(4) => Some(4), // INT32, FLOAT
            Some(2) | Some(5) => Some(8), // INT64, DOUBLE
            Some(3) => Some(12),          // INT96
            _ => None,                    // BYTE_ARRAY / FIXED_LEN_BYTE_ARRAY: variable here
        };
        if let Some(wd) = width {
            if let Some(n) = lens.iter().find(|&&n| n != wd) {
                return unsupported(format!("column statistic of {n} bytes for a {wd}-byte type"));
            }
        }
    }
    // pyarrow refuses a column chunk whose ColumnMetaData.type differs from
    // its schema leaf's physical type (for any column); parquet-rs ignores
    // the chunk's own field. Leaves = elements without children, in order.
    let leaf_types: Vec<Option<i64>> = w
        .children
        .iter()
        .zip(&w.types)
        .skip(1) // the root
        .filter(|(c, _)| c.unwrap_or(0) == 0)
        .map(|(_, t)| *t)
        .collect();
    if !leaf_types.is_empty() {
        for (k, ct) in w.chunk_types.iter().enumerate() {
            if *ct != leaf_types[k % leaf_types.len()] {
                return unsupported("column chunk type differs from its schema leaf type");
            }
        }
    }
    // Schema tree sanity BEFORE parquet-rs builds it (recursively, with
    // num_children-sized allocations): plausible fan-out and depth only.
    let n = w.children.len();
    let mut stack: Vec<i64> = Vec::new(); // remaining children per open group
    for (i, c) in w.children.iter().enumerate() {
        if let Some(top) = stack.last_mut() {
            *top -= 1;
        } else if i != 0 {
            break; // trailing elements: parquet-rs/pyarrow decide; tree is closed
        }
        match *c {
            Some(k) if k < 0 || k as usize > n => {
                return unsupported(format!("schema element with {k} children"));
            }
            Some(k) if k > 0 => {
                stack.push(k);
                if stack.len() > 64 {
                    return unsupported("schema nested more than 64 levels");
                }
            }
            _ => {}
        }
        while stack.last() == Some(&0) {
            stack.pop();
        }
    }
    Ok(())
}

struct Thrift<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Thrift<'a> {
    fn byte(&mut self) -> DResult<u8> {
        let v = *self.b.get(self.pos).ok_or_else(|| DenseError::Corrupt("page header truncated".into()))?;
        self.pos += 1;
        Ok(v)
    }

    fn uvarint(&mut self) -> DResult<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        corrupt("varint too long in page header")
    }

    fn zigzag(&mut self) -> DResult<i64> {
        let u = self.uvarint()?;
        Ok((u >> 1) as i64 ^ -((u & 1) as i64))
    }

    fn i32(&mut self) -> DResult<i32> {
        // pyarrow's thrift truncates the raw varint to 32 bits before zigzag;
        // a wider one would mean something different to it: refuse.
        let raw = self.uvarint()?;
        if raw >> 32 != 0 {
            return unsupported("page header i32 varint wider than 32 bits");
        }
        let raw = raw as u32;
        Ok((raw >> 1) as i32 ^ -((raw & 1) as i32))
    }

    fn skip_bytes(&mut self, n: u64) -> DResult<()> {
        let end = (self.pos as u64).checked_add(n).filter(|&e| e <= self.b.len() as u64);
        let Some(end) = end else {
            return corrupt("page header truncated");
        };
        self.pos = end as usize;
        Ok(())
    }

    /// Next field header of the current struct: (field id, type), or None at STOP.
    fn field(&mut self, last_id: &mut i16) -> DResult<Option<(i16, u8)>> {
        let h = self.byte()?;
        if h == 0 {
            return Ok(None);
        }
        let ty = h & 0x0f;
        let delta = (h >> 4) as i16;
        let id = if delta == 0 {
            i16::try_from(self.zigzag()?).map_err(|_| DenseError::Corrupt("field id".into()))?
        } else {
            last_id
                .checked_add(delta)
                .ok_or_else(|| DenseError::Corrupt("field id overflow in page header".into()))?
        };
        *last_id = id;
        Ok(Some((id, ty)))
    }

    fn skip(&mut self, ty: u8, depth: u32) -> DResult<()> {
        if depth > 32 {
            return corrupt("page header nested too deeply");
        }
        match ty {
            1 | 2 => Ok(()),                          // bool (value in the type nibble)
            3 => self.skip_bytes(1),                  // i8
            4..=6 => self.uvarint().map(|_| ()),      // i16 / i32 / i64
            7 => self.skip_bytes(8),                  // double
            8 => {
                let n = self.uvarint()?;
                if n > MAX_THRIFT_BINARY {
                    return unsupported(format!("thrift binary of {n} bytes"));
                }
                self.skip_bytes(n)
            }
            9 | 10 => {
                let h = self.byte()?;
                let n = if h >> 4 == 15 { self.uvarint()? } else { (h >> 4) as u64 };
                if n > MAX_THRIFT_CONTAINER {
                    return unsupported(format!("thrift container of {n} elements"));
                }
                let et = h & 0x0f;
                for _ in 0..n {
                    if et == 1 || et == 2 {
                        self.skip_bytes(1)?; // bools inside containers are one byte
                    } else {
                        self.skip(et, depth + 1)?;
                    }
                }
                Ok(())
            }
            11 => {
                let n = self.uvarint()?;
                if n > MAX_THRIFT_CONTAINER {
                    return unsupported(format!("thrift map of {n} entries"));
                }
                if n == 0 {
                    return Ok(());
                }
                let kv = self.byte()?;
                for _ in 0..n {
                    for t in [kv >> 4, kv & 0x0f] {
                        if t == 1 || t == 2 {
                            self.skip_bytes(1)?;
                        } else {
                            self.skip(t, depth + 1)?;
                        }
                    }
                }
                Ok(())
            }
            12 => {
                let mut last = 0i16;
                let mut seen = FieldSet::default();
                while let Some((id, t)) = self.field(&mut last)? {
                    seen.insert(id)?;
                    self.skip(t, depth + 1)?;
                }
                Ok(())
            }
            _ => corrupt(format!("unknown thrift type {ty} in page header")),
        }
    }

    fn bool_field(ty: u8) -> DResult<bool> {
        match ty {
            1 => Ok(true),
            2 => Ok(false),
            _ => corrupt("expected a bool field"),
        }
    }
}

const PAGE_DATA: i32 = 0;
const PAGE_INDEX: i32 = 1;
const PAGE_DICT: i32 = 2;
const PAGE_DATA_V2: i32 = 3;

const ENC_PLAIN: i32 = 0;
const ENC_PLAIN_DICTIONARY: i32 = 2;
const ENC_RLE: i32 = 3;
const ENC_RLE_DICTIONARY: i32 = 8;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PageHeader {
    pub page_type: i32,
    pub uncompressed: i32,
    pub compressed: i32,
    pub num_values: i32,
    pub encoding: i32,
    pub def_encoding: i32,
    pub rep_encoding: i32,
    // v2 only
    pub num_nulls: i32,
    pub def_len: i32,
    pub rep_len: i32,
    pub is_compressed: bool,
    /// Header size in bytes.
    pub header_len: usize,
}

/// Parse a thrift-compact PageHeader at the start of `b`.
pub fn parse_page_header(b: &[u8]) -> DResult<PageHeader> {
    let mut t = Thrift { b, pos: 0 };
    let mut h = PageHeader { page_type: -1, is_compressed: true, ..Default::default() };
    let (mut have_type, mut have_sizes) = (false, 0u8);
    // Each sub-header is parsed into its OWN slot; only the one matching
    // page_type is used afterwards (a stray sub-struct of another kind must
    // never override the real one -- that is what pyarrow does too).
    // Values: (num_values, encoding, def_enc / num_nulls, rep_enc / def_len,
    // rep_len, is_compressed), plus a bitmask of the fields present.
    #[derive(Default, Clone, Copy)]
    struct Sub {
        v: [i32; 6],
        is_compressed: Option<bool>,
        seen: u8,
    }
    let (mut dp, mut dict, mut v2): (Option<Sub>, Option<Sub>, Option<Sub>) = (None, None, None);
    let mut last = 0i16;
    let mut seen = FieldSet::default();
    while let Some((id, ty)) = t.field(&mut last)? {
        seen.insert(id)?;
        match (id, ty) {
            (1, 5) => {
                h.page_type = t.i32()?;
                have_type = true;
            }
            (2, 5) => {
                h.uncompressed = t.i32()?;
                have_sizes |= 1;
            }
            (3, 5) => {
                h.compressed = t.i32()?;
                have_sizes |= 2;
            }
            (5, 12) | (7, 12) | (8, 12) => {
                // thrift field id -> slot index in Sub.v
                let map: &[(i16, usize)] = match id {
                    5 => &[(1, 0), (2, 1), (3, 2), (4, 3)],    // num_values, encoding, def_enc, rep_enc
                    7 => &[(1, 0), (2, 1)],                    // num_values, encoding
                    _ => &[(1, 0), (4, 1), (2, 2), (5, 3), (6, 4), (3, 5)], // num_values, encoding, num_nulls, def_len, rep_len, num_rows
                };
                let mut sub = Sub::default();
                let mut l = 0i16;
                let mut sub_seen = FieldSet::default();
                while let Some((fid, fty)) = t.field(&mut l)? {
                    sub_seen.insert(fid)?;
                    if let Some(&(_, slot)) = map.iter().find(|(f, _)| *f == fid).filter(|_| fty == 5) {
                        sub.v[slot] = t.i32()?;
                        sub.seen |= 1 << slot;
                    } else if id == 8 && fid == 7 && (fty == 1 || fty == 2) {
                        sub.is_compressed = Some(Thrift::bool_field(fty)?);
                    } else {
                        t.skip(fty, 1)?;
                    }
                }
                match id {
                    5 => dp = Some(sub),
                    7 => dict = Some(sub),
                    _ => v2 = Some(sub),
                }
            }
            _ => t.skip(ty, 0)?,
        }
    }
    if !have_type || have_sizes != 3 {
        return corrupt("page header missing required fields");
    }
    // Any sub-header that is present must be complete (pyarrow's reader
    // fails on an incomplete one even when it belongs to another page type).
    for (sub, mask, what) in [(dp, 0b1111u8, "data_page_header"), (dict, 0b11, "dictionary_page_header"),
                              (v2, 0b111111, "data_page_header_v2")] {
        if let Some(s) = sub {
            if s.seen & mask != mask {
                return corrupt(format!("{what} missing required fields"));
            }
        }
    }
    let need = |sub: Option<Sub>, mask: u8, what: &str| -> DResult<Sub> {
        match sub {
            Some(s) if s.seen & mask == mask => Ok(s),
            _ => corrupt(format!("{what} page without its required sub-header fields")),
        }
    };
    match h.page_type {
        PAGE_DATA => {
            let s = need(dp, 0b1111, "DATA_PAGE")?;
            (h.num_values, h.encoding, h.def_encoding, h.rep_encoding) = (s.v[0], s.v[1], s.v[2], s.v[3]);
        }
        PAGE_DICT => {
            let s = need(dict, 0b11, "DICTIONARY_PAGE")?;
            (h.num_values, h.encoding) = (s.v[0], s.v[1]);
        }
        PAGE_DATA_V2 => {
            let s = need(v2, 0b111111, "DATA_PAGE_V2")?;
            (h.num_values, h.encoding, h.num_nulls, h.def_len, h.rep_len) = (s.v[0], s.v[1], s.v[2], s.v[3], s.v[4]);
            if s.v[5] < 0 {
                return corrupt("negative num_rows in DATA_PAGE_V2 header");
            }
            h.is_compressed = s.is_compressed.unwrap_or(true);
        }
        _ => {}
    }
    if h.uncompressed < 0 || h.compressed < 0 || h.num_values < 0 || h.def_len < 0 || h.rep_len < 0 || h.num_nulls < 0 {
        return corrupt("negative size in page header");
    }
    h.header_len = t.pos;
    if h.header_len > MAX_PAGE_HEADER {
        return unsupported(format!("page header of {} bytes", h.header_len));
    }
    Ok(h)
}

// ---------------------------------------------------------------------------
// RLE / bit-packed hybrid
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub enum Run<'a> {
    Rle { value: u32, count: usize },
    /// `count` values (a multiple of 8, possibly more than the caller needs:
    /// the last group is padded), `bw` bits each, LSB-first.
    Packed { bytes: &'a [u8], count: usize },
}

pub struct Hybrid<'a> {
    data: &'a [u8],
    pos: usize,
    bw: u32,
    /// Reject a truncated final bit-packed run (pyarrow does, for levels).
    strict: bool,
}

impl<'a> Hybrid<'a> {
    pub fn new(data: &'a [u8], bw: u32) -> DResult<Self> {
        Self::with_strict(data, bw, true)
    }

    pub fn with_strict(data: &'a [u8], bw: u32, strict: bool) -> DResult<Self> {
        if bw > 32 {
            return corrupt(format!("bit width {bw}"));
        }
        Ok(Self { data, pos: 0, bw, strict })
    }

    pub fn next_run(&mut self) -> DResult<Option<Run<'a>>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let mut t = Thrift { b: self.data, pos: self.pos };
        let h = t.uvarint()?;
        self.pos = t.pos;
        if h & 1 == 0 {
            let count = usize::try_from(h >> 1).map_err(|_| DenseError::Corrupt("run length".into()))?;
            if count == 0 || count > i32::MAX as usize {
                return corrupt(format!("RLE run of {count} values"));
            }
            let nb = self.bw.div_ceil(8) as usize;
            if self.pos + nb > self.data.len() {
                return corrupt("RLE run truncated");
            }
            let mut v = 0u32;
            for (i, &b) in self.data[self.pos..self.pos + nb].iter().enumerate() {
                v |= (b as u32) << (8 * i);
            }
            self.pos += nb;
            if self.bw < 32 && v >> self.bw != 0 {
                return corrupt("RLE value wider than its bit width");
            }
            Ok(Some(Run::Rle { value: v, count }))
        } else {
            let groups = usize::try_from(h >> 1).map_err(|_| DenseError::Corrupt("run length".into()))?;
            let run_len = || DenseError::Corrupt("run length".into());
            if groups == 0 || groups > (i32::MAX as usize) / 8 {
                return corrupt(format!("bit-packed run of {groups} groups"));
            }
            let want = groups.checked_mul(self.bw as usize).ok_or_else(run_len)?;
            let max_count = groups.checked_mul(8).ok_or_else(run_len)?;
            // A run whose declared length runs past the data is rejected
            // (pyarrow refuses those too); non-strict mode keeps whatever
            // whole values are present -- only used by unit tests.
            let have = want.min(self.data.len() - self.pos);
            if self.strict && have < want {
                return corrupt("bit-packed level run truncated");
            }
            let bytes = &self.data[self.pos..self.pos + have];
            self.pos += have;
            let count = if self.bw == 0 { max_count } else { (have * 8 / self.bw as usize).min(max_count) };
            Ok(Some(Run::Packed { bytes, count }))
        }
    }
}

/// Value `k` of a bit-packed run.
#[inline]
fn unpack(bytes: &[u8], bw: u32, k: usize) -> u32 {
    if bw == 0 {
        return 0;
    }
    let bit = k * bw as usize;
    let byte = bit / 8;
    let mut tmp = [0u8; 8];
    let n = (bytes.len() - byte).min(8);
    tmp[..n].copy_from_slice(&bytes[byte..byte + n]);
    let w = u64::from_le_bytes(tmp) >> (bit % 8);
    (w & ((1u64 << bw) - 1)) as u32
}

fn level_width(max: i16) -> u32 {
    if max <= 0 { 0 } else { 16 - (max as u16).leading_zeros() }
}

/// Every one of the first `n` definition levels must equal `max_def`.
#[cfg(test)]
fn check_def(data: &[u8], bw: u32, max_def: i16, n: usize) -> DResult<()> {
    check_def_stop(data, bw, max_def, n, &AtomicBool::new(false))
}

/// Polled every ~1M levels / 4096 runs so a cancel need not wait out a huge page.
#[inline]
fn poll(stop: &AtomicBool, tick: &mut u32) -> DResult<()> {
    *tick = tick.wrapping_add(1);
    if *tick & 0xFFF == 0 && stop.load(Ordering::Relaxed) {
        return Err(DenseError::Fetch(FetchError::Cancelled));
    }
    Ok(())
}

fn check_def_stop(data: &[u8], bw: u32, max_def: i16, n: usize, stop: &AtomicBool) -> DResult<()> {
    let want = max_def as u32;
    let mut h = Hybrid::with_strict(data, bw, true)?;
    let mut seen = 0usize;
    let mut tick = 0u32;
    while seen < n {
        poll(stop, &mut tick)?;
        match h.next_run()? {
            None => return corrupt(format!("definition levels end after {seen} of {n}")),
            Some(Run::Rle { value, count }) => {
                if value != want {
                    return unsupported("null or empty list in the dense column");
                }
                seen += count.min(n - seen);
            }
            Some(Run::Packed { bytes, count }) => {
                let take = count.min(n - seen);
                for k in 0..take {
                    if k & 0xF_FFFF == 0xF_FFFF && stop.load(Ordering::Relaxed) {
                        return Err(DenseError::Fetch(FetchError::Cancelled));
                    }
                    if unpack(bytes, bw, k) != want {
                        return unsupported("null or empty list in the dense column");
                    }
                }
                seen += take;
            }
        }
    }
    Ok(())
}

/// Repetition levels for values `first .. first+n` of the chunk: value `i`
/// must have level 0 (starts a row) iff `i % dim == 0`, else 1. This is what
/// proves every row has exactly `dim` values.
#[cfg(test)]
fn check_rep(data: &[u8], bw: u32, dim: u64, first: u64, n: usize) -> DResult<()> {
    check_rep_stop(data, bw, dim, first, n, &AtomicBool::new(false))
}

fn check_rep_stop(data: &[u8], bw: u32, dim: u64, first: u64, n: usize, stop: &AtomicBool) -> DResult<()> {
    let ragged = || unsupported::<()>("rows are not all `dim` long");
    let mut h = Hybrid::with_strict(data, bw, true)?;
    let mut i = first;
    let end = first + n as u64;
    let mut tick = 0u32;
    while i < end {
        poll(stop, &mut tick)?;
        match h.next_run()? {
            None => return corrupt(format!("repetition levels end at value {i} of {end}")),
            Some(Run::Rle { value, count }) => {
                let c = (count as u64).min(end - i);
                if c == 0 {
                    continue;
                }
                match value {
                    // c consecutive row starts: only possible with dim == 1.
                    0 => {
                        if i % dim != 0 || (c > 1 && dim != 1) {
                            return ragged();
                        }
                    }
                    // c continuations: must not cross a row start.
                    1 => {
                        let off = i % dim;
                        if off == 0 || off + c > dim {
                            return ragged();
                        }
                    }
                    _ => return ragged(),
                }
                i += c;
            }
            Some(Run::Packed { bytes, count }) => {
                let take = (count as u64).min(end - i) as usize;
                for k in 0..take {
                    if k & 0xF_FFFF == 0xF_FFFF && stop.load(Ordering::Relaxed) {
                        return Err(DenseError::Fetch(FetchError::Cancelled));
                    }
                    let lvl = unpack(bytes, bw, k);
                    let starts = (i + k as u64) % dim == 0;
                    if lvl != if starts { 0 } else { 1 } {
                        return ragged();
                    }
                }
                i += take as u64;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Chunk decode
// ---------------------------------------------------------------------------

/// Copy `n` dictionary-encoded values (indices in `data`) into `out`.
#[cfg(test)]
fn decode_dict(data: &[u8], dict: &[u8], elem: usize, n: usize, out: &mut [u8]) -> DResult<()> {
    decode_dict_stoppable(data, dict, elem, n, out, &AtomicBool::new(false))
}

/// `decode_dict`, polling `stop` every ~1M values.
fn decode_dict_stoppable(
    data: &[u8],
    dict: &[u8],
    elem: usize,
    n: usize,
    out: &mut [u8],
    stop: &AtomicBool,
) -> DResult<()> {
    let Some((&bw, rest)) = data.split_first() else {
        return corrupt("dictionary page data is empty");
    };
    let n_dict = dict.len() / elem;
    let mut h = Hybrid::new(rest, bw as u32)?;
    let mut w = 0usize;
    let put = |idx: u32, w: usize, out: &mut [u8]| -> DResult<()> {
        let i = idx as usize;
        if i >= n_dict {
            return corrupt(format!("dictionary index {i} >= dictionary size {n_dict}"));
        }
        out[w * elem..(w + 1) * elem].copy_from_slice(&dict[i * elem..(i + 1) * elem]);
        Ok(())
    };
    while w < n {
        match h.next_run()? {
            None => return corrupt(format!("dictionary indices end after {w} of {n}")),
            Some(Run::Rle { value, count }) => {
                let c = count.min(n - w);
                for _ in 0..c {
                    if w & 0xF_FFFF == 0 && stop.load(Ordering::Relaxed) {
                        return Err(DenseError::Fetch(FetchError::Cancelled));
                    }
                    put(value, w, out)?;
                    w += 1;
                }
            }
            Some(Run::Packed { bytes, count }) => {
                let c = count.min(n - w);
                for k in 0..c {
                    if w & 0xF_FFFF == 0 && stop.load(Ordering::Relaxed) {
                        return Err(DenseError::Fetch(FetchError::Cancelled));
                    }
                    put(unpack(bytes, bw as u32, k), w, out)?;
                    w += 1;
                }
            }
        }
    }
    Ok(())
}

/// Upper bound on a sane decompressed data page holding `n` values: the
/// values themselves (PLAIN `n*elem`, or dictionary indices -- at most a
/// few bytes each even as 1-value RLE runs) plus both level streams (each
/// at most ~6 bytes per value as 1-value runs) plus headers/prefixes. Keeps
/// a header that LIES about its size from allocating gigabytes.
fn page_cap(n: usize, elem: usize) -> usize {
    n.saturating_mul(elem.max(6) + 12).saturating_add(256)
}


/// Resize `v` to exactly `len` bytes, failing cleanly (not aborting the
/// process) if the allocation is refused.
fn sized(v: &mut Vec<u8>, len: usize) -> DResult<()> {
    if len > v.len() {
        v.try_reserve(len - v.len())
            .map_err(|_| DenseError::Unsupported(format!("cannot allocate {len} bytes for a page")))?;
    }
    v.resize(len, 0);
    Ok(())
}

/// Decode one column chunk (`buf` = its bytes) into `out`, which is exactly
/// `chunk.n_rows * dim * elem` bytes: the chunk's rows of the output matrix.
/// `scratch` is reused for decompression.
pub fn decode_chunk(plan: &DensePlan, chunk: &ChunkPlan, buf: &[u8], out: &mut [u8], scratch: &mut Vec<u8>) -> DResult<()> {
    decode_chunk_stoppable(plan, chunk, buf, out, scratch, &AtomicBool::new(false))
}

/// `decode_chunk`, checking `stop` between pages so a cancelled read does
/// not have to finish decoding a whole (possibly multi-GB) row group.
pub fn decode_chunk_stoppable(
    plan: &DensePlan,
    chunk: &ChunkPlan,
    buf: &[u8],
    out: &mut [u8],
    scratch: &mut Vec<u8>,
    stop: &AtomicBool,
) -> DResult<()> {
    let elem = plan.elem;
    let total = chunk.num_values;
    if out.len() as u64 != total * elem as u64 {
        return Err(DenseError::Corrupt("output slice does not match the chunk".into()));
    }
    let rep_bw = level_width(1);
    let def_bw = level_width(plan.max_def);
    let mut dict: Option<Vec<u8>> = None;
    let mut pos = 0usize;
    // Whether the previous page was a data page holding values (see the
    // empty-page rule below).
    let mut prev_nonempty_data = false;
    let mut done = 0u64; // values written so far

    while done < total {
        if stop.load(Ordering::Relaxed) {
            return Err(DenseError::Fetch(FetchError::Cancelled));
        }
        if pos >= buf.len() {
            return corrupt(format!("column chunk ends after {done} of {total} values"));
        }
        let h = parse_page_header(&buf[pos..])?;
        let body_start = pos + h.header_len;
        let body_end = body_start
            .checked_add(h.compressed as usize)
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| DenseError::Corrupt("page runs past its column chunk".into()))?;
        let body = &buf[body_start..body_end];
        pos = body_end;
        let usize_u = h.uncompressed as usize;

        match h.page_type {
            PAGE_DICT => {
                prev_nonempty_data = false;
                if dict.is_some() {
                    return corrupt("two dictionary pages in one column chunk");
                }
                if h.encoding != ENC_PLAIN && h.encoding != ENC_PLAIN_DICTIONARY {
                    return unsupported(format!("dictionary page encoding {}", h.encoding));
                }
                // A dictionary larger than the chunk's own output would be
                // pointless; refuse it before allocating anything.
                if usize_u as u64 > chunk.num_values.saturating_mul(elem as u64).max(64) {
                    return unsupported("dictionary page larger than its column chunk's values");
                }
                let mut d = Vec::new();
                chunk.codec.decompress_vec(body, usize_u, &mut d)?;
                if d.len() != h.num_values as usize * elem {
                    return corrupt("dictionary page size does not match its value count");
                }
                dict = Some(d);
            }
            PAGE_INDEX => {
                prev_nonempty_data = false;
                continue;
            }
            PAGE_DATA | PAGE_DATA_V2 => {
                let n = h.num_values as usize;
                if n == 0 {
                    // pyarrow reads exactly one shape of mid-chunk empty data
                    // page correctly (and its own writer emits it): a single
                    // empty page at a row boundary right after a non-empty
                    // data page. As the first page of a chunk (even after a
                    // dictionary/index page), after another empty page, or
                    // mid-row, pyarrow drops the rest of the row group or
                    // errors -- never ours (measured, batch-size independent).
                    if !(prev_nonempty_data && done % plan.dim == 0) {
                        return unsupported("empty data page where pyarrow would drop or reject data");
                    }
                    // ... and it is otherwise validated exactly like any
                    // other page below (pyarrow decodes it fully too).
                }
                prev_nonempty_data = n > 0;
                if done + n as u64 > total {
                    return corrupt("pages hold more values than the column chunk");
                }
                // levels: (rep bytes, def bytes); values: decompressed bytes.
                let (rep, def, values): (&[u8], &[u8], &[u8]);
                if h.page_type == PAGE_DATA {
                    if h.rep_encoding != ENC_RLE || h.def_encoding != ENC_RLE {
                        return unsupported("BIT_PACKED (deprecated) level encoding");
                    }
                    let page: &[u8] = if chunk.codec == Codec::Uncompressed {
                        if body.len() != usize_u {
                            return corrupt("uncompressed page size mismatch");
                        }
                        body
                    } else {
                        if usize_u > page_cap(n, elem) {
                            return unsupported(format!("page claims {usize_u} bytes for {n} values"));
                        }
                        chunk.codec.decompress_vec(body, usize_u, scratch)?;
                        &scratch[..]
                    };
                    let mut p = 0usize;
                    let take_levels = |page: &'_ [u8], p: &mut usize| -> DResult<(usize, usize)> {
                        if *p + 4 > page.len() {
                            return corrupt("level length prefix truncated");
                        }
                        let l = u32::from_le_bytes(page[*p..*p + 4].try_into().unwrap()) as usize;
                        let s = *p + 4;
                        if s + l > page.len() {
                            return corrupt("levels run past the page");
                        }
                        *p = s + l;
                        Ok((s, s + l))
                    };
                    let (rs, re) = take_levels(page, &mut p)?; // max_rep == 1 always
                    let (ds, de) = if plan.max_def > 0 { take_levels(page, &mut p)? } else { (p, p) };
                    rep = &page[rs..re];
                    def = &page[ds..de];
                    values = &page[p..];
                } else {
                    if h.num_nulls != 0 {
                        return unsupported("null values in the dense column");
                    }
                    let (rl, dl) = (h.rep_len as usize, h.def_len as usize);
                    if rl + dl > body.len() || rl + dl > usize_u {
                        return corrupt("v2 levels run past the page");
                    }
                    rep = &body[..rl];
                    def = &body[rl..rl + dl];
                    let vbody = &body[rl + dl..];
                    let vlen = usize_u - rl - dl;
                    values = if !h.is_compressed || chunk.codec == Codec::Uncompressed {
                        if vbody.len() != vlen {
                            return corrupt("v2 values size mismatch");
                        }
                        vbody
                    } else {
                        if vlen > page_cap(n, elem) {
                            return unsupported(format!("page claims {vlen} value bytes for {n} values"));
                        }
                        chunk.codec.decompress_vec(vbody, vlen, scratch)?;
                        &scratch[..]
                    };
                }

                check_rep_stop(rep, rep_bw, plan.dim, done, n, stop)?;
                if plan.max_def > 0 {
                    check_def_stop(def, def_bw, plan.max_def, n, stop)?;
                }
                let dst = &mut out[done as usize * elem..(done as usize + n) * elem];
                match h.encoding {
                    ENC_PLAIN => {
                        if values.len() != n * elem {
                            return corrupt(format!(
                                "PLAIN page holds {} value bytes for {n} values of {elem} bytes",
                                values.len()
                            ));
                        }
                        // In slices, so a cancel need not wait out a huge page.
                        for (d, v) in dst.chunks_mut(64 << 20).zip(values.chunks(64 << 20)) {
                            if stop.load(Ordering::Relaxed) {
                                return Err(DenseError::Fetch(FetchError::Cancelled));
                            }
                            d.copy_from_slice(v);
                        }
                    }
                    ENC_PLAIN_DICTIONARY | ENC_RLE_DICTIONARY => {
                        let d = dict.as_deref().ok_or_else(|| {
                            DenseError::Corrupt("dictionary-encoded page without a dictionary page".into())
                        })?;
                        decode_dict_stoppable(values, d, elem, n, dst, stop)?;
                    }
                    e => return unsupported(format!("value encoding {e}")),
                }
                done += n as u64;
            }
            t => return unsupported(format!("page type {t}")),
        }
    }
    // After the last value: only index pages or empty data pages may follow.
    while pos < buf.len() {
        let h = parse_page_header(&buf[pos..])?;
        let end = (pos + h.header_len)
            .checked_add(h.compressed as usize)
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| DenseError::Corrupt("page runs past its column chunk".into()))?;
        let empty_data = matches!(h.page_type, PAGE_DATA | PAGE_DATA_V2) && h.num_values == 0;
        if h.page_type != PAGE_INDEX && !empty_data {
            return unsupported(format!(
                "{} trailing bytes after the column chunk's last value",
                buf.len() - pos
            ));
        }
        pos = end;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fetch + decode, pipelined over row groups
// ---------------------------------------------------------------------------

/// Fetch every chunk of `plan` and decode it into `dest` (at least
/// `plan.nbytes()` bytes). Up to `in_flight` chunks are fetched/decoded at
/// once; decoding runs on tokio's blocking pool, so chunks decode in parallel
/// while later ones download.
///
/// Every task is joined before this returns -- on error too: tasks are never
/// aborted (an aborted task could leave a blocking decode still writing into
/// `dest`); a shared flag stops new work instead. On error `dest` contents
/// are unspecified.
pub async fn read_dense(
    f: Arc<Fetcher>,
    plan: Arc<DensePlan>,
    dest: Dest,
    in_flight: usize,
    stop: Arc<AtomicBool>,
) -> DResult<u64> {
    let need = plan.nbytes();
    if dest.len < need {
        return Err(DenseError::Fetch(FetchError::Invalid(format!(
            "dest too small: need {need} bytes, buffer has {}",
            dest.len
        ))));
    }
    let in_flight = in_flight.max(1);
    // One flag for the whole read: an error in any chunk (or the caller)
    // stops new chunks AND the parts of chunks already downloading.
    // Staging buffers are recycled across chunks: a fresh multi-MB
    // allocation per chunk would page-fault its way through every byte.
    let pool: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let row_bytes = plan.dim * plan.elem as u64;

    let mut set: JoinSet<DResult<()>> = JoinSet::new();
    let mut next = 0usize;
    let mut first_err: Option<DenseError> = None;
    loop {
        while first_err.is_none() && !stop.load(Ordering::Relaxed) && set.len() < in_flight && next < plan.chunks.len() {
            let ci = next;
            next += 1;
            let (f, plan, stop, pool) = (f.clone(), plan.clone(), stop.clone(), pool.clone());
            set.spawn(async move {
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let c = &plan.chunks[ci];
                let clen = c.len as usize;
                let mut buf = pool.lock().unwrap().pop().unwrap_or_default();
                // Grow only: re-zeroing a recycled buffer would be a wasted
                // memory pass (the fetch overwrites [0, clen) completely).
                if buf.len() < clen {
                    buf.resize(clen, 0);
                }
                let staged = Dest { addr: buf.as_mut_ptr() as usize, len: c.len };
                let opts = ReadOpts { stop: stop.clone(), if_match: plan.etag.clone() };
                let fetched = f.read_spans_opts(&plan.url, &[Span { src: c.offset, len: c.len, dst: 0 }], staged, opts).await;
                if let Err(e) = fetched {
                    pool.lock().unwrap().push(buf);
                    return Err(e.into());
                }
                if stop.load(Ordering::Relaxed) {
                    pool.lock().unwrap().push(buf);
                    return Ok(());
                }
                let out_off = c.row_start * row_bytes;
                let out_len = c.n_rows * row_bytes;
                let (plan2, pool2, stop2) = (plan.clone(), pool.clone(), stop.clone());
                // Awaited to completion (never aborted), so the blocking
                // closure is finished before this task -- and read_dense -- end.
                tokio::task::spawn_blocking(move || {
                    // SAFETY: [out_off, out_off+out_len) lies inside dest
                    // (plan rows are disjoint, prefix-summed, and sum to
                    // nbytes <= dest.len) and only this chunk writes it.
                    let out = unsafe {
                        std::slice::from_raw_parts_mut((dest.addr + out_off as usize) as *mut u8, out_len as usize)
                    };
                    let mut scratch = Vec::new();
                    let r = decode_chunk_stoppable(&plan2, &plan2.chunks[ci], &buf[..clen], out, &mut scratch, &stop2);
                    pool2.lock().unwrap().push(buf);
                    r
                })
                .await
                .map_err(|e| DenseError::Corrupt(format!("decode task failed: {e}")))?
            });
        }
        match set.join_next().await {
            None => break,
            Some(joined) => {
                let r = joined.unwrap_or_else(|e| Err(DenseError::Corrupt(format!("chunk task failed: {e}"))));
                if let Err(e) = r {
                    stop.store(true, Ordering::Relaxed);
                    // Keep the first REAL error; a Cancelled part only
                    // echoes it (or the caller's stop).
                    let echo = matches!(e, DenseError::Fetch(FetchError::Cancelled));
                    if first_err.is_none() && !echo {
                        first_err = Some(e);
                    }
                }
            }
        }
    }
    match first_err {
        Some(e) => Err(e),
        // Stopped from outside (or chunks never ran): not a complete read.
        None if stop.load(Ordering::Relaxed) || next < plan.chunks.len() => {
            Err(DenseError::Fetch(FetchError::Cancelled))
        }
        None => Ok(need),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uleb(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn rle(value: u32, count: u64, bw: u32, out: &mut Vec<u8>) {
        uleb(count << 1, out);
        for i in 0..bw.div_ceil(8) {
            out.push((value >> (8 * i)) as u8);
        }
    }

    fn packed(vals: &[u32], bw: u32, out: &mut Vec<u8>) {
        assert_eq!(vals.len() % 8, 0);
        uleb(((vals.len() / 8) as u64) << 1 | 1, out);
        let mut acc = 0u128;
        let mut nbits = 0;
        for &v in vals {
            acc |= (v as u128) << nbits;
            nbits += bw;
            while nbits >= 8 {
                out.push(acc as u8);
                acc >>= 8;
                nbits -= 8;
            }
        }
        assert_eq!(nbits, 0);
    }

    #[test]
    fn hybrid_runs_roundtrip() {
        let mut d = Vec::new();
        rle(5, 3, 3, &mut d);
        packed(&[0, 1, 2, 3, 4, 5, 6, 7], 3, &mut d);
        let mut h = Hybrid::new(&d, 3).unwrap();
        assert_eq!(h.next_run().unwrap(), Some(Run::Rle { value: 5, count: 3 }));
        match h.next_run().unwrap() {
            Some(Run::Packed { bytes, count }) => {
                assert_eq!(count, 8);
                let got: Vec<u32> = (0..8).map(|k| unpack(bytes, 3, k)).collect();
                assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
            }
            r => panic!("{r:?}"),
        }
        assert_eq!(h.next_run().unwrap(), None);
    }

    #[test]
    fn rep_levels_accept_exact_rows_and_reject_ragged() {
        // dim 4, two rows: 0 1 1 1 | 0 1 1 1 -- as packed and as RLE runs.
        let lv = [0, 1, 1, 1, 0, 1, 1, 1];
        let mut p = Vec::new();
        packed(&lv, 1, &mut p);
        check_rep(&p, 1, 4, 0, 8).unwrap();
        let mut r = Vec::new();
        rle(0, 1, 1, &mut r);
        rle(1, 3, 1, &mut r);
        rle(0, 1, 1, &mut r);
        rle(1, 3, 1, &mut r);
        check_rep(&r, 1, 4, 0, 8).unwrap();
        // Same levels but dim 3 / 5: ragged.
        assert!(matches!(check_rep(&p, 1, 3, 0, 8), Err(DenseError::Unsupported(_))));
        assert!(matches!(check_rep(&r, 1, 5, 0, 8), Err(DenseError::Unsupported(_))));
        // A page that starts mid-row (first = 2): levels 1 1 0 1 1 1 0 1.
        let mut m = Vec::new();
        packed(&[1, 1, 0, 1, 1, 1, 0, 1], 1, &mut m);
        check_rep(&m, 1, 4, 2, 8).unwrap();
        assert!(check_rep(&m, 1, 4, 0, 8).is_err());
        // dim 1: a long run of zeros is fine.
        let mut z = Vec::new();
        rle(0, 100, 1, &mut z);
        check_rep(&z, 1, 1, 0, 100).unwrap();
        assert!(check_rep(&z, 1, 2, 0, 100).is_err());
        // RLE run of continuations crossing a row boundary.
        let mut x = Vec::new();
        rle(0, 1, 1, &mut x);
        rle(1, 4, 1, &mut x);
        assert!(check_rep(&x, 1, 4, 0, 5).is_err());
        // Levels that run out early are corrupt, not ragged.
        assert!(matches!(check_rep(&x, 1, 8, 0, 9), Err(DenseError::Corrupt(_))));
    }

    #[test]
    fn def_levels() {
        let mut d = Vec::new();
        rle(3, 10, 2, &mut d);
        check_def(&d, 2, 3, 10).unwrap();
        let mut n = Vec::new();
        rle(3, 4, 2, &mut n);
        packed(&[3, 3, 2, 3, 3, 3, 3, 3], 2, &mut n);
        assert!(matches!(check_def(&n, 2, 3, 12), Err(DenseError::Unsupported(_))));
        // The null sits past n: not looked at.
        check_def(&n, 2, 3, 6).unwrap();
    }

    #[test]
    fn dict_gather() {
        let dict: Vec<u8> = [10u32, 20, 30].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut data = vec![2u8];
        rle(2, 3, 2, &mut data);
        packed(&[0, 1, 2, 1, 0, 0, 0, 0], 2, &mut data);
        let mut out = vec![0u8; 5 * 4];
        decode_dict(&data, &dict, 4, 5, &mut out).unwrap();
        let got: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![30, 30, 30, 10, 20]);
        let mut bad = vec![2u8];
        rle(3, 1, 2, &mut bad);
        assert!(decode_dict(&bad, &dict, 4, 1, &mut out[..4]).is_err());
    }

    #[test]
    fn page_header_parse() {
        // PageHeader{type=0, uncompressed=100, compressed=100,
        //   data_page_header{num_values=8, encoding=0, def=3, rep=3}}
        let mut b = Vec::new();
        let i32f = |id_delta: u8, v: i32, b: &mut Vec<u8>| {
            b.push(id_delta << 4 | 5);
            uleb(((v << 1) ^ (v >> 31)) as u32 as u64, b);
        };
        i32f(1, 0, &mut b);
        i32f(1, 100, &mut b);
        i32f(1, 100, &mut b);
        b.push(2 << 4 | 12); // field 5 (delta 2), struct
        i32f(1, 8, &mut b);
        i32f(1, 0, &mut b);
        i32f(1, 3, &mut b);
        i32f(1, 3, &mut b);
        b.push(0); // end data_page_header
        b.push(0); // end PageHeader
        let n = b.len();
        b.extend_from_slice(&[0xAA; 4]); // body follows
        let h = parse_page_header(&b).unwrap();
        assert_eq!((h.page_type, h.uncompressed, h.compressed, h.num_values), (0, 100, 100, 8));
        assert_eq!((h.encoding, h.def_encoding, h.rep_encoding, h.header_len), (0, 3, 3, n));
        assert!(parse_page_header(&b[..n - 3]).is_err());
    }
}
