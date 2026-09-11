//! Native `match_text` corpus scanning.
//!
//! `scan_into` reproduces the Arrow tokenization/lowercasing pipeline in one
//! compiled pass over the string value buffer, writing matches directly into
//! the caller's `(n_tokens, n_rows)` grid while the GIL is released.
//!
//! Token boundaries and lowercase mappings come from tables derived by Python
//! from the installed Arrow build, so the native path follows Arrow's behavior
//! rather than hardcoding Unicode rules.
//!
//! Vocabulary hashes are used only to locate candidates. Every candidate is
//! compared against the full lowered token by `tokenizer::token_eq`, so hash
//! collisions cannot change the result.
//!
//! If the required Arrow behavior cannot be represented or verified, Python
//! refuses the native path and falls back to the Arrow implementation.

use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod tokenizer;
use tokenizer::{hash_bytes, lower_into, token_eq, walk_row, Tables};

const N_CODEPOINTS: usize = 0x11_0000;

/// Owned per-codepoint tables used by GIL-free scans.
///
/// Owning the data prevents caller mutation or deallocation during a scan.
#[pyclass(frozen)]
pub struct ScanTables {
    alnum: Vec<u8>,
    lower: Vec<u32>,
}
#[pymethods]
impl ScanTables {
    #[new]
    fn new(alnum: PyReadonlyArray1<u8>, lower: PyReadonlyArray1<u32>) -> PyResult<Self> {
        if alnum.len() != N_CODEPOINTS || lower.len() != N_CODEPOINTS {
            return Err(PyValueError::new_err(format!(
                "tables must have {N_CODEPOINTS} entries, got {} and {}",
                alnum.len(),
                lower.len()
            )));
        }

        Ok(Self {
            alnum: alnum.as_slice()?.to_vec(),
            lower: lower.as_slice()?.to_vec(),
        })
    }
}

impl ScanTables {
    /// Borrow the owned tables for tokenization.
    fn view(&self) -> Tables<'_> {
        Tables::new(&self.alnum, &self.lower)
    }
}
/// Query vocabulary with a bit filter in front of an open-addressed hash table.
///
/// Vocabulary entries are stored as raw bytes because Arrow compares the
/// lowered corpus token against the provided value set verbatim.
///
/// The bit filter and hash table only narrow candidate matches. Every candidate
/// is confirmed by `token_eq`, so neither filter nor hash collisions can change
/// the result.
#[pyclass(frozen)]
pub struct ScanVocab {
    /// Vocabulary entries in caller order.
    tokens: Vec<Vec<u8>>,

    /// `slot -> (hash tag, token index + 1)`; index 0 marks an empty slot.
    slots: Vec<(u32, u32)>,

    mask: usize,
    filter: Vec<u64>,
    fmask: u64,
    min_len: usize,
    max_len: usize,
}

#[pymethods]
impl ScanVocab {
    #[new]
    fn new(tokens: Vec<Vec<u8>>) -> PyResult<Self> {
        if tokens.is_empty() {
            return Err(PyValueError::new_err("empty vocabulary"));
        }
        if tokens.len() >= u32::MAX as usize {
            return Err(PyValueError::new_err("vocabulary too large"));
        }
        if tokens.iter().any(|t| t.is_empty()) {
            // Empty tokens cannot be produced by the native walk.
            return Err(PyValueError::new_err(
                "an empty vocabulary token cannot be served by the native scan",
            ));
        }

        let min_len = tokens.iter().map(|t| t.len()).min().unwrap();
        let max_len = tokens.iter().map(|t| t.len()).max().unwrap();

        // Keep the hash table at load factor <= 1/4.
        let mut cap = 16usize;
        while cap < tokens.len() * 4 {
            cap <<= 1;
        }
        let mut slots = vec![(0u32, 0u32); cap];

        // Small bit filter avoids most hash-table probes.
        let mut fbits = 1024usize;
        while fbits < tokens.len() * 32 && fbits < (1 << 20) {
            fbits <<= 1;
        }
        let mut filter = vec![0u64; fbits / 64];
        let fmask = (fbits - 1) as u64;

        for (i, tok) in tokens.iter().enumerate() {
            let h = hash_bytes(tok);

            let b = h & fmask;
            filter[(b >> 6) as usize] |= 1u64 << (b & 63);

            let mut p = (h as usize) & (cap - 1);
            while slots[p].1 != 0 {
                p = (p + 1) & (cap - 1);
            }
            slots[p] = ((h >> 32) as u32, (i + 1) as u32);
        }

        Ok(Self {
            tokens,
            slots,
            mask: cap - 1,
            filter,
            fmask,
            min_len,
            max_len,
        })
    }

    /// Return the vocabulary size.
    fn __len__(&self) -> usize {
        self.tokens.len()
    }
}
impl ScanVocab {
    /// Return the vocabulary index matching the lowered form of `tok`.
    ///
    /// Length and the bit filter reject cheap misses before probing the
    /// hash table. Candidate matches are always confirmed by `token_eq`.
    #[inline(always)]
    fn lookup(&self, tok: &[u8], h: u64, low_len: usize, t: &Tables) -> Option<u32> {
        if low_len < self.min_len || low_len > self.max_len {
            return None;
        }

        let b = h & self.fmask;
        if (self.filter[(b >> 6) as usize] >> (b & 63)) & 1 == 0 {
            return None;
        }

        let tag = (h >> 32) as u32;
        let mut p = (h as usize) & self.mask;

        loop {
            let (sh, si) = self.slots[p];
            if si == 0 {
                return None;
            }

            if sh == tag {
                let cand = &self.tokens[(si - 1) as usize];
                if cand.len() == low_len && token_eq(tok, cand, t) {
                    return Some(si - 1);
                }
            }

            p = (p + 1) & self.mask;
        }
    }
}
/// Scan one string batch into a zeroed `(n_tokens, n_rows)` match grid.
///
/// `offsets` are relative to `values`, `valid` optionally marks null rows, and
/// `grid` is updated in place without being cleared. The GIL is released during
/// the scan.
#[pyfunction]
#[pyo3(signature = (values, offsets, valid, tables, vocab, grid))]
fn scan_into(
    py: Python<'_>,
    values: PyReadonlyArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    valid: Option<PyReadonlyArray1<bool>>,
    tables: &ScanTables,
    vocab: &ScanVocab,
    mut grid: PyReadwriteArray2<bool>,
) -> PyResult<()> {
    let vals = values.as_slice()?;
    let offs = offsets.as_slice()?;

    if offs.is_empty() {
        return Err(PyValueError::new_err("offsets must hold at least one entry"));
    }

    let n_rows = offs.len() - 1;
    let dims = grid.shape().to_vec();
    if dims.len() != 2 || dims[0] != vocab.tokens.len() || dims[1] != n_rows {
        return Err(PyValueError::new_err(format!(
            "grid is {dims:?}, expected [{}, {n_rows}]",
            vocab.tokens.len()
        )));
    }

    let valid_slice: Option<&[bool]> = match &valid {
        Some(v) => {
            if v.len() != n_rows {
                return Err(PyValueError::new_err(format!(
                    "valid has {} entries, expected {n_rows}",
                    v.len()
                )));
            }
            Some(v.as_slice()?)
        }
        None => None,
    };

    // Validate offsets before entering the GIL-free scan.
    if offs[0] != 0 {
        return Err(PyValueError::new_err("offsets must be relative, starting at 0"));
    }
    for w in offs.windows(2) {
        if w[0] < 0 || w[1] < w[0] || w[1] as usize > vals.len() {
            return Err(PyValueError::new_err(
                "offsets are not ascending inside the value buffer",
            ));
        }
    }

    let g = grid.as_slice_mut()?;
    let t = tables.view();

    py.allow_threads(|| {
        for row in 0..n_rows {
            if let Some(v) = valid_slice {
                if !v[row] {
                    continue;
                }
            }

            let s = offs[row] as usize;
            let e = offs[row + 1] as usize;
            if s == e {
                continue;
            }

            let bytes = &vals[s..e];
            walk_row(bytes, &t, |ts, te, h, low_len| {
                if let Some(code) = vocab.lookup(&bytes[ts..te], h, low_len, &t) {
                    g[code as usize * n_rows + row] = true;
                }
            });
        }
    });

    Ok(())
}

/// Return this tokenizer's lowered tokens for one string.
#[pyfunction]
fn tokens_of(py: Python<'_>, text: &[u8], tables: &ScanTables) -> Vec<Py<pyo3::types::PyBytes>> {
    let t = tables.view();
    let mut out = Vec::new();

    walk_row(text, &t, |ts, te, _h, _l| out.push((ts, te)));

    out.into_iter()
        .map(|(ts, te)| {
            // Rebuild the lowered bytes using the same table as the scanner.
            let mut buf = Vec::with_capacity(te - ts);
            lower_into(&text[ts..te], &t, &mut buf);
            pyo3::types::PyBytes::new(py, &buf).unbind()
        })
        .collect()
}
/// Return the extension version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Expose the scanner hash for testing.
#[pyfunction]
fn hash_of(b: &[u8]) -> u64 {
    hash_bytes(b)
}

#[pymodule]
fn nova_textscan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScanTables>()?;
    m.add_class::<ScanVocab>()?;
    m.add_function(wrap_pyfunction!(scan_into, m)?)?;
    m.add_function(wrap_pyfunction!(tokens_of, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(hash_of, m)?)?;
    Ok(())
}