//! Allocation-free tokenizer used by the native `match_text` scan.
//!
//! Text is split into maximal runs of `\p{L}\p{N}` codepoints and each token
//! is then lowercased. Token classification and lowercase mappings come from
//! tables derived by Python from the installed Arrow build, so this module does
//! not carry its own Unicode semantics.
//!
//! Token boundaries are determined from the original codepoints before
//! lowercasing. The native path is enabled only when Arrow's lowercase mapping
//! can be represented one codepoint at a time.
//!
//! Malformed UTF-8 is treated conservatively as a separator and always advances
//! by at least one byte, keeping the walk total and in bounds.


/// Arrow-derived Unicode tables plus ASCII fast-path representations.
pub struct Tables<'a> {
    /// `\p{L}\p{N}` membership for every Unicode codepoint.
    pub alnum: &'a [u8],

    /// Arrow's one-codepoint lowercase mapping.
    pub lower: &'a [u32],

    /// ASCII alphanumeric membership as a bitmap over all byte values.
    pub ascii_alnum: [u64; 4],

    /// Lowercase mappings for ASCII codepoints.
    pub ascii_lower: [u8; 128],
}
impl<'a> Tables<'a> {
    pub fn new(alnum: &'a [u8], lower: &'a [u32]) -> Self {
        assert_eq!(alnum.len(), 0x11_0000);
        assert_eq!(lower.len(), 0x11_0000);

        let mut ascii_alnum = [0u64; 4];
        let mut ascii_lower = [0u8; 128];

        for b in 0..128usize {
            if alnum[b] != 0 {
                ascii_alnum[b >> 6] |= 1u64 << (b & 63);
            }

            debug_assert!(lower[b] < 0x80);
            ascii_lower[b] = lower[b] as u8;
        }

        Self {
            alnum,
            lower,
            ascii_alnum,
            ascii_lower,
        }
    }

    #[inline(always)]
    fn ascii_is_alnum(&self, b: u8) -> bool {
        (self.ascii_alnum[(b >> 6) as usize] >> (b & 63)) & 1 != 0
    }

    #[inline(always)]
    fn ascii_low(&self, b: u8) -> u8 {
        debug_assert!(b < 0x80);
        self.ascii_lower[(b & 0x7F) as usize]
    }

    #[inline(always)]
    fn is_alnum(&self, cp: u32) -> bool {
        debug_assert!((cp as usize) < self.alnum.len());
        unsafe { *self.alnum.get_unchecked(cp as usize) != 0 }
    }

    #[inline(always)]
    pub fn lower_cp(&self, cp: u32) -> u32 {
        debug_assert!((cp as usize) < self.lower.len());
        unsafe { *self.lower.get_unchecked(cp as usize) }
    }
}
/// Decode one UTF-8 codepoint at `b[i]`.
///
/// Malformed input becomes `(0, 1)`, treating the byte as a separator while
/// guaranteeing forward progress.
#[inline]
fn decode(b: &[u8], i: usize) -> (u32, usize) {
    let c0 = b[i];

    if c0 < 0x80 {
        return (c0 as u32, 1);
    }

    let n = match c0 {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return (0, 1),
    };

    if i + n > b.len() {
        return (0, 1);
    }

    let mut cp = (c0 as u32) & (0x7F >> n);
    for k in 1..n {
        let cc = b[i + k];
        if cc & 0xC0 != 0x80 {
            return (0, 1);
        }
        cp = (cp << 6) | ((cc as u32) & 0x3F);
    }

    let min = match n {
        2 => 0x80,
        3 => 0x800,
        _ => 0x10000,
    };

    if cp < min || cp > 0x10FFFF || (0xD800..0xE000).contains(&cp) {
        return (0, 1);
    }

    (cp, n)
}
/// UTF-8-encode a valid Unicode scalar value into `buf`.
#[inline(always)]
pub fn encode(cp: u32, buf: &mut [u8; 4]) -> usize {
    debug_assert!(cp <= 0x10FFFF);
    debug_assert!(!(0xD800..0xE000).contains(&cp));

    if cp < 0x80 {
        buf[0] = cp as u8;
        1
    } else if cp < 0x800 {
        buf[0] = 0xC0 | (cp >> 6) as u8;
        buf[1] = 0x80 | (cp & 0x3F) as u8;
        2
    } else if cp < 0x10000 {
        buf[0] = 0xE0 | (cp >> 12) as u8;
        buf[1] = 0x80 | ((cp >> 6) & 0x3F) as u8;
        buf[2] = 0x80 | (cp & 0x3F) as u8;
        3
    } else {
        buf[0] = 0xF0 | (cp >> 18) as u8;
        buf[1] = 0x80 | ((cp >> 12) & 0x3F) as u8;
        buf[2] = 0x80 | ((cp >> 6) & 0x3F) as u8;
        buf[3] = 0x80 | (cp & 0x3F) as u8;
        4
    }
}
// Hash used only for candidate lookup; matches are confirmed by `token_eq`.
//
// Both `Hasher` and `hash_bytes` fold little-endian 8-byte groups and then
// fold the byte length, so they produce the same hash for the same byte stream.

pub const HASH_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline(always)]
fn fold(h: u64, w: u64) -> u64 {
    let x = (h ^ w).wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^ (x >> 29)
}

pub fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = HASH_SEED;
    let mut i = 0;

    while i + 8 <= b.len() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&b[i..i + 8]);
        h = fold(h, u64::from_le_bytes(w));
        i += 8;
    }

    let mut last = 0u64;
    for (k, &x) in b[i..].iter().enumerate() {
        last |= (x as u64) << (8 * k);
    }

    fold(fold(h, last), b.len() as u64)
}
/// Streaming equivalent of `hash_bytes`.
struct Hasher {
    h: u64,
    word: u64,
    k: u32,
    len: usize,
}

impl Hasher {
    #[inline(always)]
    fn new() -> Self {
        Self {
            h: HASH_SEED,
            word: 0,
            k: 0,
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, b: u8) {
        self.word |= (b as u64) << (8 * self.k);
        self.len += 1;
        self.k += 1;

        if self.k == 8 {
            self.h = fold(self.h, self.word);
            self.word = 0;
            self.k = 0;
        }
    }

    #[inline(always)]
    fn finish(&self) -> u64 {
        fold(fold(self.h, self.word), self.len as u64)
    }
}

/// Does the lowered form of `tok` equal `needle`, byte for byte?
///
/// This is what makes the hash a hint and never a decision, and it is also
/// what makes an 8-byte vocabulary entry safe against a longer token that
/// starts with the same eight bytes: the comparison runs to the end of BOTH.
pub fn token_eq(tok: &[u8], needle: &[u8], t: &Tables) -> bool {
    let mut i = 0;
    let mut j = 0;
    let mut buf = [0u8; 4];
    while i < tok.len() {
        let (cp, len) = decode(tok, i);
        let n = encode(t.lower_cp(cp), &mut buf);
        if j + n > needle.len() || needle[j..j + n] != buf[..n] {
            return false;
        }
        i += len;
        j += n;
    }
    j == needle.len()
}
/// Walk one row and call `hit(start, end, hash, lowered_len)` for each token.
///
/// `start` and `end` are byte offsets into the original row. The hash and
/// length are computed from the lowered token without materializing it.
#[inline]
pub fn walk_row<F: FnMut(usize, usize, u64, usize)>(
    row: &[u8],
    t: &Tables,
    mut hit: F,
) {
    let n = row.len();
    let mut i = 0;

    while i < n {
        // Skip separators.
        let b = row[i];
        if b < 0x80 {
            if !t.ascii_is_alnum(b) {
                i += 1;
                continue;
            }
        } else {
            let (cp, len) = decode(row, i);
            if !t.is_alnum(cp) {
                i += len;
                continue;
            }
        }

        let start = i;
        let mut hs = Hasher::new();

        loop {
            // Consume the ASCII portion of the token.
            while i < n {
                let b = row[i];
                if b >= 0x80 || !t.ascii_is_alnum(b) {
                    break;
                }

                hs.push(t.ascii_low(b));
                i += 1;
            }

            if i >= n || row[i] < 0x80 {
                break;
            }

            let (cp, len) = decode(row, i);
            if !t.is_alnum(cp) {
                break;
            }

            let mut buf = [0u8; 4];
            let m = encode(t.lower_cp(cp), &mut buf);
            for &x in &buf[..m] {
                hs.push(x);
            }

            i += len;
        }

        hit(start, i, hs.finish(), hs.len);
    }
}

// --- test aids ---------------------------------------------------------------
//
// `tokens_of` in `lib.rs` rebuilds a token's lowered bytes so a Python test
// can look at them. Never on the scan path.

pub fn lower_into(tok: &[u8], t: &Tables, out: &mut Vec<u8>) {
    let mut i = 0;
    let mut buf = [0u8; 4];
    while i < tok.len() {
        let (cp, len) = decode(tok, i);
        let n = encode(t.lower_cp(cp), &mut buf);
        out.extend_from_slice(&buf[..n]);
        i += len;
    }
}
