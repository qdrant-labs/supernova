"""Equivalence tests for `nova_bf.nativetok` / the `nova_textscan` crate.

ONE oracle: the exact
`_token_row_masks._arrow_scan` body (`split_pattern_regex` -> `utf8_lower` ->
`list_parent_indices` -> `index_in` -> boolean scatter). The native scan is
only ever allowed to be a performance change, so the whole
`(n_tokens, n_rows)` grid has to come out identical — `np.array_equal`, never
a spot check.

The native scanner has no fallback path, which is what makes these tests
There is nothing to route, so every case here is answered entirely in Rust,
non-ASCII vocabulary and all; any disagreement is a bug.

Skipped wholesale when the extension is not built — a source checkout without
`crates/nova-textscan` compiled is a legitimate state, and
`_token_row_masks` falls back to Arrow there.
"""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pytest

from nova_bf import nativetok

pytestmark = pytest.mark.skipif(
    not nativetok.available(),
    reason=f"nova_textscan unavailable: {nativetok.unavailable_reason()}",
)


def _check(rows, vocab, typ=pa.large_string()):
    """The native grid against Arrow's, for one column and one vocabulary."""
    vocab = sorted(set(vocab))
    arr = pa.array(rows, type=typ)
    v = nativetok.prepare(vocab)
    assert v is not None, "the scanner declined a vocabulary it should serve"
    got = np.zeros((len(vocab), len(arr)), dtype=bool)
    nativetok.scan_into(arr, v, got)
    want = nativetok.arrow_grid(arr, vocab)
    if not np.array_equal(got, want):
        d = np.argwhere(got != want)[0]
        raise AssertionError(
            f"{len(np.argwhere(got != want))} cell(s) differ; first is token "
            f"{vocab[d[0]]!r} row {rows[d[1]]!r}: native={got[d[0], d[1]]} "
            f"arrow={want[d[0], d[1]]}"
        )
    return got


# --------------------------------------------------------------------------
# the tokenizer's own definition
# --------------------------------------------------------------------------

def test_tokens_match_the_arrow_tokenizer():
    """`nova_bf.tokenize` IS the definition; the scanner must reproduce it,
    including its split-then-lower order."""
    from nova_bf.tokenize import tokenize

    for text in [
        "The quick brown Fox", "fox_jumps-over", "", "   ",
        "Café résumé", "İstanbul", "ΣΟΦΟΣ σοφος", "emoji 🙂 here",
        "ǅungla ǄUNGLA ǆungla", "100K 100k", "héllo",
        "a-b_c.d/e:f!g?h", "١٢٣ ०१२ 42",
    ]:
        assert nativetok.tokens_of(text) == tokenize(text), text


def test_the_two_codepoints_that_lower_into_ascii_are_ordinary_here():
    """U+0130 and U+212A are why the numpy byte walk needs a fallback list.
    With a full-width case map they are just letters."""
    _check(["İstanbul", "100K", "K and k", "İ"],
           ["istanbul", "100k", "k", "i"])


def test_a_nonascii_vocabulary_token_is_served_natively():
    """The other reason the numpy walk falls back. Nothing routes here."""
    _check(["Café au lait", "cafe", "CAFÉ", "текст hello мир"],
           ["café", "cafe", "текст", "мир", "hello"])


def test_a_combining_mark_splits_a_word_and_an_accent_does_not():
    # U+0301 is a Mark: neither \p{L} nor \p{N}, so Arrow yields `he` and
    # `llo`. `é` as one codepoint is a letter, so `café` stays whole.
    _check(["héllo", "café"], ["he", "llo", "hello", "café", "caf"])


# --------------------------------------------------------------------------
# the vocabulary lookup, where a hash could hide a bug
# --------------------------------------------------------------------------

def test_eight_byte_entry_versus_a_longer_token_sharing_its_prefix():
    """The classic packed-key trap: an 8-byte entry and a 9-byte corpus token
    agree on all eight bytes, and only the length separates them."""
    _check(["aaaaaaaa", "aaaaaaaaa", "aaaaaaa"],
           ["aaaaaaaa", "aaaaaaaaa"])


def test_same_length_siblings_sharing_eight_bytes():
    _check(["internation internationalisation internationalisatiox"],
           ["internationalisation", "internationalisatiox", "internation"])


def test_vocab_token_is_a_prefix_of_a_corpus_token_and_vice_versa():
    _check(["inter", "internal", "international"],
           ["inter", "internal", "international", "internationale"])


def test_uppercase_vocabulary_token_never_matches():
    """The Arrow pipeline compares the LOWERED corpus token against the value
    set verbatim, so an entry that is not lowercase is unreachable. The
    scanner must reproduce that rather than 'helpfully' lowering it."""
    got = _check(["FOO foo Foo"], ["FOO", "foo"])
    vocab = sorted({"FOO", "foo"})
    assert not got[vocab.index("FOO"), 0]
    assert got[vocab.index("foo"), 0]


def test_a_long_vocabulary_stresses_the_filter_and_the_probe():
    rng = random.Random(11)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(1, 24)))
             for _ in range(4000)]
    rows = [" ".join(rng.choice(words) for _ in range(30)) for _ in range(200)]
    _check(rows, words)


# --------------------------------------------------------------------------
# the shapes an Arrow column can actually arrive in
# --------------------------------------------------------------------------

def test_nulls_and_empties_and_separator_only_rows():
    _check([None, "", "   ", ",,,", "a", None, "b"], ["a", "b"])


def test_all_null_column():
    _check([None, None, None], ["a"])


@pytest.mark.parametrize("typ", [pa.string(), pa.large_string()])
def test_both_string_widths(typ):
    _check(["hello world", None, "HELLO"], ["hello", "world"], typ=typ)


def test_a_sliced_array_reads_from_its_own_offset():
    """`arr.offset != 0` is the case a buffer-level scanner gets wrong
    silently: it reads the neighbours' bytes and answers for the wrong rows."""
    full = pa.array(["zzz", "hello there", "world", "nothing"],
                    type=pa.large_string())
    vocab = ["hello", "there", "world", "zzz"]
    for start in range(len(full)):
        for length in range(1, len(full) - start + 1):
            sub = full.slice(start, length)
            v = nativetok.prepare(vocab)
            got = np.zeros((len(vocab), len(sub)), dtype=bool)
            nativetok.scan_into(sub, v, got)
            assert np.array_equal(got, nativetok.arrow_grid(sub, vocab)), \
                (start, length)


def test_a_multi_chunk_column_is_combined_first():
    col = pa.chunked_array([
        pa.array(["alpha beta"], type=pa.large_string()),
        pa.array([None, "gamma"], type=pa.large_string()),
    ])
    vocab = ["alpha", "beta", "gamma"]
    v = nativetok.prepare(vocab)
    got = np.zeros((len(vocab), col.length()), dtype=bool)
    nativetok.scan_into(col, v, got)
    assert np.array_equal(got, nativetok.arrow_grid(col.combine_chunks(), vocab))


# --------------------------------------------------------------------------
# it refuses rather than guesses
# --------------------------------------------------------------------------

def test_an_empty_vocabulary_token_is_refused():
    """Arrow's split emits `""` for a leading or trailing separator, so `""`
    in the value set genuinely matches such rows and this walk cannot produce
    it. Refusing sends the caller to Arrow; answering would be wrong."""
    assert nativetok.prepare(["", "a"]) is None


def test_the_env_kill_switch_is_honoured(monkeypatch):
    monkeypatch.setenv("NOVA_BF_NO_NATIVE_TOKENIZER", "1")
    assert nativetok.prepare(["a"]) is None
    assert nativetok.unavailable_reason() == "NOVA_BF_NO_NATIVE_TOKENIZER"


def test_a_wrong_sized_grid_raises_rather_than_writing_past_it():
    v = nativetok.prepare(["a", "b"])
    arr = pa.array(["a b"], type=pa.large_string())
    for shape in [(1, 1), (3, 1), (2, 2)]:
        with pytest.raises(ValueError):
            nativetok.scan_into(arr, v, np.zeros(shape, dtype=bool))


# --------------------------------------------------------------------------
# malformed UTF-8: Arrow cannot produce it, the walk must survive it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    b"ab\xffcd",              # not a lead byte
    b"ab\xc0\x80cd",          # overlong 2-byte
    b"ab\xe0\x80\x80cd",      # overlong 3-byte
    b"ab\xed\xa0\x80cd",      # a surrogate
    b"ab\xf4\x90\x80\x80cd",  # past U+10FFFF
    b"ab\xc3",                # truncated at the end of the row
    b"ab\x80cd",              # a lone continuation byte
    b"\xff" * 32,
])
def test_malformed_utf8_is_total_and_splits_rather_than_joins(raw):
    """Arrow will not have produced any of these in a valid `large_string`, so
    there is no oracle to compare against — the requirement is that the walk
    stays in bounds, terminates, and treats the garbage as a SEPARATOR, which
    is the safe direction (it splits a token rather than joining two)."""
    import nova_textscan

    t = nativetok._tables()
    toks = [b.decode("utf-8", "replace") for b in nova_textscan.tokens_of(raw, t)]
    assert all(toks), "no empty token may be emitted"
    if raw.startswith(b"ab"):
        assert "ab" in toks and "cd" in [x for x in toks] or "ab" in toks


def test_malformed_bytes_never_fuse_two_tokens():
    import nova_textscan

    t = nativetok._tables()
    toks = [b.decode("utf-8", "replace")
            for b in nova_textscan.tokens_of(b"ab\xffcd", t)]
    assert toks == ["ab", "cd"]


# --------------------------------------------------------------------------
# the fuzz
# --------------------------------------------------------------------------

_ALPH = (list("abcXYZ019 ,._-/:!?\n\t")
         + ["é", "É", "İ", "ı", "K", "Å", "ǅ", "ǆ", "中", "文", "́",
            "―", "ß", "ẞ", "ﬁ", "🙂", "٠", "०", "\x00", "Σ", "σ", "ς"])


@pytest.mark.parametrize("seed", range(120))
def test_fuzz_against_the_arrow_oracle(seed):
    """Random rows over an alphabet chosen to hit every branch: ASCII case,
    separators, marks, non-ASCII letters, the dotted I, the Kelvin sign,
    astral planes, NUL, and Greek final sigma."""
    rng = random.Random(seed)
    rows = []
    for _ in range(rng.randint(1, 40)):
        if rng.random() < 0.06:
            rows.append(None)
            continue
        rows.append("".join(rng.choice(_ALPH) for _ in range(rng.randint(0, 80))))

    from nova_bf.tokenize import tokenize_many

    seen = set()
    for toks in tokenize_many([r for r in rows if r]):
        seen.update(toks or [])
    vocab = sorted(set(list(seen)[:24]) | {
        "abc", "é", "i̇", "ǆ", "k", "aaaaaaaa", "aaaaaaaaa", "中文", "σ", "ς",
    })
    _check(rows, vocab)


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_with_nonascii_vocabulary_only(seed):
    """The case the numpy walk sends to Arrow wholesale. Here it is served
    natively, so it needs its own fuzz."""
    rng = random.Random(1000 + seed)
    letters = ["é", "ü", "Σ", "σ", "İ", "ı", "中", "文", "ǆ", "ß", "å", "ю"]
    words = ["".join(rng.choice(letters) for _ in range(rng.randint(1, 6)))
             for _ in range(40)]
    rows = [" ".join(rng.choice(words) for _ in range(rng.randint(1, 20)))
            for _ in range(rng.randint(1, 20))]
    from nova_bf.tokenize import tokenize_many
    vocab = sorted({t for toks in tokenize_many(rows) for t in (toks or [])})
    _check(rows, vocab)
