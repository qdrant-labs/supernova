r"""Correctness tests for the filter optimizations:

  B  — single tokenization pass for plain-ASCII ``\w+`` words
       (`_simple_word_masks`) instead of one full-column scan per word,
  D  — lowercase the text column once (via ``utf8_lower``) rather than per word,
  large_string — cast ``string`` text columns to ``large_string`` so the
       regex-verify ``.take()`` never overflows 32-bit offsets.

Every test asserts the optimized `_match_text_from_query_mask` / `evaluate`
output is BIT-IDENTICAL to a reference that uses the original per-word
``\bword\b`` regex with ``ignore_case=True`` — the semantics the run is
parity-validated against. If any optimization ever diverged (e.g. a Unicode
word-boundary edge), these would catch it.
"""

import random
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import _match_text_from_query_mask, _simple_word_masks, evaluate


def _ref_word(col, w):
    """Original single-word semantics: ``\bword\b`` regex, case-insensitive,
    nulls -> False."""
    return pc.fill_null(
        pc.match_substring_regex(col, pattern=rf"\b{re.escape(w)}\b", ignore_case=True), False
    ).to_numpy(zero_copy_only=False)


def _ref_mask(col, phrases):
    """Reference for `_match_text_from_query_mask`: AND of per-word regex masks,
    blank/None phrase -> all False."""
    n = len(col)
    res = np.zeros((len(phrases), n), dtype=bool)
    for q, ph in enumerate(phrases):
        if ph is None or (isinstance(ph, float) and ph != ph):
            continue
        if not str(ph).strip():
            continue
        m = None
        for w in str(ph).split():
            part = _ref_word(col, w)
            m = part if m is None else (m & part)
        res[q] = m
    return res


def _cond(field="text", key="kw"):
    return FilterCondition(field=field, match_text_from_query=key)


_ASCII = ["fever", "dna", "gene", "protein", "cell", "x", "a1", "under_score", "COVID", "Fever", "the"]
_UNI = ["café", "naïve", "über", "señor", "Straße"]
_PUNCT = ["high-fat", "c++", "u.s.a", "covid-19", "e-mail"]
_SEP = [" ", "  ", "\t", ", ", ". ", "-", "/", "\n", "_", "!", "—"]


def _rand_text(rng):
    parts = [rng.choice(_ASCII + _UNI + _PUNCT + ["Mouse", "KEYBOARD"]) for _ in range(rng.randint(0, 8))]
    s = ""
    for i, p in enumerate(parts):
        s += p
        if i < len(parts) - 1:
            s += rng.choice(_SEP)
    return s


def _rand_phrase(rng):
    r = rng.random()
    if r < 0.08:
        return None
    if r < 0.14:
        return rng.choice(["", "   ", "\t"])
    return " ".join(rng.choice(_ASCII + _UNI + _PUNCT) for _ in range(rng.randint(1, 3)))


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_match_text_from_query_bit_identical(seed):
    rng = random.Random(seed)
    nrows = rng.randint(1, 60)
    texts = [None if rng.random() < 0.1 else _rand_text(rng) for _ in range(nrows)]
    nq = rng.randint(1, 12)
    phrases = np.array([_rand_phrase(rng) for _ in range(nq)], dtype=object)

    for typ in (pa.string(), pa.large_string()):
        t = pa.table({"text": pa.array(texts, type=typ)})
        got = _match_text_from_query_mask(_cond(), t, {"kw": phrases})
        ref = _ref_mask(t["text"], phrases)
        assert got.shape == ref.shape
        assert np.array_equal(got, ref), (
            f"seed={seed} type={typ}\ntexts={texts}\nphrases={phrases.tolist()}\n"
            f"got={got.tolist()}\nref={ref.tolist()}"
        )


@pytest.mark.parametrize("seed", range(30))
def test_fuzz_evaluate_must_should_bit_identical(seed):
    """Mirror the real config: must = text keyword-AND, should = OR of url slots."""
    rng = random.Random(seed)
    nrows = rng.randint(1, 40)
    texts = [None if rng.random() < 0.1 else _rand_text(rng) for _ in range(nrows)]
    urls = [rng.choice(["nih.gov docs", "webmd health", "arxiv paper", "blog", "mayoclinic info"]) for _ in range(nrows)]
    nq = rng.randint(1, 8)
    kw = np.array([_rand_phrase(rng) for _ in range(nq)], dtype=object)
    d1 = np.array([rng.choice(["nih", "webmd", "arxiv", "mayoclinic", "zzznone"]) for _ in range(nq)], dtype=object)
    d2 = np.array([rng.choice(["gov", "health", "paper", "zzznone"]) for _ in range(nq)], dtype=object)

    t = pa.table({"text": pa.array(texts), "url": pa.array(urls)})
    f = Filter(
        must=[FilterCondition(field="text", match_text_from_query="kw")],
        should=[
            FilterCondition(field="url", match_text_from_query="d1"),
            FilterCondition(field="url", match_text_from_query="d2"),
        ],
    )
    got = evaluate(f, t, {"kw": kw, "d1": d1, "d2": d2})
    ref = _ref_mask(t["text"], kw) & (_ref_mask(t["url"], d1) | _ref_mask(t["url"], d2))
    assert np.array_equal(got, ref), f"seed={seed}"


def test_punctuated_and_unicode_take_regex_path():
    """Words the tokenizer can't answer (punctuation, non-ASCII) must still
    match the exact regex semantics."""
    t = pa.table({"text": pa.array(["high-fat diet", "high fat", "low-fat", "café society", "cafe bar"])})
    qv = {"kw": np.array(["high-fat", "café"], dtype=object)}
    got = _match_text_from_query_mask(_cond(), t, qv)
    assert np.array_equal(got, _ref_mask(t["text"], qv["kw"]))


def test_simple_word_masks_across_batches():
    """Row-batching (batch offset accumulation) must not change results."""
    col = pa.chunked_array([pa.array(["fever x", "y dna", None, "fever dna", "gene"] * 40)])
    got = _simple_word_masks(col, {"fever", "dna", "gene"}, len(col), batch_rows=7)
    for w in ("fever", "dna", "gene"):
        assert np.array_equal(got[w], _ref_word(col, w)), w


def test_large_string_over_2gb_offset_path_smoke():
    """A string column that would overflow int32 offsets on concat is handled
    (the cast to large_string). Kept small here — correctness of the cast, not
    the 2GB threshold, is what matters; the fuzz tests cover value parity."""
    t = pa.table({"text": pa.array(["chronic fatigue", "acute onset", None], type=pa.string())})
    got = _match_text_from_query_mask(_cond(), t, {"kw": np.array(["chronic", "acute"], dtype=object)})
    assert got.tolist() == [[True, False, False], [False, True, False]]
