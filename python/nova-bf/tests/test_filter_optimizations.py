"""Correctness tests for the tokenized text-filter path.

`match_text`/`match_text_from_query` use Qdrant `word`-tokenizer semantics:
split into maximal alphanumeric runs, lowercase each token, and a row matches
a phrase iff every phrase token is one of the row's tokens. The production
implementation answers this via a single Arrow tokenization pass per field
(`_token_row_masks`: `split_pattern_regex` + `utf8_lower` + `index_in`,
byte-bounded row batches scattered from a thread pool into one shared grid).

Coverage here has three layers:
- the fuzz tests check the full Arrow pipeline against an independently
  FORMULATED pure-Python reference (`itertools.groupby` over `isalnum` runs
  — not a copy of any production code). The fuzz alphabet is deliberately
  benign (Latin letters, digits, common punctuation) where Python's and
  Arrow's Unicode tables agree, so the reference is exact there;
- `test_query_and_corpus_tokenize_identically_on_divergent_codepoints` pins
  the property the fuzz alphabet can't reach: on codepoints where Python and
  Arrow DISAGREE (Turkish İ, Greek final sigma), query and corpus sides must
  still agree with each other, because both run the same Arrow kernels;
- the named example tests pin concrete semantics (hyphen splitting, `C++`,
  underscores) with hardcoded expectations.
"""

import random
from itertools import groupby

import numpy as np
import pyarrow as pa
import pytest

from nova_bf.config import Filter, FilterCondition
from nova_bf.filters import _match_text_from_query_mask, _token_row_masks, evaluate
from nova_bf.tokenize import tokenize


def _ref_tokens(text):
    """Independent reference tokenizer — same SEMANTICS (split alphanumeric
    runs, then lowercase), deliberately different formulation and engine
    from both `nova_bf.tokenize` and the Arrow corpus path. Exact on the
    fuzz alphabet below; not valid on codepoints where Python and Arrow
    Unicode tables diverge (covered by the dedicated divergence test)."""
    return {"".join(run).lower() for is_alnum, run in groupby(text, key=str.isalnum) if is_alnum}


def _ref_phrase_mask(texts, phrase):
    """(rows,) reference for one phrase: every phrase token in the row's token
    set; null text, null phrase, or token-less phrase -> False."""
    if not isinstance(phrase, str):
        return np.zeros(len(texts), dtype=bool)
    want = _ref_tokens(phrase)
    if not want:
        return np.zeros(len(texts), dtype=bool)
    return np.array(
        [t is not None and want <= _ref_tokens(t) for t in texts], dtype=bool
    )


def _ref_mask(texts, phrases):
    """(n_queries, rows) reference for `_match_text_from_query_mask`."""
    return np.stack([_ref_phrase_mask(texts, ph) for ph in phrases])


def _cond(field="text", key="kw"):
    return FilterCondition(field=field, match_text_from_query=key)


_ASCII = ["fever", "dna", "gene", "protein", "cell", "x", "a1", "under_score", "COVID", "Fever", "the"]
_UNI = ["café", "naïve", "über", "señor", "straße"]
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
        return rng.choice(["", "   ", "\t", "!!!", "--", "_"])
    return " ".join(rng.choice(_ASCII + _UNI + _PUNCT) for _ in range(rng.randint(1, 3)))


def test_tokenize_examples():
    """The semantics change vs the old \\b-regex approximation, spelled out."""
    assert tokenize("chronic fatigue syndrome") == ["chronic", "fatigue", "syndrome"]
    # hyphenated words split into their parts (old: one literal token)
    assert tokenize("high-fat") == ["high", "fat"]
    # trailing punctuation is stripped (old: `C++` could never match)
    assert tokenize("C++") == ["c"]
    # underscores separate tokens (old: `\w` glued them together)
    assert tokenize("under_score") == ["under", "score"]
    # unicode letters stay inside tokens (old: RE2 ASCII `\b` split at them)
    assert tokenize("Café société") == ["café", "société"]
    assert tokenize("!!! --") == []


def test_query_and_corpus_tokenize_identically_on_divergent_codepoints():
    """Query strings and corpus text run the SAME Arrow kernels, so they
    agree even on the codepoints where Python's str.lower/str.isalnum and
    Arrow's utf8_lower/RE2 classes disagree (Turkish İ's case mapping, Greek
    word-final Σ→σ vs ς) — the failure class an earlier draft had, where a
    Python-tokenized query token could never equal any Arrow-tokenized
    corpus token and silently matched nothing."""
    rows = ["İstanbul kebap", "ΟΔΥΣΣΕΥΣ ΗΡΩΑΣ", "plain row"]
    t = pa.table({"text": pa.array(rows)})
    for phrase, expect in [
        ("İstanbul", [True, False, False]),   # identical word must match itself
        ("ΟΔΥΣΣΕΥΣ", [False, True, False]),   # uppercase final-sigma word
        ("οδυσσευσ", [False, True, False]),   # arrow-lowercased spelling (σ)
    ]:
        f = Filter(must=[FilterCondition(field="text", match_text=phrase)])
        assert evaluate(f, t).tolist() == expect, phrase
        got = _match_text_from_query_mask(_cond(), t, {"kw": np.array([phrase], dtype=object)})
        assert got[0].tolist() == expect, phrase


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_match_text_from_query_vs_reference(seed):
    rng = random.Random(seed)
    nrows = rng.randint(1, 60)
    texts = [None if rng.random() < 0.1 else _rand_text(rng) for _ in range(nrows)]
    nq = rng.randint(1, 12)
    phrases = np.array([_rand_phrase(rng) for _ in range(nq)], dtype=object)

    for typ in (pa.string(), pa.large_string()):
        t = pa.table({"text": pa.array(texts, type=typ)})
        got = _match_text_from_query_mask(_cond(), t, {"kw": phrases})
        ref = _ref_mask(texts, phrases)
        assert got.shape == ref.shape
        assert np.array_equal(got, ref), (
            f"seed={seed} type={typ}\ntexts={texts}\nphrases={phrases.tolist()}\n"
            f"got={got.tolist()}\nref={ref.tolist()}"
        )


@pytest.mark.parametrize("seed", range(30))
def test_fuzz_evaluate_must_should_vs_reference(seed):
    """Mirror the real config: must = text keyword-AND, should = OR of url
    slots — several text conditions, two on the same FIELD, so this also
    exercises the shared per-field tokenization pass (`_text_token_masks`)."""
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
    ref = _ref_mask(texts, kw) & (_ref_mask(urls, d1) | _ref_mask(urls, d2))
    assert np.array_equal(got, ref), f"seed={seed}"


def test_static_match_text_same_tokenized_semantics():
    """Static `match_text` goes through the same tokenizer: hyphen/punct/case
    variants of the query all match the same rows."""
    t = pa.table({"text": pa.array(["high-fat diet", "a HIGH fat meal", "low-fat", "fat", None])})
    for query in ("high-fat", "high fat", "HIGH FAT!", "fat...high"):
        f = Filter(must=[FilterCondition(field="text", match_text=query)])
        got = evaluate(f, t)
        assert got.tolist() == [True, True, False, False, False], query


def test_unicode_and_underscore_now_match_qdrant_tokenizer():
    """The cases the old \\b-regex approximation got wrong vs Qdrant."""
    t = pa.table({"text": pa.array(["walk dayÉ home", "com_content page", "café society"])})
    qv = {"kw": np.array(["day", "content", "café"], dtype=object)}
    got = _match_text_from_query_mask(_cond(), t, qv)
    # `day` must NOT match inside the token `dayé` (old RE2 ASCII \b did);
    # `content` MUST match inside `com_content` (old \w glued the underscore);
    # `café` matches its own token (old path fell back to a regex whose \b
    # behavior at accented letters was engine-dependent).
    assert got.tolist() == [
        [False, False, False],
        [False, True, False],
        [False, False, True],
    ]


def test_token_row_masks_multi_batch_and_threads():
    """A column big enough to split into several batches (the adaptive batch
    size floors at 4096 rows) must produce the same masks as the reference —
    batch offset arithmetic and the concurrent scatter into the shared grid
    must not change results."""
    base = ["fever x", "y dna", None, "fever dna", "gene", "high-fat"]
    texts = base * 2000  # 12,000 rows → multiple 4096-row batches
    col = pa.chunked_array([pa.array(texts)])
    got = _token_row_masks(col, {"fever", "dna", "gene", "fat"}, len(col))
    for tok in ("fever", "dna", "gene", "fat"):
        base_ref = [t is not None and tok in _ref_tokens(t) for t in base]
        ref = np.array(base_ref * 2000, dtype=bool)
        assert np.array_equal(got[tok], ref), tok


def test_large_string_column_smoke():
    """`string` columns are cast to large_string up front (32-bit offset
    overflow protection on multi-GB text columns) — value parity is covered
    by the fuzz tests; this pins the plumbing."""
    t = pa.table({"text": pa.array(["chronic fatigue", "acute onset", None], type=pa.string())})
    got = _match_text_from_query_mask(_cond(), t, {"kw": np.array(["chronic", "acute"], dtype=object)})
    assert got.tolist() == [[True, False, False], [False, True, False]]
