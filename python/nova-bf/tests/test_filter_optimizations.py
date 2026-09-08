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

from nova_bf.config import Filter, FilterCondition, RangeCondition, RangeFromQuery
from nova_bf.filters import _condition_mask, _match_text_from_query_mask, _token_row_masks, evaluate
from nova_bf.tokenize import tokenize


def _ev(filt, table, query_values=None):
    """`evaluate()` with the per-query result expanded.

    A filter with any per-query condition returns a `filters.PackedRowMask`
    (row-bit-packed) rather than an `(n_queries, rows)` bool array — the
    production `filtered_text` mask is 10.8 GB per file unpacked, so the packed
    form is the real one and `.unpack()` is the debug view. These tests assert
    on cell values at fixture sizes, so they expand. A uniform filter still
    returns a plain `(rows,)` array and passes straight through.
    """
    mask = evaluate(filt, table, query_values)
    return mask.unpack() if hasattr(mask, "unpack") else mask




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
        assert _ev(f, t).tolist() == expect, phrase
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
    exercises the shared per-field tokenization pass (`_text_prep`)."""
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
    got = _ev(f, t, {"kw": kw, "d1": d1, "d2": d2})
    ref = _ref_mask(texts, kw) & (_ref_mask(urls, d1) | _ref_mask(urls, d2))
    assert np.array_equal(got, ref), f"seed={seed}"


def _condition_major_evaluate(filt, table, qv=None):
    """The pre-fusion combine: every condition expands to its own mask
    (per-query text conditions each materialize (n_queries, rows)), then
    groups AND/OR the full arrays — the reference the fused, query-major
    `_ev()` must match bit-for-bit."""
    n = len(table)
    keep = np.ones(n, dtype=bool)
    for cond in filt.must:
        keep = keep & _condition_mask(cond, table, qv)
    if filt.should:
        any_m = np.zeros(n, dtype=bool)
        for cond in filt.should:
            any_m = any_m | _condition_mask(cond, table, qv)
        keep = keep & any_m
    for cond in filt.must_not:
        keep = keep & ~_condition_mask(cond, table, qv)
    return keep


@pytest.mark.parametrize("seed", range(50))
def test_fuzz_fused_evaluate_bit_identical_to_condition_major(seed):
    """Random filters mixing per-query text conditions with static text,
    static match/range, and non-text per-query conditions across all three
    groups: the fused path must return exactly what condition-major
    combination returns — same bits, same ndim."""
    rng = random.Random(1000 + seed)
    nrows = rng.randint(1, 50)
    texts = [None if rng.random() < 0.1 else _rand_text(rng) for _ in range(nrows)]
    urls = [rng.choice(["nih.gov docs", "webmd health", "arxiv paper", "blog"]) for _ in range(nrows)]
    cats = [rng.choice(["a", "b", "c", None]) for _ in range(nrows)]
    costs = [None if rng.random() < 0.15 else round(rng.uniform(0, 10), 2) for _ in range(nrows)]
    t = pa.table({
        "text": pa.array(texts), "url": pa.array(urls),
        "category": pa.array(cats), "cost": pa.array(costs, type=pa.float64()),
    })

    nq = rng.randint(1, 10)
    qv = {
        "kw": np.array([_rand_phrase(rng) for _ in range(nq)], dtype=object),
        "d1": np.array([rng.choice(["nih", "webmd", "arxiv", "zzznone", None]) for _ in range(nq)], dtype=object),
        "d2": np.array([rng.choice(["gov", "health", "paper", ""]) for _ in range(nq)], dtype=object),
        "cat_q": np.array([rng.choice(["a", "b", "zz", None]) for _ in range(nq)], dtype=object),
        "budget": np.array([rng.choice([rng.uniform(0, 10), np.nan]) for _ in range(nq)]),
    }

    def rand_cond():
        kind = rng.randrange(7)
        if kind == 0:
            return FilterCondition(field="text", match_text_from_query="kw")
        if kind == 1:
            return FilterCondition(field="url", match_text_from_query=rng.choice(["d1", "d2"]))
        if kind == 2:
            return FilterCondition(field="text", match_text="fever dna")
        if kind == 3:
            return FilterCondition(field="category", match=rng.choice(["a", "b"]))
        if kind == 4:
            return FilterCondition(field="cost", range=RangeCondition(lt=rng.uniform(2, 9)))
        if kind == 5:
            return FilterCondition(field="category", match_from_query="cat_q")
        return FilterCondition(field="cost", range_from_query=RangeFromQuery(lt="budget"))

    groups = {"must": [], "should": [], "must_not": []}
    for g in groups:
        for _ in range(rng.randrange(3)):
            groups[g].append(rand_cond())
    if not any(groups.values()):
        groups["must"].append(rand_cond())
    filt = Filter(**{g: tuple(cs) for g, cs in groups.items()})

    got = _ev(filt, t, qv)
    ref = _condition_major_evaluate(filt, t, qv)
    assert got.ndim == ref.ndim and got.shape == ref.shape, f"seed={seed} {got.shape} vs {ref.shape}"
    assert np.array_equal(got, ref), (
        f"seed={seed}\nfilter={filt}\nqv={ {k: v.tolist() for k, v in qv.items()} }\n"
        f"got={got.tolist()}\nref={ref.tolist()}"
    )


def test_static_match_text_same_tokenized_semantics():
    """Static `match_text` goes through the same tokenizer: hyphen/punct/case
    variants of the query all match the same rows."""
    t = pa.table({"text": pa.array(["high-fat diet", "a HIGH fat meal", "low-fat", "fat", None])})
    for query in ("high-fat", "high fat", "HIGH FAT!", "fat...high"):
        f = Filter(must=[FilterCondition(field="text", match_text=query)])
        got = _ev(f, t)
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
        assert np.array_equal(got.mask(tok), ref), tok


def test_large_string_column_smoke():
    """`string` columns are cast to large_string up front (32-bit offset
    overflow protection on multi-GB text columns) — value parity is covered
    by the fuzz tests; this pins the plumbing."""
    t = pa.table({"text": pa.array(["chronic fatigue", "acute onset", None], type=pa.string())})
    got = _match_text_from_query_mask(_cond(), t, {"kw": np.array(["chronic", "acute"], dtype=object)})
    assert got.tolist() == [[True, False, False], [False, True, False]]


# ==========================================================================
# R4b: the shared scan pool
# ==========================================================================
def _multi_batch_column(n_rows=40_000, seed=3):
    """A text column big enough that `_token_row_masks` splits it into
    several batches — otherwise the pool is never exercised and a test on it
    proves nothing."""
    rng = np.random.default_rng(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
             "theta", "iota", "kappa"]
    rows = []
    for _ in range(n_rows):
        k = int(rng.integers(3, 12))
        rows.append(" ".join(rng.choice(vocab, size=k)))
    # a few nulls, which take the "null row splits to a null token list" path
    rows[7] = None
    rows[n_rows // 2] = None
    return pa.chunked_array([pa.array(rows, type=pa.large_string())])


def test_shared_scan_pool_gives_identical_masks():
    """R4b moves tokenisation onto one process-wide pool so filter throughput
    stops being a function of `io_workers`. It must be a pure scheduling
    change: the masks are a scatter into disjoint column ranges of one grid,
    so WHICH thread runs a batch cannot matter — this pins that it does not.

    Run against the same column three ways: no pool (serial/private), a
    private pool (today's default), and a shared pool of a deliberately
    awkward width that does not divide the batch count.
    """
    from concurrent.futures import ThreadPoolExecutor

    from nova_bf.filters import _token_row_masks

    col = _multi_batch_column()
    tokens = {"alpha", "delta", "kappa", "absent"}
    n = len(col)

    ref = _token_row_masks(col, tokens, n, None)
    # The premise: this really did split into more than one batch. `nbytes > 0`
    # used to stand in for that and proved nothing — ask the sizer directly.
    from nova_bf.filters import _scan_batch_rows

    _bs = _scan_batch_rows(col.nbytes, n, 1)
    assert _bs < n, (
        f"batch of {_bs} rows over {n} rows is a single batch; this test is "
        f"not exercising the split it claims to")
    for width in (1, 3, 8, 32):
        with ThreadPoolExecutor(max_workers=width) as pool:
            got = _token_row_masks(col, tokens, n, pool)
        assert set(got) == set(ref)
        for t in ref:
            np.testing.assert_array_equal(got[t], ref[t], err_msg=f"{t} @ {width}")
    # and the masks are not vacuous
    assert ref.mask("alpha").any() and not ref.mask("absent").any()


def test_shared_pool_batch_sizing_tracks_the_pool_not_the_machine():
    """`batch_rows` is derived from whatever will RUN the batches.

    With a shared pool the machine's core count is no longer that number, and
    sizing against it would hand a 2-thread pool 64 tiny batches. The byte cap
    is orthogonal and must still bind on a wide pool with fat rows.
    """
    from nova_bf.filters import _BATCH_TEXT_BYTES, _scan_batch_rows

    n_rows = 1_000_000
    nbytes = 50 * n_rows                       # 50 B/row: the byte cap is loose here
    narrow = _scan_batch_rows(nbytes, n_rows, 2)
    wide = _scan_batch_rows(nbytes, n_rows, 32)
    assert narrow > wide, (narrow, wide)
    # R7: every answer owns whole bytes of the packed grid.
    assert narrow % 8 == 0 and wide % 8 == 0
    assert narrow == (n_rows // 4) & ~7         # ~2 batches per thread
    assert wide == (n_rows // 64) & ~7

    # Fat rows: the BYTE cap binds instead, and the pool width stops mattering.
    #
    # The cap really does bind now. This used to assert
    # `max(4096, (_BATCH_TEXT_BYTES // fat) & ~7)`, i.e. the anti-tiny-batch
    # floor overriding the byte cap — which contradicted this test's own
    # docstring and let a batch allocate far past the documented bound. The
    # floor is a preference on the PARALLELISM term only; a memory bound wins.
    fat = 4 << 20                              # 4 MiB/row
    capped = max(8, (_BATCH_TEXT_BYTES // fat) & ~7)
    assert _scan_batch_rows(fat * n_rows, n_rows, 2) == \
        _scan_batch_rows(fat * n_rows, n_rows, 32) == capped
    assert capped < 4096, "this case is meant to show the cap beating the floor"

    # And the 4096-row floor survives a pool so wide it would ask for less.
    assert _scan_batch_rows(100 * 10_000, 10_000, 4096) == 4096

    # R7's third cap: the per-batch BOOL sub-grid. A big vocabulary shrinks
    # the batch even when text bytes and pool width would both allow more.
    from nova_bf.filters import _SUBGRID_BYTES
    many = _scan_batch_rows(nbytes, n_rows, 2, n_tokens=20_000)
    assert many < narrow
    # Same correction as the byte cap above: the token bound binds, and the
    # 4096 floor does not get to override it. 832 rows x 20k tokens is a
    # 16.6 MB sub_grid; the old `max(4096, ...)` asked for 82 MB, per thread.
    assert many == max(8, (_SUBGRID_BYTES // 20_000) & ~7)
    assert many * 20_000 <= _SUBGRID_BYTES


def test_a_failing_reader_does_not_leak_the_shared_scan_pool(tmp_path):
    """The shared `bf-scan` pool lives for the whole scan, so `run_compute` owns
    it in a `try/finally`. Without that, a reader thread that raises (here: a
    filter on a column the corpus does not have) propagates out of `run_compute`
    past the `shutdown()` and leaves `cpu_thread_count` idle threads alive for
    the life of the process — every subsequent run stacking another pool on top.
    """
    import threading

    import pyarrow.parquet as pq
    import torch  # noqa: F401  — run_compute needs it

    from nova_bf.compute import run_compute
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig,
        QueriesConfig, SearchSpec,
    )

    def _live_scan_threads():
        return [t for t in threading.enumerate()
                if t.is_alive() and t.name.startswith("bf-scan")]

    assert not _live_scan_threads(), "a previous test already leaked one"

    # File 0 is big enough that the text scan splits into several batches and
    # therefore actually SUBMITS to the shared pool (a one-batch scan never
    # spawns a thread, and the leak would be invisible). File 1 has no `text`
    # column at all, so `evaluate()` raises in the reader after the pool's
    # threads exist.
    rng = np.random.default_rng(0)
    cdir = tmp_path / "c"
    cdir.mkdir()
    n0 = 12_000
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((n0, 3)).astype(np.float32).tolist(),
            pa.list_(pa.float32())),
        "id": pa.array([f"a{r}" for r in range(n0)]),
        "text": pa.array([f"doc {r}" for r in range(n0)]),
    }), str(cdir / "f0.parquet"))
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((4, 3)).astype(np.float32).tolist(),
            pa.list_(pa.float32())),
        "id": pa.array([f"b{r}" for r in range(4)]),
    }), str(cdir / "f1.parquet"))
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((2, 3)).astype(np.float32).tolist(),
            pa.list_(pa.float32())),
        "qid": pa.array(["q0", "q1"]),
    }), str(qpath))
    out = tmp_path / "out"
    out.mkdir()

    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, cpu_thread_count=3),
        searches=[SearchSpec(
            name="bad", metric="dot", k=2,
            # a `match_text` leaf keeps this off the GPU path, so it reaches
            # `evaluate()` — fine on file 0, absent column on file 1.
            filter=Filter(must=[FilterCondition(field="text", match_text="doc")]),
        )],
    )
    with pytest.raises(RuntimeError, match="reader thread failed"):
        run_compute(cfg)

    assert not _live_scan_threads(), (
        "run_compute left its shared scan pool running after a reader failure")
