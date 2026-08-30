"""The CPU-fallback per-query filter mask is built at its FILTER's height.

A filter with a `match_text`/`match_text_from_query` leaf can't be evaluated
GPU-natively (torch has no string tensor type), so it takes `filters.
evaluate()`'s CPU path and materializes a real `(n_queries, file_rows)`
boolean mask — bit-packed, but built once per corpus file and held for that
file's whole batch loop. It is the only allocation in a run whose size is
`n_queries x file_rows`.

Its query axis used to be the whole queries FILE. That made it the one cost
that grew when unrelated query sets were unioned into a single file behind
`SearchSpec.rows` selectors, even though the filter's own specs never look at
the foreign rows: on FineWeb-10B (~1.15M rows/shard) a 5,000-query text search
paid 1.33 GiB per in-flight file from its own 10k-row queries file and 14.7 GiB
from a 110k-row union file — an 11x rise for a search whose own row count never
changed.

It is now built over the union of its own specs' `rows` (`run_compute`'s
`filter_rows`), and `spec_qrows[m]` indexes POSITIONS WITHIN that union rather
than file rows.

WHAT THIS FILE PINS
-------------------
1. The height itself — the filter's own union, shrinking with that union and
   not with the queries file.
2. Results are BIT-IDENTICAL to the old full-height behavior, which is forced
   back on via `_union_rows_by_key` so the comparison isolates the mask height
   and nothing else (matrix shapes, and therefore matmul accumulation order,
   stay put). Includes a 40-seed randomized differential over random corpora,
   filter shapes, row subsets and spec counts.
3. The load-bearing identity, on `filters.evaluate` directly:
       evaluate(f, table, {c: v[rows]}) == evaluate(f, table, v)[rows]
   If evaluate had any cross-query coupling — a shared token vocabulary, the
   query-major `combos` grouping, broadcasting, a null/MatchAny convention that
   depended on what OTHER queries asked — this is where it would show.
4. The sharing rules that decide the height: specs sharing a filter VALUE pool
   their rows (`keeps` is keyed by the filter, and is vector_type-agnostic, so
   a dense and a sparse spec sharing one filter share one mask); a spec owning
   every row pins it back to full height.
5. What must NOT be narrowed — GPU-eligible filters (per-query state shared
   across filters by FilterCondition, and their mask is per-BATCH, ~1000x
   smaller) and uniform filters (no query axis at all).
6. Downstream consequences of the corpus-row union getting tighter: exact
   tie-breaking, a corpus file emptied entirely, concurrent evaluation of
   filters with different heights, distributed ranks plus merge, and the
   has_baseline seam where one mask is read through both a compacted and an
   uncompacted batch.
7. The selector's type contract — None / slice / contiguous int64 tensor on the
   requested device — which is what carries the invariant onto a GPU. Front A
   itself remains untested on CUDA hardware.

A NOTE ON THE FIXTURES
----------------------
Several tests assert something about a mechanism only if the fixture actually
exercises it, so they carry explicit guards that fail when the fixture stops
discriminating (mask heights that differ, unions that differ, a file that is
really emptied, a generator that really emits every shape). Those guards are
load-bearing: without them a narrowing test can pass by testing nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import nova_bf.compute as compute_mod
from nova_bf.compute import (
    _local_positions,
    _row_selector,
    _union_rows_by_key,
    run_compute,
)
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.filters import evaluate
from nova_bf.merge import run_merge

DIM, VOCAB, NNZ = 8, 12, 6
# The randomized/distributed/seam fixtures build their own corpora at a
# narrower width; kept separate so changing one cannot silently reshape the
# other's seeded cases.
RAND_DIM = 6
WORDS = ["charlie", "delta", "echo", "foxtrot"]
TENANTS = ["A", "B", "C"]

# Two per-query text filters, distinguished by the queries column they read.
KEYWORD_FILTER = Filter(
    must=[FilterCondition(field="text", match_text_from_query="keyword_phrase")]
)
PHRASE_FILTER = Filter(
    must=[FilterCondition(field="text", match_text_from_query="phrase")]
)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _sparse_row(rng, vocab=VOCAB, nnz=NNZ):
    idx = np.sort(rng.choice(vocab, size=nnz, replace=False))
    return idx.tolist(), rng.standard_normal(nnz).astype(np.float32).tolist()


def _write(path, dense, sparse_rows=None, **columns):
    data = {
        "dense_embedding": pa.array(
            np.asarray(dense).tolist(), type=pa.list_(pa.float32())
        )
    }
    if sparse_rows is not None:
        data["sparse_embedding"] = pa.array(
            [{"indices": i, "values": v} for i, v in sparse_rows],
            type=pa.struct([
                pa.field("indices", pa.list_(pa.uint32())),
                pa.field("values", pa.list_(pa.float32())),
            ]),
        )
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


def _rows_of(path):
    t = pq.read_table(path).to_pydict()
    return {q: (h, s) for q, h, s in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


def _force_full_height(monkeypatch):
    """Restore the pre-change behavior for the FILTER-keyed call only, leaving
    the vector_type call (string keys) alone so matrix shapes don't move."""
    real = compute_mod._union_rows_by_key
    monkeypatch.setattr(
        compute_mod, "_union_rows_by_key",
        lambda keys, sr, n: (
            real(keys, sr, n) if all(isinstance(k, str) for k in keys)
            else {k: None for k in real(keys, sr, n)}
        ),
    )


def _spy_union_widths(monkeypatch):
    """Record how many corpus rows each file's `_union_keep` admitted, so a
    test can PROVE the batch composition it claims to vary actually varied.
    Without this a narrowing test can pass by simply not exercising anything."""
    widths: list[int] = []
    real = compute_mod._union_keep

    def spy(fs, keeps):
        u = real(fs, keeps)
        widths.append(int(u.sum()))
        return u

    monkeypatch.setattr(compute_mod, "_union_keep", spy)
    return widths


@pytest.fixture
def mask_spy(monkeypatch):
    """Record the shape of every mask that reaches `_pack_query_axis` — i.e.
    every CPU-fallback per-query mask the run actually materializes."""
    seen: list[tuple[int, int]] = []
    real = compute_mod._pack_query_axis

    def spy(mask):
        seen.append(mask.shape)
        return real(mask)

    monkeypatch.setattr(compute_mod, "_pack_query_axis", spy)
    return seen


@pytest.fixture
def full_height(monkeypatch):
    """`_force_full_height` as a fixture, for tests whose whole run is the
    control arm."""
    _force_full_height(monkeypatch)



@pytest.fixture
def text_ds(tmp_path):
    """One corpus file, and a queries file where only SOME rows belong to the
    text search.

    The owned rows are deliberately not a prefix and not contiguous: a mask
    indexed by file row instead of by position-within-the-filter's-union would
    run off the end (or silently read a neighbour's row) rather than quietly
    agreeing, which a leading contiguous block would hide.
    """
    rng = np.random.default_rng(11)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    texts = [
        "physics of motion",
        "dna replication basics",
        "physics and energy",
        "cooking with dna? no",
        "quantum physics primer",
        "bread baking guide",
        "dna and physics together",
        "unrelated pottery notes",
    ]
    _write(
        cdir / "f0.parquet",
        rng.standard_normal((len(texts), DIM)).astype(np.float32),
        id=[f"c{i}" for i in range(len(texts))],
        text=texts,
        tenant=[["A", "B"][i % 2] for i in range(len(texts))],
    )

    #        row:   0        1     2        3      4     5        6      7
    owner = ["other", "kw", "other", "kw2", "kw", "other", "kw2", "other"]
    keyword = ["", "physics", "", "dna", "dna", "", "physics", ""]
    n_q = len(owner)
    qpath = tmp_path / "queries.parquet"
    _write(
        qpath,
        rng.standard_normal((n_q, DIM)).astype(np.float32),
        qid=[f"q{i}" for i in range(n_q)],
        owner=owner,
        keyword_phrase=keyword,
        tenant_want=[["A", "B"][i % 2] for i in range(n_q)],
    )
    return {
        "cdir": str(cdir),
        "qpath": str(qpath),
        "tmp": tmp_path,
        "texts": texts,
        "n_q": n_q,
    }




def _run(ds, name, searches):
    out = ds["tmp"] / name
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(
            path=ds["qpath"], id_column="qid", payload_fields=["owner"]
        ),
        output=OutputConfig(path=str(out)),
        searches=searches,
    )
    return {n: _rows_of(p) for n, p in run_compute(cfg).items()}


# --------------------------------------------------------------------------
# 1. the height itself
# --------------------------------------------------------------------------
def test_mask_is_built_at_the_filters_own_height(text_ds, mask_spy):
    """2 owned rows out of an 8-row queries file -> a 2-row mask."""
    res = _run(
        text_ds,
        "own_height",
        [
            SearchSpec(
                name="kw",
                vector_type="dense",
                metric="dot",
                k=4,
                filter=KEYWORD_FILTER,
                rows={"column": "owner", "isin": ["kw"]},
            )
        ],
    )
    assert mask_spy, "no CPU-fallback mask was built — the fixture stopped testing this"
    assert all(shape[0] == 2 for shape in mask_spy), (
        f"mask height should be the filter's 2 owned rows, got {mask_spy}"
    )
    assert set(res["kw"]) == {"q1", "q4"}


def test_mask_height_does_not_grow_with_the_queries_file(tmp_path, mask_spy):
    """The regression this change exists for: pad the queries file with rows
    no search owns and the mask must not notice. Under the old behavior the
    height tracked the file (4 -> 404); it must now track the subset (4)."""
    rng = np.random.default_rng(5)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    texts = [f"alpha doc{i} " + ("charlie" if i % 2 else "delta") for i in range(12)]
    _write(
        cdir / "f0.parquet",
        rng.standard_normal((len(texts), DIM)).astype(np.float32),
        id=[f"c{i}" for i in range(len(texts))],
        text=texts,
    )

    def build(n_foreign):
        owner = ["kw"] * 4 + ["foreign"] * n_foreign
        keyword = ["charlie", "delta", "charlie", "delta"] + [""] * n_foreign
        qpath = tmp_path / f"q{n_foreign}.parquet"
        _write(
            qpath,
            rng.standard_normal((len(owner), DIM)).astype(np.float32),
            qid=[f"q{i}" for i in range(len(owner))],
            owner=owner,
            keyword_phrase=keyword,
        )
        out = tmp_path / f"out{n_foreign}"
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            searches=[
                SearchSpec(
                    name="kw",
                    vector_type="dense",
                    metric="dot",
                    k=3,
                    filter=KEYWORD_FILTER,
                    rows={"column": "owner", "isin": ["kw"]},
                )
            ],
        )

    heights = {}
    for n_foreign in (0, 400):
        mask_spy.clear()
        run_compute(build(n_foreign))
        assert mask_spy, "no CPU-fallback mask built"
        heights[n_foreign] = {s[0] for s in mask_spy}

    assert heights[0] == {4}
    assert heights[400] == {4}, (
        f"mask height grew with 400 unowned queries in the file: {heights[400]}"
    )


def test_two_specs_sharing_one_filter_span_their_union(text_ds, mask_spy):
    """`keeps` is keyed by the Filter, and one entry serves every spec using
    it — so the height has to cover all of them, and each spec indexes its own
    positions out. Two specs, disjoint rows, one shared filter value: height
    must be the 4-row union, not either spec's 2."""
    res = _run(
        text_ds,
        "shared_filter",
        [
            SearchSpec(
                name="a", vector_type="dense", metric="dot", k=4,
                filter=KEYWORD_FILTER, rows={"column": "owner", "isin": ["kw"]},
            ),
            SearchSpec(
                name="b", vector_type="dense", metric="dot", k=4,
                filter=KEYWORD_FILTER, rows={"column": "owner", "isin": ["kw2"]},
            ),
        ],
    )
    assert mask_spy, "no CPU-fallback mask built"
    assert all(shape[0] == 4 for shape in mask_spy), (
        f"height should be the 4-row union of both specs, got {mask_spy}"
    )
    assert set(res["a"]) == {"q1", "q4"}
    assert set(res["b"]) == {"q3", "q6"}
    # and each spec really got ITS OWN queries' filter applied, not the other's
    assert all("physics" in text_ds["texts"][int(c[1:])] for c in res["a"]["q1"][0])
    assert all("dna" in text_ds["texts"][int(c[1:])] for c in res["a"]["q4"][0])
    assert all("dna" in text_ds["texts"][int(c[1:])] for c in res["b"]["q3"][0])
    assert all("physics" in text_ds["texts"][int(c[1:])] for c in res["b"]["q6"][0])


def test_a_spec_owning_every_row_pins_the_filter_to_full_height(text_ds, mask_spy):
    """One spec with no `rows` sharing the filter -> nothing to narrow to."""
    _run(
        text_ds,
        "one_full",
        [
            SearchSpec(
                name="sub", vector_type="dense", metric="dot", k=4,
                filter=KEYWORD_FILTER, rows={"column": "owner", "isin": ["kw"]},
            ),
            SearchSpec(
                name="all", vector_type="dense", metric="dot", k=4,
                filter=KEYWORD_FILTER,
            ),
        ],
    )
    assert mask_spy, "no CPU-fallback mask built"
    assert all(shape[0] == text_ds["n_q"] for shape in mask_spy), (
        f"a full-row spec must keep the mask at file height, got {mask_spy}"
    )


def test_two_specs_with_different_filters_are_narrowed_independently(text_ds, mask_spy):
    """Distinct filter VALUES get distinct `keeps` entries, so each narrows to
    its own owner — one mask of height 2 and one of height 2, never one of 4
    covering both."""
    other_filter = Filter(
        must=[
            FilterCondition(field="text", match_text_from_query="keyword_phrase"),
            FilterCondition(field="tenant", match_from_query="tenant_want"),
        ]
    )
    _run(
        text_ds,
        "two_filters",
        [
            SearchSpec(
                name="a", vector_type="dense", metric="dot", k=4,
                filter=KEYWORD_FILTER, rows={"column": "owner", "isin": ["kw"]},
            ),
            SearchSpec(
                name="b", vector_type="dense", metric="dot", k=4,
                filter=other_filter, rows={"column": "owner", "isin": ["kw2"]},
            ),
        ],
    )
    assert len(mask_spy) >= 2, f"expected a mask per filter, got {mask_spy}"
    assert all(shape[0] == 2 for shape in mask_spy), (
        f"each filter should narrow to its own 2 rows, got {mask_spy}"
    )


# --------------------------------------------------------------------------
# 2. results are unchanged
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "searches_desc",
    ["single", "shared_filter", "with_full_row_spec", "mixed_filters"],
)
def test_narrowed_mask_reproduces_full_height_results_exactly(
    text_ds, tmp_path, monkeypatch, searches_desc
):
    """Bit-for-bit, not approx: only the mask's height changes, so every
    matmul shape — and therefore every f32 accumulation order — is identical.
    Anything but exact equality here means the narrowing moved real work."""
    tenant_filter = Filter(
        must=[
            FilterCondition(field="text", match_text_from_query="keyword_phrase"),
            FilterCondition(field="tenant", match_from_query="tenant_want"),
        ]
    )
    kw = {"column": "owner", "isin": ["kw"]}
    kw2 = {"column": "owner", "isin": ["kw2"]}
    variants = {
        "single": lambda: [
            SearchSpec(name="a", vector_type="dense", metric="dot", k=4,
                       filter=KEYWORD_FILTER, rows=kw)
        ],
        "shared_filter": lambda: [
            SearchSpec(name="a", vector_type="dense", metric="dot", k=4,
                       filter=KEYWORD_FILTER, rows=kw),
            SearchSpec(name="b", vector_type="dense", metric="dot", k=4,
                       filter=KEYWORD_FILTER, rows=kw2),
        ],
        "with_full_row_spec": lambda: [
            SearchSpec(name="a", vector_type="dense", metric="dot", k=4,
                       filter=KEYWORD_FILTER, rows=kw),
            SearchSpec(name="b", vector_type="dense", metric="dot", k=4),
        ],
        "mixed_filters": lambda: [
            SearchSpec(name="a", vector_type="dense", metric="dot", k=4,
                       filter=KEYWORD_FILTER, rows=kw),
            SearchSpec(name="b", vector_type="dense", metric="dot", k=4,
                       filter=tenant_filter, rows=kw2),
            SearchSpec(name="c", vector_type="dense", metric="cosine", k=4),
        ],
    }

    narrowed = _run(text_ds, f"narrow_{searches_desc}", variants[searches_desc]())

    real = compute_mod._union_rows_by_key

    def patched(keys, spec_rows, n_q):
        out = real(keys, spec_rows, n_q)
        return out if all(isinstance(k, str) for k in keys) else {k: None for k in out}

    monkeypatch.setattr(compute_mod, "_union_rows_by_key", patched)
    full = _run(text_ds, f"full_{searches_desc}", variants[searches_desc]())

    assert set(narrowed) == set(full)
    for name in narrowed:
        assert set(narrowed[name]) == set(full[name]), f"{name}: query set changed"
        for q in narrowed[name]:
            assert narrowed[name][q][0] == full[name][q][0], f"{name}/{q}: hit_ids"
            assert narrowed[name][q][1] == full[name][q][1], f"{name}/{q}: hit_scores"


def test_narrowing_survives_a_gapped_non_prefix_subset(text_ds, full_height):
    """`full_height` is applied to THIS run, so it is the control; the
    narrowed run above it is the subject. Rows 1 and 4 are neither a prefix
    nor contiguous, so `spec_qrows` is a real gather into the narrowed mask —
    the case where confusing file rows with union positions gives a wrong
    answer rather than an IndexError."""
    control = _run(
        text_ds, "gapped_control",
        [SearchSpec(name="kw", vector_type="dense", metric="dot", k=4,
                    filter=KEYWORD_FILTER, rows={"column": "owner", "isin": ["kw"]})],
    )
    assert set(control["kw"]) == {"q1", "q4"}
    assert control["kw"]["q1"][0], "fixture produced no hits to compare"
    assert all("physics" in text_ds["texts"][int(c[1:])] for c in control["kw"]["q1"][0])
    assert all("dna" in text_ds["texts"][int(c[1:])] for c in control["kw"]["q4"][0])


# --------------------------------------------------------------------------
# 3. what must NOT be narrowed
# --------------------------------------------------------------------------
def test_gpu_eligible_filter_is_left_at_full_height(text_ds, mask_spy):
    """`match_from_query` alone is GPU-eligible (Front A): no CPU mask is
    built at all, its per-query state is shared across filters by
    FilterCondition, and its mask is per-BATCH rather than per-file. Narrowing
    it would mean un-sharing that state for a ~1000x smaller allocation, so it
    is deliberately excluded — assert we really did stay off the CPU path."""
    res = _run(
        text_ds,
        "gpu_elig",
        [
            SearchSpec(
                name="t", vector_type="dense", metric="dot", k=4,
                filter=Filter(must=[FilterCondition(field="tenant",
                                                    match_from_query="tenant_want")]),
                rows={"column": "owner", "isin": ["kw"]},
            )
        ],
    )
    assert mask_spy == [], "a GPU-eligible filter must not build a CPU mask"
    assert set(res["t"]) == {"q1", "q4"}


def test_uniform_text_filter_has_no_query_axis_to_narrow(text_ds, mask_spy):
    """A static `match_text` is uniform: `evaluate` returns `(rows,)`, which
    `_pack_query_axis` never sees. Narrowing must not invent a query axis."""
    res = _run(
        text_ds,
        "uniform",
        [
            SearchSpec(
                name="u", vector_type="dense", metric="dot", k=4,
                filter=Filter(must=[FilterCondition(field="text",
                                                    match_text="physics")]),
                rows={"column": "owner", "isin": ["kw"]},
            )
        ],
    )
    assert mask_spy == [], "a uniform filter has no per-query mask to pack"
    assert set(res["u"]) == {"q1", "q4"}
    for q in res["u"]:
        assert all("physics" in text_ds["texts"][int(c[1:])] for c in res["u"][q][0])


# --------------------------------------------------------------------------
# 4. _union_rows_by_key
# --------------------------------------------------------------------------
def test_union_rows_by_key_unions_within_a_key():
    out = _union_rows_by_key(
        ["x", "y", "x"],
        [np.array([0, 3]), np.array([1]), np.array([3, 5])],
        n_q=8,
    )
    assert out["x"].tolist() == [0, 3, 5], "same-key rows must union, deduped and sorted"
    assert out["y"].tolist() == [1]


def test_union_rows_by_key_propagates_none():
    """`None` means "every row"; unioning anything with it stays `None`, in
    both orders (a later `None` must not be swallowed by an earlier array)."""
    assert _union_rows_by_key(["x", "x"], [None, np.array([1])], n_q=4)["x"] is None
    assert _union_rows_by_key(["x", "x"], [np.array([1]), None], n_q=4)["x"] is None


def test_union_rows_by_key_collapses_a_full_cover_to_none():
    """A union that turns out to cover the whole file is reported as `None` so
    the caller skips the indexing entirely — the same collapse
    `_resolve_spec_rows` has always done for vector_types."""
    out = _union_rows_by_key(["x", "x"], [np.array([0, 1]), np.array([2, 3])], n_q=4)
    assert out["x"] is None
    out = _union_rows_by_key(["x", "x"], [np.array([0, 1]), np.array([2])], n_q=4)
    assert out["x"].tolist() == [0, 1, 2]


def test_union_rows_by_key_accepts_filter_and_none_keys():
    """The filter-keyed call passes `Filter | None`; `Filter` is frozen so it
    hashes, and `None` (the unfiltered entry) is a legitimate key."""
    out = _union_rows_by_key(
        [KEYWORD_FILTER, None, KEYWORD_FILTER],
        [np.array([0]), np.array([1]), np.array([2])],
        n_q=9,
    )
    assert out[KEYWORD_FILTER].tolist() == [0, 2]
    assert out[None].tolist() == [1]


def test_union_rows_by_key_matches_resolve_spec_rows_for_vector_types(text_ds):
    """`_resolve_spec_rows` delegates its vt_rows to this helper; if the two
    ever diverge, the query matrix and the mask would disagree about what a
    subset means."""
    from nova_bf.compute import _resolve_spec_rows

    specs = [
        SearchSpec(name="a", vector_type="dense", metric="dot", k=2,
                   rows={"column": "owner", "isin": ["kw"]}),
        SearchSpec(name="b", vector_type="dense", metric="dot", k=2,
                   rows={"column": "owner", "isin": ["kw2"]}),
    ]
    vals = {"owner": np.array(["other", "kw", "other", "kw2", "kw", "other", "kw2", "other"])}
    spec_rows, vt_rows = _resolve_spec_rows(specs, vals, 8)
    direct = _union_rows_by_key([s.vector_type for s in specs], spec_rows, 8)
    assert direct.keys() == vt_rows.keys()
    assert np.array_equal(direct["dense"], vt_rows["dense"])


# ==========================================================================
# 5. the load-bearing claim, tested on filters.evaluate directly
# ==========================================================================
@pytest.fixture
def corpus_table():
    return pa.table({
        "text": pa.array([
            "physics of motion and energy",
            "dna replication basics",
            "quantum physics primer",
            "bread baking guide",
            "dna and physics together",
            "pottery glazing notes",
            "energy from dna? no",
            "motion capture rigs",
        ]),
        "url": pa.array([
            "a.edu/p", "b.com/d", "c.edu/q", "d.org/b",
            "e.edu/dp", "f.net/g", "g.com/e", "h.io/m",
        ]),
        "tenant": pa.array(["A", "B", "A", "C", "B", "A", "C", "B"]),
        "score": pa.array([0.1, 0.5, 0.9, 0.2, 0.7, 0.4, 0.8, 0.3]),
    })


# Every filter shape that reaches the CPU-fallback path, plus one that mixes a
# text leaf with a non-text per-query leaf (so `evaluate` has to broadcast a
# 2-D accumulator against the fused query-major text path).
EQUIV_FILTERS = {
    "must_text": Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase"),
    ]),
    "should_text_group": Filter(should=[
        FilterCondition(field="url", match_text_from_query="slot1"),
        FilterCondition(field="url", match_text_from_query="slot2"),
    ]),
    "must_not_text": Filter(
        must=[FilterCondition(field="text", match_text_from_query="phrase")],
        must_not=[FilterCondition(field="url", match_text_from_query="slot1")],
    ),
    "text_plus_match_from_query": Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase"),
        FilterCondition(field="tenant", match_from_query="tenant_want"),
    ]),
    "text_plus_range_from_query": Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase"),
        FilterCondition(field="score", range_from_query={"gte": "score_gte"}),
    ]),
    "text_plus_static": Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase"),
        FilterCondition(field="tenant", match=["A", "B"]),
    ]),
    "all_three_groups": Filter(
        must=[FilterCondition(field="text", match_text_from_query="phrase")],
        should=[
            FilterCondition(field="url", match_text_from_query="slot1"),
            FilterCondition(field="url", match_text_from_query="slot2"),
        ],
        must_not=[FilterCondition(field="tenant", match_from_query="tenant_want")],
    ),
}


@pytest.fixture
def query_values():
    """8 queries. Deliberately awkward:
      * row 3's phrase token ("pottery") appears in NO other query — if the
        shared tokenization pass leaked, narrowing the query set would change
        which tokens got masks built and, if masks were mis-shared, another
        query's answer.
      * rows 1 and 6 are IDENTICAL in every column, so `combos` groups them —
        narrowing that keeps one and drops the other must not disturb the
        survivor.
      * row 5 carries a null (`None` = no restriction) and row 2 an empty
        phrase (token-less = matches nothing in a `must`): the two opposite
        conventions, both present.
      * `tenant_want` mixes a scalar and a list (MatchAny).
    """
    return {
        "phrase":      np.array(["physics", "dna", "", "pottery", "physics energy", "motion", "dna", "energy"], dtype=object),
        "slot1":       np.array(["edu", "com", "edu", "net", "edu", None, "com", "io"], dtype=object),
        "slot2":       np.array(["org", "io", "net", "edu", "com", "org", "io", "edu"], dtype=object),
        "tenant_want": np.array(["A", "B", ["A", "C"], "C", None, "A", "B", ["B", "C"]], dtype=object),
        "score_gte":   np.array([0.0, 0.3, 0.5, 0.2, np.nan, 0.1, 0.3, 0.6]),
    }


# Row subsets that a `rows` selector could plausibly produce.
EQUIV_SUBSETS = {
    "prefix": [0, 1, 2],
    "suffix": [5, 6, 7],
    "gapped": [1, 3, 6],
    "single": [4],
    "interleaved": [0, 2, 4, 6],
    "identical_pair_split": [1],   # keeps one half of the combos-grouped pair
    "identical_pair_both": [1, 6],
    "all": [0, 1, 2, 3, 4, 5, 6, 7],
}


@pytest.mark.parametrize("filter_name", sorted(EQUIV_FILTERS))
@pytest.mark.parametrize("subset_name", sorted(EQUIV_SUBSETS))
def test_evaluate_on_narrowed_values_equals_the_row_slice(
    corpus_table, query_values, filter_name, subset_name
):
    """THE claim the whole change rests on:

        evaluate(f, table, {c: v[rows]}) == evaluate(f, table, v)[rows]

    If `filters.evaluate` had any cross-query coupling, this is where it would
    show up — and no amount of end-to-end agreement on a friendly fixture
    would rule it out.
    """
    f = EQUIV_FILTERS[filter_name]
    rows = np.array(EQUIV_SUBSETS[subset_name], dtype=np.int64)

    full = evaluate(f, corpus_table, query_values)
    narrow = evaluate(f, corpus_table, {c: v[rows] for c, v in query_values.items()})

    assert full.ndim == 2, "fixture no longer exercises the per-query path"
    assert narrow.shape == (len(rows), len(corpus_table))
    np.testing.assert_array_equal(
        narrow, full[rows],
        err_msg=f"{filter_name}/{subset_name}: narrowing changed a query's own mask",
    )


def test_the_equivalence_fixture_actually_discriminates(corpus_table, query_values):
    """Guard on the fixture: if every query produced the same mask row, the
    test above would pass no matter how badly narrowing were implemented."""
    full = evaluate(EQUIV_FILTERS["all_three_groups"], corpus_table, query_values)
    distinct = {row.tobytes() for row in full}
    assert len(distinct) >= 4, f"only {len(distinct)} distinct mask rows — too weak"
    assert full.any(), "fixture matches nothing at all"
    assert not full.all(), "fixture matches everything"


# ==========================================================================
# 6. grouping: the height must cover every SHARER of the filter value
# ==========================================================================
@pytest.fixture
def two_vt_ds(tmp_path):
    """A queries file whose dense and sparse searches share one filter VALUE
    but own different rows."""
    rng = np.random.default_rng(23)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n_c = 12
    texts = [f"doc{i} " + ("charlie" if i % 2 else "delta") for i in range(n_c)]
    _write(
        cdir / "f0.parquet",
        rng.standard_normal((n_c, DIM)).astype(np.float32),
        [_sparse_row(rng) for _ in range(n_c)],
        id=[f"c{i}" for i in range(n_c)], text=texts,
    )
    owner = ["d", "s", "none", "d", "s", "none"]
    phrase = ["charlie", "delta", "", "delta", "charlie", ""]
    n_q = len(owner)
    qpath = tmp_path / "queries.parquet"
    _write(
        qpath,
        rng.standard_normal((n_q, DIM)).astype(np.float32),
        [_sparse_row(rng) for _ in range(n_q)],
        qid=[f"q{i}" for i in range(n_q)], owner=owner, phrase=phrase,
    )
    return {"cdir": str(cdir), "qpath": str(qpath), "tmp": tmp_path, "n_q": n_q}




def test_a_filter_shared_across_vector_types_spans_both(two_vt_ds, monkeypatch):
    """`keeps` is per-FILTER and vector_type-agnostic (filters read payload
    columns, not vector columns), so one dense spec and one sparse spec
    sharing a filter value read the SAME mask. Its height must be the union of
    both specs' rows — a height computed per (filter, vector_type) would be
    too short for whichever one ran second, and would silently mask the wrong
    queries rather than raise."""
    seen = []
    real = compute_mod._pack_query_axis
    monkeypatch.setattr(
        compute_mod, "_pack_query_axis",
        lambda m: (seen.append(m.shape), real(m))[1],
    )
    out = two_vt_ds["tmp"] / "two_vt"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=two_vt_ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=two_vt_ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="d", vector_type="dense", metric="dot", k=4,
                       filter=PHRASE_FILTER, rows={"column": "owner", "isin": ["d"]}),
            SearchSpec(name="s", vector_type="sparse", metric="dot", k=4,
                       filter=PHRASE_FILTER, rows={"column": "owner", "isin": ["s"]}),
        ],
    )
    res = {n: _rows_of(p) for n, p in run_compute(cfg).items()}

    assert seen, "no CPU-fallback mask built"
    assert all(s[0] == 4 for s in seen), (
        f"height must span BOTH vector_types' rows (4), got {seen}"
    )
    assert set(res["d"]) == {"q0", "q3"}
    assert set(res["s"]) == {"q1", "q4"}
    # each got its OWN phrase applied, not its neighbour's
    assert all(int(c[1:]) % 2 == 1 for c in res["d"]["q0"][0]), "q0 wanted 'charlie'"
    assert all(int(c[1:]) % 2 == 0 for c in res["d"]["q3"][0]), "q3 wanted 'delta'"
    assert all(int(c[1:]) % 2 == 0 for c in res["s"]["q1"][0]), "q1 wanted 'delta'"
    assert all(int(c[1:]) % 2 == 1 for c in res["s"]["q4"][0]), "q4 wanted 'charlie'"


def test_equal_but_distinct_filter_objects_are_one_group(two_vt_ds, monkeypatch):
    """`Filter` is compared by VALUE everywhere (`dict.fromkeys`, `Counter`,
    the `keeps` dict), so two separately-constructed but equal filters are ONE
    entry. The height grouping must use the same notion of identity, or the
    single shared mask ends up sized for only one of them."""
    a = Filter(must=[FilterCondition(field="text", match_text_from_query="phrase")])
    b = Filter(must=[FilterCondition(field="text", match_text_from_query="phrase")])
    assert a is not b and a == b and hash(a) == hash(b), "premise broke"

    seen = []
    real = compute_mod._pack_query_axis
    monkeypatch.setattr(
        compute_mod, "_pack_query_axis",
        lambda m: (seen.append(m.shape), real(m))[1],
    )
    out = two_vt_ds["tmp"] / "equal_filters"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=two_vt_ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=two_vt_ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="x", vector_type="dense", metric="dot", k=4,
                       filter=a, rows={"column": "owner", "isin": ["d"]}),
            SearchSpec(name="y", vector_type="dense", metric="dot", k=4,
                       filter=b, rows={"column": "owner", "isin": ["s"]}),
        ],
    )
    res = {n: _rows_of(p) for n, p in run_compute(cfg).items()}
    assert len(seen) == 1, f"equal filters should evaluate once, got {len(seen)} masks"
    assert seen[0][0] == 4, f"height must span both specs' rows, got {seen}"
    assert set(res["x"]) == {"q0", "q3"}
    assert set(res["y"]) == {"q1", "q4"}


# ==========================================================================
# 7. the corpus-row union, and when it actually tightens
# ==========================================================================
@pytest.mark.parametrize(
    "foreign_phrase, expect_tighter",
    [
        ("echo", True),    # a REAL value: only then does a foreign query widen the union
        ("", False),       # the sentinel convention: token-less `must` already matched nothing
        (None, False),     # null: same, for a TEXT leaf (unlike match_from_query, where
                           # null means "no restriction" and really does widen it)
    ],
    ids=["real_foreign_value", "sentinel_foreign_value", "null_foreign_value"],
)
def test_union_keep_tightens_only_when_foreign_rows_carry_real_values(
    tmp_path, monkeypatch, foreign_phrase, expect_tighter
):
    """With no unfiltered spec the shared batch grid is `_union_keep`'s
    OR-reduce, computed over the mask's query axis — so narrowing can only
    ever shrink it. But by how much depends entirely on what the foreign rows
    hold, and it is easy to overclaim here:

      * real foreign values      -> the grid genuinely shrinks
      * sentinel ("" / no tokens) -> no change: a token-less phrase in a `must`
        already matched nothing, so those queries contributed nothing to the OR
      * null                      -> no change either, for a TEXT leaf

    So for the shipped `filtered_text` config, whose foreign rows carry the
    empty-phrase sentinel, this is worth exactly zero. The narrowing's payoff
    is the ALLOCATION, not the union. Both directions are pinned so the claim
    can't drift back to the optimistic version.

    Results must be bit-identical in every case: a row no owning query wants
    cannot contribute a score, and tie-break ordinals are derived from true
    file rows, not from position within the compacted batch.
    """
    rng = np.random.default_rng(41)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # "echo" appears ONLY in doc 2 — a row no owning query can reach.
    texts = ["alpha charlie", "beta delta", "gamma echo", "zeta charlie",
             "eta delta", "theta echo"]
    _write(cdir / "f0.parquet",
           rng.standard_normal((len(texts), DIM)).astype(np.float32),
           id=[f"c{i}" for i in range(len(texts))], text=texts)

    owner = ["kw", "foreign", "kw", "foreign"]
    phrase = ["charlie", foreign_phrase, "delta", foreign_phrase]
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((len(owner), DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(len(owner))], owner=owner, phrase=phrase)

    def build(name):
        out = tmp_path / name
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            # the only spec is filtered => has_baseline False => _union_keep runs
            searches=[SearchSpec(name="kw", vector_type="dense", metric="dot", k=4,
                                 filter=PHRASE_FILTER,
                                 rows={"column": "owner", "isin": ["kw"]})],
        )

    def run_and_measure(cfg):
        widths = []
        real = compute_mod._union_keep

        def spy(fs, keeps):
            u = real(fs, keeps)
            widths.append(int(u.sum()))
            return u

        monkeypatch.setattr(compute_mod, "_union_keep", spy)
        res = {n: _rows_of(p) for n, p in run_compute(cfg).items()}
        monkeypatch.undo()
        return res, widths

    narrowed, narrow_w = run_and_measure(build("union_narrow"))
    assert narrow_w, "_union_keep was never called — this path is not under test"

    real_union = compute_mod._union_rows_by_key
    monkeypatch.setattr(
        compute_mod, "_union_rows_by_key",
        lambda keys, sr, n: (
            real_union(keys, sr, n) if all(isinstance(k, str) for k in keys)
            else {k: None for k in real_union(keys, sr, n)}
        ),
    )
    full, full_w = run_and_measure(build("union_full"))
    monkeypatch.undo()

    if expect_tighter:
        assert sum(narrow_w) < sum(full_w), (
            f"expected the union to tighten: narrowed {narrow_w}, full {full_w}"
        )
    else:
        assert sum(narrow_w) == sum(full_w), (
            f"expected NO tightening with foreign_phrase={foreign_phrase!r}: "
            f"narrowed {narrow_w}, full {full_w}"
        )

    assert set(narrowed["kw"]) == set(full["kw"]) == {"q0", "q2"}
    assert narrowed["kw"]["q0"][0], "fixture produced no hits"
    for q in narrowed["kw"]:
        assert narrowed["kw"][q][0] == full["kw"][q][0], f"{q}: hit_ids changed"
        assert narrowed["kw"][q][1] == full["kw"][q][1], f"{q}: hit_scores changed"


# ==========================================================================
# 8. cross-file coalescing must stay off for a per-query filter
# ==========================================================================
def test_per_query_filter_survives_multiple_files_with_a_batch_size(tmp_path):
    """`_flush_coalesce_group` rebuilds `keeps` entries as `file_keeps[f][orig_rows]`
    — the ROW axis for a uniform `(rows,)` mask, but the QUERY axis for a
    packed 2-D one. It is meant to be disabled whenever a per-query filter
    shares the vector_type. This is the configuration that would trip it if
    the guard ever lapsed: several corpus files, an explicit batch size, and
    no unfiltered spec — compared against a single-file run over the same
    rows, which cannot coalesce at all."""
    rng = np.random.default_rng(31)
    n_per, n_files = 9, 3
    texts, dense = [], []
    for fi in range(n_files):
        for i in range(n_per):
            texts.append(f"f{fi}doc{i} " + ("charlie" if (fi + i) % 2 else "delta"))
    dense = rng.standard_normal((n_per * n_files, DIM)).astype(np.float32)

    multi = tmp_path / "multi"
    multi.mkdir()
    for fi in range(n_files):
        lo, hi = fi * n_per, (fi + 1) * n_per
        _write(multi / f"f{fi}.parquet", dense[lo:hi],
               id=[f"c{i}" for i in range(lo, hi)], text=texts[lo:hi])
    single = tmp_path / "single"
    single.mkdir()
    _write(single / "f0.parquet", dense,
           id=[f"c{i}" for i in range(len(texts))], text=texts)

    owner = ["kw", "other", "kw", "other"]
    phrase = ["charlie", "", "delta", ""]
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((len(owner), DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(len(owner))], owner=owner, phrase=phrase)

    def run(cdir, name):
        out = tmp_path / name
        out.mkdir()
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            # small enough that several files' post-compaction batches would
            # be coalesced if coalescing were eligible here
            params=ParamsConfig(dense_batch_size=4),
            searches=[SearchSpec(name="kw", vector_type="dense", metric="dot", k=6,
                                 filter=PHRASE_FILTER,
                                 rows={"column": "owner", "isin": ["kw"]})],
        )
        return _rows_of(run_compute(cfg)["kw"])

    many, one = run(multi, "multi_out"), run(single, "single_out")
    assert set(many) == set(one) == {"q0", "q2"}
    for q in many:
        assert many[q][0], "fixture produced no hits"
        assert sorted(many[q][0]) == sorted(one[q][0]), f"{q}: hits differ across files"
        assert sorted(many[q][1], reverse=True) == pytest.approx(
            sorted(one[q][1], reverse=True)
        ), f"{q}: scores differ across files"
    # and the filter really bit
    assert all("charlie" in texts[int(c[1:])] for c in many["q0"][0])
    assert all("delta" in texts[int(c[1:])] for c in many["q2"][0])


# ==========================================================================
# 9. tie-breaking must not notice the union
# ==========================================================================
@pytest.fixture
def tied_ds(tmp_path):
    """Every corpus vector is IDENTICAL, so every score ties exactly and the
    tie-break ordinal alone decides the top-K. Split across 3 files so
    `ordinal_base` accumulation is in play.

    Half the docs say "charlie" (what the owning queries want), half say
    "echo" (what the FOREIGN queries want). The foreign rows carry real
    values, which is the only case where the union actually tightens — so
    under narrowing the echo docs never enter the batch, and under full
    height they do. Same tied candidates either way; only the batch
    composition differs.
    """
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n_files, per = 3, 8
    vec = np.full((per, DIM), 0.5, dtype=np.float32)  # identical => exact ties
    texts_all = []
    for fi in range(n_files):
        texts = [("charlie" if i % 2 == 0 else "echo") + f" doc" for i in range(per)]
        texts_all += texts
        lo = fi * per
        _write(cdir / f"f{fi}.parquet", vec,
               id=[f"c{lo + i}" for i in range(per)], text=texts)

    owner = ["kw", "foreign", "kw", "foreign"]
    phrase = ["charlie", "echo", "charlie", "echo"]
    qpath = tmp_path / "q.parquet"
    _write(qpath, np.full((len(owner), DIM), 0.25, dtype=np.float32),
           qid=[f"q{i}" for i in range(len(owner))], owner=owner, phrase=phrase)
    return {"cdir": str(cdir), "qpath": str(qpath), "tmp": tmp_path,
            "texts": texts_all, "n_charlie": sum(1 for t in texts_all if "charlie" in t)}


def _tied_cfg(ds, name, k):
    out = ds["tmp"] / name
    out.mkdir()
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=3),  # force several batches per file
        searches=[SearchSpec(name="kw", vector_type="dense", metric="dot", k=k,
                             filter=PHRASE_FILTER,
                             rows={"column": "owner", "isin": ["kw"]})],
    )


def test_exact_ties_break_identically_despite_a_tighter_union(tied_ds, monkeypatch):
    """k=5 out of 12 exactly-tied candidates: which 5, and in what order, is
    decided purely by the tie-break ordinal. Narrowing changes the batch's
    composition (the echo rows stop being compacted in) but must not change
    the answer by even one position."""
    k = 5
    assert tied_ds["n_charlie"] > k, "fixture must force a real tie-break choice"

    narrow_w = _spy_union_widths(monkeypatch)
    narrowed = _rows_of(run_compute(_tied_cfg(tied_ds, "tie_narrow", k))["kw"])
    monkeypatch.undo()

    full_w = _spy_union_widths(monkeypatch)
    _force_full_height(monkeypatch)
    full = _rows_of(run_compute(_tied_cfg(tied_ds, "tie_full", k))["kw"])
    monkeypatch.undo()

    # Guard: if the two runs saw the same batch grid, this test proves nothing
    # about tie-breaking being independent of it. Observed 12 vs 24 rows.
    assert sum(narrow_w) < sum(full_w), (
        f"fixture stopped varying the batch composition: narrowed kept "
        f"{narrow_w}, full-height {full_w} — the tie-break claim is untested"
    )

    assert set(narrowed) == set(full) == {"q0", "q2"}
    for q in narrowed:
        hits, scores = narrowed[q]
        assert len(hits) == k, f"{q}: expected a saturated top-{k}, got {len(hits)}"
        # all tied, so this really is the tie-break deciding
        assert len(set(scores)) == 1, f"{q}: fixture no longer produces exact ties"
        assert hits == full[q][0], (
            f"{q}: tie-break order changed with the union\n"
            f"  narrowed={hits}\n  full    ={full[q][0]}"
        )
        assert all("charlie" in tied_ds["texts"][int(c[1:])] for c in hits)


def test_tie_break_is_stable_across_repeated_narrowed_runs(tied_ds):
    """Determinism of the narrowed path on its own terms — reader threads
    finish in arbitrary order, so a tie-break that depended on arrival order
    would show up as run-to-run drift rather than as a mismatch above."""
    first = _rows_of(run_compute(_tied_cfg(tied_ds, "tie_rep0", 5))["kw"])
    for i in range(1, 4):
        again = _rows_of(run_compute(_tied_cfg(tied_ds, f"tie_rep{i}", 5))["kw"])
        for q in first:
            assert again[q][0] == first[q][0], f"run {i}, {q}: tie-break drifted"


# ==========================================================================
# 10. a file the tightened union empties out entirely
# ==========================================================================
def test_a_file_with_no_surviving_rows_is_handled(tmp_path, monkeypatch):
    """The union can now drop every row of a corpus file that only foreign
    queries wanted. Exercises the empty-compaction path, `select`'s
    `not cell_mask.any()` early-out and the `corpus_ids` retention check, and
    must still agree with full height."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # file 0: ONLY echo (what the foreign queries want) -> empty under narrowing
    # file 1: only charlie   file 2: mixed
    files = [
        ["echo one", "echo two", "echo three"],
        ["charlie one", "charlie two", "charlie three"],
        ["echo four", "charlie four", "echo five"],
    ]
    rng = np.random.default_rng(77)
    texts_all, n = [], 0
    for fi, texts in enumerate(files):
        _write(cdir / f"f{fi}.parquet",
               rng.standard_normal((len(texts), DIM)).astype(np.float32),
               id=[f"c{n + i}" for i in range(len(texts))], text=texts)
        texts_all += texts
        n += len(texts)

    owner, phrase = ["kw", "foreign"], ["charlie", "echo"]
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((2, DIM)).astype(np.float32),
           qid=["q0", "q1"], owner=owner, phrase=phrase)

    def build(name):
        out = tmp_path / name
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            searches=[SearchSpec(name="kw", vector_type="dense", metric="dot", k=4,
                                 filter=PHRASE_FILTER,
                                 rows={"column": "owner", "isin": ["kw"]})],
        )

    narrow_w = _spy_union_widths(monkeypatch)
    narrowed = _rows_of(run_compute(build("empty_narrow"))["kw"])
    monkeypatch.undo()

    _force_full_height(monkeypatch)
    full = _rows_of(run_compute(build("empty_full"))["kw"])
    monkeypatch.undo()

    # Guard: file 0 is all-echo, so under narrowing its union must be EMPTY —
    # otherwise the empty-batch path this test exists for never runs.
    assert 0 in narrow_w, (
        f"no corpus file was emptied by the tightened union ({narrow_w}); "
        "the empty-batch path is not under test"
    )

    assert set(narrowed) == {"q0"}
    hits = narrowed["q0"][0]
    assert len(hits) == 4, f"expected all 4 charlie docs, got {hits}"
    assert all("charlie" in texts_all[int(c[1:])] for c in hits)
    assert hits == full["q0"][0]
    assert narrowed["q0"][1] == full["q0"][1]


# ==========================================================================
# 11. two CPU-fallback filters, different heights, built concurrently
# ==========================================================================
def test_concurrent_filters_with_different_heights_stay_independent(tmp_path):
    """`evaluate` is dispatched on a thread pool once there is more than one
    CPU-fallback filter, and each call now receives its OWN narrowed value
    dict and produces its own height. Two filters of DIFFERENT heights over
    several files, repeated, must give a stable and individually-correct
    answer — a shared or swapped value dict would show up as one search
    getting the other's phrases."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    rng = np.random.default_rng(99)
    words = ["charlie", "delta", "echo"]
    texts_all, n = [], 0
    for fi in range(3):
        texts = [f"f{fi} doc{i} {words[(fi + i) % 3]}" for i in range(6)]
        _write(cdir / f"f{fi}.parquet",
               rng.standard_normal((len(texts), DIM)).astype(np.float32),
               id=[f"c{n + i}" for i in range(len(texts))], text=texts,
               tenant=[["A", "B"][i % 2] for i in range(len(texts))])
        texts_all += texts
        n += len(texts)

    # search "a" owns 3 rows, search "b" owns 1 -> deliberately unequal heights
    owner = ["a", "a", "a", "b", "none", "none"]
    phrase = ["charlie", "delta", "echo", "charlie", "", ""]
    want = ["A", "B", "A", "B", "A", "A"]
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((len(owner), DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(len(owner))],
           owner=owner, phrase=phrase, tenant_want=want)

    # distinct VALUES -> two keeps entries -> the ThreadPoolExecutor branch
    filt_b = Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase"),
        FilterCondition(field="tenant", match_from_query="tenant_want"),
    ])

    def build(name):
        out = tmp_path / name
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=4, dense_batch_size=2),
            searches=[
                SearchSpec(name="a", vector_type="dense", metric="dot", k=4,
                           filter=PHRASE_FILTER, rows={"column": "owner", "isin": ["a"]}),
                SearchSpec(name="b", vector_type="dense", metric="dot", k=4,
                           filter=filt_b, rows={"column": "owner", "isin": ["b"]}),
            ],
        )

    runs = [{n: _rows_of(p) for n, p in run_compute(build(f"conc{i}")).items()}
            for i in range(3)]
    for i, res in enumerate(runs):
        assert set(res["a"]) == {"q0", "q1", "q2"}, f"run {i}"
        assert set(res["b"]) == {"q3"}, f"run {i}"
        # each query got ITS OWN phrase, not a neighbour's
        for q, word in [("q0", "charlie"), ("q1", "delta"), ("q2", "echo")]:
            assert res["a"][q][0], f"run {i}: {q} produced no hits"
            assert all(word in texts_all[int(c[1:])] for c in res["a"][q][0]), \
                f"run {i}: {q} got hits not matching {word!r}"
        assert res["b"]["q3"][0], f"run {i}: q3 produced no hits"
        assert all("charlie" in texts_all[int(c[1:])] for c in res["b"]["q3"][0])
        if i:
            assert res == runs[0], f"run {i} differed from run 0 — nondeterminism"


# ==========================================================================
# 12. dtype: a MatchAny column whose only list rows are foreign
# ==========================================================================
def test_object_dtype_from_foreign_rows_survives_narrowing(tmp_path):
    """`_to_query_array` scans the WHOLE column to decide between an object
    MatchAny array and a plain one, and the narrowing slices that array after
    the fact. Here only the FOREIGN rows hold lists, so the column is an
    object array whose owned entries are bare scalars — narrowing keeps it an
    object array, and `evaluate` must read the scalars the same way it would
    in a plain array."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    rng = np.random.default_rng(123)
    texts = [f"doc{i} charlie" for i in range(6)]
    tenants = ["A", "B", "C", "A", "B", "C"]
    _write(cdir / "f0.parquet",
           rng.standard_normal((6, DIM)).astype(np.float32),
           id=[f"c{i}" for i in range(6)], text=texts, tenant=tenants)

    # owned rows: scalar tenant. foreign rows: a LIST -> forces object dtype.
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((3, DIM)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32())),
        "qid": pa.array(["q0", "q1", "q2"]),
        "owner": pa.array(["kw", "kw", "foreign"]),
        "phrase": pa.array(["charlie", "charlie", "charlie"]),
        "tenant_want": pa.array([["A"], ["B"], ["A", "B", "C"]]),
    }), str(qpath / ".." / "q.parquet") if False else str(qpath))

    out = tmp_path / "obj_out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="kw", vector_type="dense", metric="dot", k=6,
            filter=Filter(must=[
                FilterCondition(field="text", match_text_from_query="phrase"),
                FilterCondition(field="tenant", match_from_query="tenant_want"),
            ]),
            rows={"column": "owner", "isin": ["kw"]},
        )],
    )
    res = _rows_of(run_compute(cfg)["kw"])
    assert set(res) == {"q0", "q1"}
    assert {tenants[int(c[1:])] for c in res["q0"][0]} == {"A"}
    assert {tenants[int(c[1:])] for c in res["q1"][0]} == {"B"}


# ==========================================================================
# 13. a selector that names every row
# ==========================================================================
def test_a_selector_covering_every_row_collapses_to_full_height(tmp_path, monkeypatch):
    """`isin` listing every value gives a non-None `spec_rows` of length n_q,
    which `_union_rows_by_key` collapses to `None`. That reaches full height
    by a different route than "no selector", and the resulting
    `_local_positions(rows, None)` must still be the identity."""
    seen = []
    real = compute_mod._pack_query_axis
    monkeypatch.setattr(compute_mod, "_pack_query_axis",
                        lambda m: (seen.append(m.shape), real(m))[1])

    cdir = tmp_path / "corpus"
    cdir.mkdir()
    rng = np.random.default_rng(5)
    texts = ["alpha charlie", "beta delta", "gamma charlie", "zeta delta"]
    _write(cdir / "f0.parquet", rng.standard_normal((4, DIM)).astype(np.float32),
           id=[f"c{i}" for i in range(4)], text=texts)
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((3, DIM)).astype(np.float32),
           qid=["q0", "q1", "q2"], owner=["x", "y", "x"],
           phrase=["charlie", "delta", "delta"])

    out = tmp_path / "allrows"
    out.mkdir()
    res = _rows_of(run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(name="kw", vector_type="dense", metric="dot", k=4,
                             filter=PHRASE_FILTER,
                             rows={"column": "owner", "isin": ["x", "y"]})],
    ))["kw"])

    assert seen and all(s[0] == 3 for s in seen), (
        f"a full-cover selector should collapse to file height 3, got {seen}"
    )
    assert set(res) == {"q0", "q1", "q2"}
    assert all("charlie" in texts[int(c[1:])] for c in res["q0"][0])
    assert all("delta" in texts[int(c[1:])] for c in res["q1"][0])
    assert all("delta" in texts[int(c[1:])] for c in res["q2"][0])


# ==========================================================================
# 14. randomized differential
# ==========================================================================
def _random_case(rng, tmp_path, tag):
    """One random (corpus, queries, searches) triple.

    Deliberately samples the awkward corners rather than avoiding them:
    overlapping subsets between specs sharing a filter, subsets that happen to
    cover every row (which collapse the height back to None), specs with no
    `rows` at all, unfiltered specs mixed in (flipping `has_baseline`), and
    foreign rows that hold REAL values as often as sentinels (the only case
    where the corpus-row union actually differs).
    """
    n_files = int(rng.integers(1, 4))
    per_file = int(rng.integers(2, 7))
    n_c = n_files * per_file
    cdir = tmp_path / f"corpus_{tag}"
    cdir.mkdir()
    texts = [f"d{i} " + " ".join(rng.choice(WORDS, size=int(rng.integers(1, 3)),
                                            replace=False))
             for i in range(n_c)]
    tenants = [str(rng.choice(TENANTS)) for _ in range(n_c)]
    # identical vectors sometimes, so exact ties (and the tie-break) are in play
    if rng.random() < 0.3:
        dense_c = np.full((n_c, RAND_DIM), 0.5, dtype=np.float32)
    else:
        dense_c = rng.standard_normal((n_c, RAND_DIM)).astype(np.float32)
    for fi in range(n_files):
        lo, hi = fi * per_file, (fi + 1) * per_file
        _write(cdir / f"f{fi}.parquet", dense_c[lo:hi],
               id=[f"c{i}" for i in range(lo, hi)],
               text=texts[lo:hi], tenant=tenants[lo:hi])

    n_q = int(rng.integers(3, 10))
    owners = [str(rng.choice(["p", "q", "r"])) for _ in range(n_q)]
    # foreign-vs-real is not knowable per row here (a row is foreign only
    # relative to a given spec), so just make values real most of the time and
    # sentinel/null the rest — both conventions appear in the same column.
    def phrase_val():
        r = rng.random()
        if r < 0.15:
            return ""
        if r < 0.25:
            return None
        return " ".join(rng.choice(WORDS, size=int(rng.integers(1, 3)), replace=False))

    phrases = [phrase_val() for _ in range(n_q)]
    wants = [None if rng.random() < 0.15 else str(rng.choice(TENANTS))
             for _ in range(n_q)]
    qpath = tmp_path / f"q_{tag}.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((n_q, RAND_DIM)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(n_q)]),
        "owner": pa.array(owners),
        "phrase": pa.array(phrases),
        "tenant_want": pa.array(wants),
    }), str(qpath))

    text_only = Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase")])
    text_plus = Filter(
        must=[FilterCondition(field="text", match_text_from_query="phrase")],
        should=[FilterCondition(field="tenant", match_from_query="tenant_want"),
                FilterCondition(field="tenant", match="A")])
    gpu_only = Filter(must=[
        FilterCondition(field="tenant", match_from_query="tenant_want")])
    choices = [text_only, text_plus, gpu_only, None]

    present = sorted(set(owners))
    searches, used = [], set()
    for i in range(int(rng.integers(1, 4))):
        f = choices[int(rng.integers(0, len(choices)))]
        # a subset of the owner values that actually occur (so it never
        # matches nothing, which config rejects); sometimes all of them
        k_owners = int(rng.integers(1, len(present) + 1))
        picked = sorted(rng.choice(present, size=k_owners, replace=False).tolist())
        rows = None if rng.random() < 0.25 else {"column": "owner", "isin": picked}
        name = f"s{i}"
        if name in used:
            continue
        used.add(name)
        searches.append(SearchSpec(
            name=name, vector_type="dense",
            metric=str(rng.choice(["dot", "cosine"])),
            k=int(rng.integers(2, 6)), filter=f, rows=rows,
        ))
    if not searches:
        searches = [SearchSpec(name="s0", vector_type="dense", metric="dot", k=3)]

    # Drawn ONCE, here — not inside `cfg`. Every call to `cfg` must produce a
    # byte-identical config apart from the output directory: if the two arms
    # of the differential got different batch sizes, their matmul shapes would
    # differ, scores would move by a ULP, and the harness would report its own
    # nondeterminism as a defect in the code under test.
    params = ParamsConfig(
        dense_batch_size=int(rng.integers(1, 6)),
        io_workers=int(rng.integers(1, 4)),
    )

    def cfg(out_name):
        out = tmp_path / out_name
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            params=params,
            searches=searches,
        )

    return cfg


@pytest.mark.parametrize("seed", range(40))
def test_randomized_differential_narrowed_vs_full_height(tmp_path, monkeypatch, seed):
    """Random shape, run twice, must agree bit-for-bit.

    Only the mask height differs between the two runs — the query and score
    matrices keep identical shapes — so f32 accumulation order is unchanged
    and exact equality is the right bar, ties included.
    """
    rng = np.random.default_rng(seed)
    cfg = _random_case(rng, tmp_path, f"s{seed}")

    narrowed = {n: _rows_of(p) for n, p in run_compute(cfg(f"narrow_{seed}")).items()}
    _force_full_height(monkeypatch)
    full = {n: _rows_of(p) for n, p in run_compute(cfg(f"full_{seed}")).items()}
    monkeypatch.undo()

    assert set(narrowed) == set(full)
    for name in narrowed:
        assert set(narrowed[name]) == set(full[name]), f"seed {seed}, {name}: query set"
        for q in narrowed[name]:
            assert narrowed[name][q][0] == full[name][q][0], \
                f"seed {seed}, {name}/{q}: hit_ids\n {narrowed[name][q][0]}\n {full[name][q][0]}"
            assert narrowed[name][q][1] == full[name][q][1], \
                f"seed {seed}, {name}/{q}: hit_scores"


def test_the_random_harness_covers_the_shapes_it_claims_to(tmp_path, monkeypatch):
    """A generator that only ever emitted one trivial shape would make the 40
    cases above worthless. Count what the seeds actually produce."""
    saw = {"narrowed_mask": 0, "full_height_mask": 0, "no_cpu_mask": 0,
           "multi_spec": 0, "shared_filter": 0, "overlapping_subsets": 0,
           "unfiltered_present": 0, "union_differs": 0}
    for seed in range(40):
        rng = np.random.default_rng(seed)
        cfg_fn = _random_case(rng, tmp_path, f"probe{seed}")
        cfg = cfg_fn(f"probe_out{seed}")
        specs = cfg.searches
        if len(specs) > 1:
            saw["multi_spec"] += 1
        filters = [s.filter for s in specs]
        if len(filters) != len(set(filters)):
            saw["shared_filter"] += 1
        if any(s.filter is None for s in specs):
            saw["unfiltered_present"] += 1
        sel = [set(s.rows.isin) for s in specs if s.rows is not None]
        if any(a & b for i, a in enumerate(sel) for b in sel[i + 1:]):
            saw["overlapping_subsets"] += 1

        heights = []
        real = compute_mod._pack_query_axis
        monkeypatch.setattr(compute_mod, "_pack_query_axis",
                            lambda m: (heights.append(m.shape[0]), real(m))[1])
        run_compute(cfg)
        monkeypatch.undo()
        n_q = pq.read_table(cfg.queries.path).num_rows
        if not heights:
            saw["no_cpu_mask"] += 1
        elif any(h < n_q for h in heights):
            saw["narrowed_mask"] += 1
        else:
            saw["full_height_mask"] += 1

    for key in ("narrowed_mask", "full_height_mask", "no_cpu_mask",
                "multi_spec", "unfiltered_present"):
        assert saw[key] >= 2, f"harness produced only {saw[key]} case(s) of {key}: {saw}"


# ==========================================================================
# 15. distributed compute + merge
# ==========================================================================
def test_narrowed_filter_survives_distributed_ranks_and_merge(tmp_path):
    """The real run is 64-way. `filter_rows` is derived from the queries file,
    which every rank reads identically, so every rank's partial should cover
    the same query set at the same mask height — and merging them must equal
    the single-worker answer."""
    rng = np.random.default_rng(4242)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n_files, per = 4, 5
    texts, n = [], 0
    for fi in range(n_files):
        ts = [f"f{fi}d{i} " + WORDS[(fi + i) % len(WORDS)] for i in range(per)]
        _write(cdir / f"f{fi}.parquet",
               rng.standard_normal((per, RAND_DIM)).astype(np.float32),
               id=[f"c{n + i}" for i in range(per)], text=ts,
               tenant=[TENANTS[i % 3] for i in range(per)])
        texts += ts
        n += per

    owner = ["kw", "other", "kw", "other", "kw"]
    phrase = ["charlie", "delta", "echo", "foxtrot", "delta"]
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((5, RAND_DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(5)], owner=owner, phrase=phrase)

    text_filter = Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase")])

    def build(out_name):
        out = tmp_path / out_name
        out.mkdir(exist_ok=True)
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            searches=[SearchSpec(name="kw", vector_type="dense", metric="dot", k=5,
                                 filter=text_filter,
                                 rows={"column": "owner", "isin": ["kw"]})],
        )

    solo = _rows_of(run_compute(build("solo"))["kw"])

    dist_cfg = build("dist")
    for rank in range(2):
        run_compute(dist_cfg, num_jobs=2, job_rank=rank)
    merged = _rows_of(run_merge(dist_cfg)["kw"])

    assert set(merged) == set(solo) == {"q0", "q2", "q4"}
    for q in merged:
        assert merged[q][0], f"{q}: no hits to compare"
        assert merged[q][0] == solo[q][0], f"{q}: hit_ids differ after merge"
        assert merged[q][1] == pytest.approx(solo[q][1]), f"{q}: hit_scores differ"
    # the filter really bit, per query
    for q, word in [("q0", "charlie"), ("q2", "echo"), ("q4", "delta")]:
        assert all(word in texts[int(c[1:])] for c in merged[q][0])


# ==========================================================================
# 16. the has_baseline seam
# ==========================================================================
def test_one_mask_read_through_both_compacted_and_uncompacted_batches(tmp_path,
                                                                      monkeypatch):
    """A filter shared by a DENSE spec whose vector_type also has an
    unfiltered spec (has_baseline -> whole file, `true_rows` spans every row)
    and a SPARSE spec whose vector_type does not (compacted to `_union_keep`'s
    subset, `true_rows` is a strict subset). One `keeps[f]` entry, two
    different row-axis slicings, one narrowed query axis — the two axes have
    to stay independent."""
    rng = np.random.default_rng(808)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n_c = 10
    texts = [f"d{i} " + WORDS[i % len(WORDS)] for i in range(n_c)]
    idx = [sorted(rng.choice(8, size=3, replace=False).tolist()) for _ in range(n_c)]
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((n_c, RAND_DIM)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32())),
        "sparse_embedding": pa.array(
            [{"indices": i, "values": rng.standard_normal(3).astype(np.float32).tolist()}
             for i in idx],
            type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                            pa.field("values", pa.list_(pa.float32()))])),
        "id": pa.array([f"c{i}" for i in range(n_c)]),
        "text": pa.array(texts),
    }), str(cdir / "f0.parquet"))

    owner = ["d", "s", "none", "d", "s"]
    phrase = ["charlie", "delta", "echo", "echo", "foxtrot"]
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((5, RAND_DIM)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32())),
        "sparse_embedding": pa.array(
            [{"indices": sorted(rng.choice(8, size=3, replace=False).tolist()),
              "values": rng.standard_normal(3).astype(np.float32).tolist()}
             for _ in range(5)],
            type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                            pa.field("values", pa.list_(pa.float32()))])),
        "qid": pa.array([f"q{i}" for i in range(5)]),
        "owner": pa.array(owner),
        "phrase": pa.array(phrase),
    }), str(qpath))

    shared = Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase")])

    def build(name):
        out = tmp_path / name
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            searches=[
                # dense: one filtered + one UNFILTERED => has_baseline True
                SearchSpec(name="d", vector_type="dense", metric="dot", k=4,
                           filter=shared, rows={"column": "owner", "isin": ["d"]}),
                SearchSpec(name="plain", vector_type="dense", metric="dot", k=4),
                # sparse: only filtered => has_baseline False => compaction
                SearchSpec(name="s", vector_type="sparse", metric="dot", k=4,
                           filter=shared, rows={"column": "owner", "isin": ["s"]}),
            ],
        )

    seen = []
    real = compute_mod._pack_query_axis
    monkeypatch.setattr(compute_mod, "_pack_query_axis",
                        lambda m: (seen.append(m.shape[0]), real(m))[1])
    narrowed = {n: _rows_of(p) for n, p in run_compute(build("seam_narrow")).items()}
    monkeypatch.undo()
    assert seen and set(seen) == {4}, (
        f"one shared mask spanning both specs' 4 rows expected, got {seen}"
    )

    _force_full_height(monkeypatch)
    full = {n: _rows_of(p) for n, p in run_compute(build("seam_full")).items()}
    monkeypatch.undo()

    assert set(narrowed["d"]) == {"q0", "q3"}
    assert set(narrowed["s"]) == {"q1", "q4"}
    assert set(narrowed["plain"]) == {f"q{i}" for i in range(5)}
    for name in ("d", "s", "plain"):
        for q in narrowed[name]:
            assert narrowed[name][q][0] == full[name][q][0], f"{name}/{q}: hit_ids"
            assert narrowed[name][q][1] == full[name][q][1], f"{name}/{q}: hit_scores"
    # each filtered spec really got its own phrase
    assert all("charlie" in texts[int(c[1:])] for c in narrowed["d"]["q0"][0])
    assert all("echo" in texts[int(c[1:])] for c in narrowed["d"]["q3"][0])
    assert all("delta" in texts[int(c[1:])] for c in narrowed["s"]["q1"][0])
    assert all("foxtrot" in texts[int(c[1:])] for c in narrowed["s"]["q4"][0])


# ==========================================================================
# 17. the diagnostic line reports one entry per MASK, not per spec
# ==========================================================================
def test_mask_height_log_counts_each_mask_once(tmp_path, caplog):
    """The line is read as a memory diagnostic, so two specs sharing one
    filter must appear as one shared allocation, not two."""
    import logging

    rng = np.random.default_rng(31337)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    texts = [f"d{i} " + WORDS[i % len(WORDS)] for i in range(8)]
    _write(cdir / "f0.parquet", rng.standard_normal((8, RAND_DIM)).astype(np.float32),
           id=[f"c{i}" for i in range(8)], text=texts)
    qpath = tmp_path / "q.parquet"
    _write(qpath, rng.standard_normal((6, RAND_DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(6)],
           owner=["a", "b", "none", "a", "b", "none"],
           phrase=["charlie", "delta", "", "echo", "foxtrot", ""])

    shared = Filter(must=[
        FilterCondition(field="text", match_text_from_query="phrase")])
    out = tmp_path / "logout"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="a", vector_type="dense", metric="dot", k=3,
                       filter=shared, rows={"column": "owner", "isin": ["a"]}),
            SearchSpec(name="b", vector_type="dense", metric="dot", k=3,
                       filter=shared, rows={"column": "owner", "isin": ["b"]}),
        ],
    )
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        run_compute(cfg)

    lines = [r.getMessage() for r in caplog.records
             if "per-query filter mask height" in r.getMessage()]
    assert len(lines) == 1, f"expected one summary line, got {lines}"
    msg = lines[0]
    # one shared mask of height 4 (2 rows each), named for both sharers
    assert "a+b=4" in msg, msg
    assert "a=4, b=4" not in msg, f"double-counted the shared mask: {msg}"


# (file rows a spec owns, the filter's row union) — the shapes `filter_rows`
# can actually produce, including the two that collapse to "full height".
CASES = {
    "sole_owner_contiguous":      ([2, 3, 4],        [2, 3, 4]),
    "sole_owner_gapped":          ([1, 4, 7],        [1, 4, 7]),
    "sole_owner_single":          ([5],              [5]),
    "shares_prefix_of_union":     ([0, 1],           [0, 1, 2, 3]),
    "shares_suffix_of_union":     ([2, 3],           [0, 1, 2, 3]),
    "shares_interleaved":         ([0, 2],           [0, 1, 2, 3]),
    "shares_gapped_union":        ([1, 6],           [1, 3, 6, 9]),
    "full_height_union_is_none":  ([1, 4, 7],        None),
    "full_height_single_row":     ([3],              None),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_selector_reads_the_same_rows_from_a_narrowed_mask(case):
    """The identity the whole change rests on, over a synthetic mask whose
    every row is distinguishable."""
    rows = np.array(CASES[case][0], dtype=np.int64)
    fr = CASES[case][1]
    filter_rows = None if fr is None else np.array(fr, dtype=np.int64)

    n_file, n_cols = 12, 5
    rng = np.random.default_rng(len(case))
    file_mask = rng.random((n_file, n_cols)) < 0.5

    height = n_file if filter_rows is None else len(filter_rows)
    narrow_mask = file_mask if filter_rows is None else file_mask[filter_rows]
    assert narrow_mask.shape[0] == height

    sel = _row_selector(_local_positions(rows, filter_rows), "cpu")
    assert sel is not None, "a spec with `rows` always gets a real selector"
    got = torch.from_numpy(narrow_mask)[sel].numpy()

    np.testing.assert_array_equal(
        got, file_mask[rows],
        err_msg=f"{case}: selector did not read the spec's own rows",
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_selector_type_contract_holds_for_the_new_inputs(case):
    """`None` | `slice` | contiguous int64 tensor on the requested device —
    what the CUDA path indexes with. A float or int32 index tensor, or a
    non-contiguous one, is the kind of thing that works on CPU and misbehaves
    (or silently copies) on device."""
    rows = np.array(CASES[case][0], dtype=np.int64)
    fr = CASES[case][1]
    filter_rows = None if fr is None else np.array(fr, dtype=np.int64)

    pos = _local_positions(rows, filter_rows)
    assert pos.dtype == np.int64, f"{case}: positions must be int64, got {pos.dtype}"
    assert pos.min() >= 0
    height = len(rows) if filter_rows is None else len(filter_rows)
    if filter_rows is not None:
        assert pos.max() < height, f"{case}: position {pos.max()} outside height {height}"

    sel = _row_selector(pos, "cpu")
    if isinstance(sel, slice):
        # only for one gap-free ascending run, and it must be equivalent
        np.testing.assert_array_equal(np.arange(sel.start, sel.stop), pos)
    else:
        assert isinstance(sel, torch.Tensor), f"{case}: unexpected {type(sel)}"
        assert sel.dtype == torch.int64, f"{case}: index dtype {sel.dtype}"
        assert sel.is_contiguous()
        assert sel.device.type == "cpu"
        np.testing.assert_array_equal(sel.numpy(), pos)


def test_positions_are_in_range_for_every_union_the_helper_can_build():
    """Randomized: whatever subsets `_union_rows_by_key` pools, every spec's
    positions must land inside the resulting height. An out-of-range position
    is an IndexError on CPU but can be an out-of-bounds read on device."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_q = int(rng.integers(2, 20))
        n_specs = int(rng.integers(1, 4))
        subsets = []
        for _ in range(n_specs):
            size = int(rng.integers(1, n_q + 1))
            subsets.append(np.sort(rng.choice(n_q, size=size, replace=False)).astype(np.int64))
        union = _union_rows_by_key(["f"] * n_specs, subsets, n_q)["f"]
        height = n_q if union is None else len(union)
        for rows in subsets:
            pos = _local_positions(rows, union)
            assert pos.min() >= 0 and pos.max() < height, (
                f"positions {pos} outside height {height} (union={union}, rows={rows})"
            )
            # and they must name the spec's own rows, not merely fit
            if union is not None:
                np.testing.assert_array_equal(union[pos], rows)
