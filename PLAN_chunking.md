# Plan: recursive and semantic text chunking in nova-embed

## TL;DR

nova-embed already owns a chunking framework — `chunkers/` with a `strategy`-keyed
registry, `passthrough` and `fixed_char` strategies, row fan-out with column
replication in `sources/base.py`, and a stubbed `SemanticChunker` (issue #13) —
so both features are new `Chunker` subclasses plus a small amount of
cross-cutting plumbing, not a new pipeline stage. The plan:

1. **`recursive` strategy** — separator-hierarchy splitting (paragraph → line →
   sentence → word → hard window) with greedy merge to a size budget and
   optional overlap. Pure Python, deterministic, no new deps. Section-based
   splitting (markdown headings) is a separator preset on the same strategy,
   not a second strategy.
2. **`semantic` strategy** — fill in the `semantic.py` stub: sentence split →
   embed sentences with a small dedicated model → break at adjacent-similarity
   valleys (per-doc percentile threshold) → enforce min/max size with a
   recursive-split fallback for oversized groups.
3. **Cross-cutting** — a batched `chunk_batch()` API so the semantic model isn't
   called one document at a time; chunk lineage columns (`chunk_index`,
   `chunk_count`, `parent_doc_id`) since nothing today links sibling chunks back
   to their source document.

No config-schema work: `ChunkingConfig` is `extra="allow"` and passes kwargs
straight to the constructor (`config.py:54-70`), and any new strategy is
automatically "splitting" (`NON_SPLITTING_STRATEGIES = {"passthrough"}`,
`config.py:40`), so the existing single-text-input-column validation applies
unchanged.

## 1. Current state (as read from the code)

**The seam already exists.** `Chunker.chunk(text) -> list[str]`
(`python/nova-embed/src/nova_embed/chunkers/base.py:16`) runs in `iter_chunks()`
between the source and the embed workers (`sources/base.py:152-154`): the split
column is rewritten to each piece and **all other columns are replicated** across
the fanned-out `Record`s. Chunking is deliberately model-agnostic and owned
outside the embedders (issue #12) so dense + sparse entries in one run see
identical pieces. Config validation guarantees a splitting chunker implies
exactly one text input column, so the target column is never ambiguous.

**What exists:** `passthrough` (default), `fixed_char` (character windows +
overlap, `chunkers/fixed_char.py:30-34`), and a `semantic` stub that raises
`NotImplementedError` with a pointer to issue #13 (`chunkers/semantic.py:15-19`).
Registration is one decorator + one import: `@CHUNKERS.register("name")`
(`registry.py:155`) and `chunkers/__init__.py`.

**Truncation is downstream and separate.** Per-entry `max_length` is a
*character* truncation in the engine (`embedders/engine.py:165`), and each
backend additionally truncates at its token limit
(`backends/sentence_transformer.py:86`, `backends/bge_m3.py:80`). The chunker
contract explicitly does NOT guarantee token fit (`chunkers/base.py:8-12`) —
oversized pieces get silently truncated at embed time. Both new strategies keep
that contract; they just make it easy to stay under budget.

**Identity is positional, lineage is absent.** Point IDs are derived at load
time from physical parquet position — `vf_point_id(filename, row)`
(`crates/nova-load/src/engine.rs:256-263`, mirrored in
`python/nova-bf/src/nova_bf/ids.py:16-21`) — so every chunk row automatically
gets a unique, GT-consistent point id with zero work. But nothing records which
source document a chunk came from or its position within it. The optional
provenance columns (`source_file_name`/`source_row_number`, gated by
`pipeline.include_source_provenance`, `sources/huggingface.py:26-28`) identify
the source *row*, which is the natural parent key.

**Call pattern:** `iter_chunks()` calls `chunker.chunk()` synchronously, one
record at a time, on the source thread. Fine for string slicing; a problem for
anything model-backed (see §2.4/§2.5).

## 2. Design questions, answered

### 2.1 One recursive strategy or separate "line" and "section" strategies?

One strategy, `recursive`, with a configurable separator hierarchy and named
presets. Line-based, paragraph-based, and section-based splitting are all the
same algorithm with different separator lists; separate registry entries would
triplicate the merge/overlap/fallback logic. Presets keep configs readable:

- `preset: text` (default): `["\n\n", "\n", ". ", " "]`
- `preset: markdown`: `["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]`,
  `keep_separator: start` so headings stay attached to their section.

An explicit `separators:` list overrides the preset. `fixed_char` stays as-is
(it's the degenerate case and the recursion's final fallback).

### 2.2 Characters or tokens for the size budget?

Characters by default, matching `fixed_char` and the engine's `max_length`
semantics — model-agnostic, zero-cost, and consistent with the existing
"chunker does not guarantee token fit" contract. Add an *optional* token mode
later (`length_unit: tokens`, `tokenizer: <hf-name>`) only if char budgets prove
too sloppy in practice; it costs a tokenizer pass per document and couples the
chunker to a specific tokenizer, which cuts against the issue-#12 "every model
sees the same pieces" ownership argument. Rule of thumb to document: budget
`chunk_chars ≈ 3–4 × target token budget` and keep it under the smallest
per-entry `max_length` in the run. Cheap guardrail worth adding: warn at launch
when `chunk_chars` exceeds any entry's `max_length` (the truncation at
`engine.py:165` would otherwise silently undo the chunker's work).

### 2.3 Build or take a dependency (langchain-text-splitters / chonkie / llama-index)?

Build. The recursive splitter is ~100 lines of dependency-free, deterministic
Python; the semantic chunker's novel part (breakpoint detection) is ~50 lines of
numpy on top of an embedding call, and sentence-transformers is already a
nova-embed dependency. The libraries bring transitive weight, their own
tokenization opinions, and version churn, while the algorithms are stable and
tiny. Determinism matters here more than in a typical RAG stack: chunk
boundaries feed brute-force ground truth, and reproducing a GT run must not
depend on a third-party splitter's minor version.

### 2.4 What model powers semantic chunking, and where does it run?

A **dedicated, configurable sentence-embedding model, defaulting to something
CPU-fast** — not the run's target embedder. Reasons: (a) issue #12's invariant —
chunk boundaries must be identical across all entries in a run, so they can't
depend on which embedder is being run; (b) the chunker runs on the source thread
while embed workers own the GPU — a big chunking model would contend for it;
(c) boundary detection only needs *relative* similarity between adjacent
sentences, which small models do fine.

Default `model: sentence-transformers/all-MiniLM-L6-v2` with `device: cpu` and
internal batching; both configurable, so a GPU can be pointed at it when it's
idle during the source phase. If CPU throughput becomes the bottleneck, a
static-embedding model (model2vec/potion, ~100–500× faster than MiniLM on CPU)
is a drop-in upgrade behind the same config knob — noted as a follow-up, not a
hard dependency now.

Environment note: the nova-embed venv pins transformers 5.x, which crashes
gte's `trust_remote_code` modeling — do not default (or recommend) a gte model
for the chunking role.

### 2.5 Per-document `chunk()` or a batched API?

Add `chunk_batch(texts: list[str]) -> list[list[str]]` to the base class with a
default implementation of `[self.chunk(t) for t in texts]`, and have
`iter_chunks()` buffer a configurable number of raw records (default ~64) before
splitting when a splitting chunker is active. Passthrough/fixed_char/recursive
behavior is bit-identical (the default just loops); `SemanticChunker` overrides
it to embed sentences *across* documents in one forward pass, which is the
difference between the model seeing batches of ~3 sentences (one short doc at a
time) and batches of hundreds. Without this, semantic chunking at any real
corpus size is throughput-dead on arrival. The buffering loop is a small,
self-contained change to `iter_chunks()` (`sources/base.py:152-166`); empty-row
policy handling stays where it is, upstream of the buffer.

### 2.6 Parent-document lineage

Today, chunks are only distinguishable by physical row position; sibling chunks
share no key. When a splitting chunker is active, `iter_chunks()` should stamp
each fanned-out record with:

- `chunk_index` (int, 0-based) and `chunk_count` (int) — always, cheap, local.
- `parent_doc_id` (string) — `"{source_file_name}:{source_row_number}"`, only
  when `pipeline.include_source_provenance` is on (that's where those values
  already exist). Recommend (and document) enabling provenance whenever
  chunking, rather than inventing a second provenance mechanism.

These are ordinary pass-through columns: the writer infers their types, and the
loader exposes them via `datasource.payload_fields` → Qdrant payload, which is
what enables doc-level dedup/aggregation of chunk hits downstream (out of scope
here, but this unblocks it). Validate at launch that none of the three names
collide with existing source columns. Existing runs without chunking are
untouched — the columns only appear when a splitting chunker is configured.

### 2.7 How does this interact with nova-bf / nova-load?

It mostly doesn't, by design — a chunked corpus is just a corpus with more rows.
`vf_point_id` keys off physical position, so GT ids and loaded point ids stay
consistent with zero changes to nova-bf or nova-load. Two things to be aware of
and document: (a) top-K GT over a chunked corpus is chunk-level top-K — multiple
chunks of one document can occupy the list, and any doc-level metric needs the
`parent_doc_id` payload; (b) re-chunking a corpus with different parameters
produces a *different* corpus — GT, collections, and configs must not be mixed
across chunking parameter sets (the output path should encode the chunking
choice, same convention as embedding-model variants).

## 3. Recommended design

### 3.1 `RecursiveChunker` (`chunkers/recursive.py`)

```yaml
chunking:
  strategy: recursive
  chunk_chars: 2000        # size budget per piece
  overlap: 0               # trailing chars of piece i prepended to piece i+1
  preset: text             # text | markdown (separator hierarchy)
  separators: null         # explicit list overrides preset
  keep_separator: end      # end | start | none
  min_chunk_chars: 200     # merge-forward pieces smaller than this
```

Algorithm (the standard recursive scheme, stated precisely so tests can pin it):

1. Split on the first separator in the hierarchy that occurs in the text
   (attaching separators per `keep_separator`).
2. Greedily merge adjacent fragments while the merged length ≤ `chunk_chars`.
3. Any single fragment still over budget recurses with the *rest* of the
   separator hierarchy; the final fallback is a hard character window (reuse
   `FixedCharChunker`'s logic).
4. Apply `min_chunk_chars` by merging a too-small trailing piece into its
   predecessor (never emit it alone unless it's the only piece).
5. Apply `overlap` at the end, across final piece boundaries.

Constructor validation mirrors `fixed_char` (positive sizes,
`0 ≤ overlap < chunk_chars`, non-empty separator list, known preset). Invariant
the tests enforce: with `overlap: 0` and `keep_separator: end`, concatenating
the pieces reproduces the input exactly.

### 3.2 `SemanticChunker` (fill in `chunkers/semantic.py`, closes issue #13)

```yaml
chunking:
  strategy: semantic
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu              # or cuda / cuda:N
  batch_size: 256          # sentence-embedding batch
  breakpoint_method: percentile   # percentile | std_dev | absolute
  breakpoint_value: 90     # split where adjacent distance > P90 (per document)
  window: 1                # compare sentence i against mean of previous `window`
  min_chunk_chars: 200
  max_chunk_chars: 4000    # oversized groups re-split via recursive fallback
```

Pipeline per batch of documents (via `chunk_batch`):

1. **Sentence split** — small regex-based splitter in `chunkers/sentences.py`
   (terminal punctuation + whitespace, with newline as a boundary). Web text
   (fineweb) is messy; the splitter just needs to be deterministic and roughly
   right — boundaries are refined by the embedding step anyway. No NLP dep.
2. **Embed** — all sentences from all docs in the batch, one batched forward
   pass on the configured model, L2-normalized.
3. **Breakpoints** — per document: cosine distance between each sentence (or the
   mean of the trailing `window` sentences) and the next; split where distance
   exceeds the per-document percentile threshold (`std_dev` and `absolute` as
   alternates). Documents with < 3 sentences pass through unsplit.
4. **Size enforcement** — merge adjacent groups under `min_chunk_chars`; any
   group over `max_chunk_chars` is re-split with an internal `RecursiveChunker`
   (so the token-fit story is no worse than recursive).

Model load is lazy (first `chunk_batch` call), keeping config validation and
`--help` paths import-cheap. Determinism note for GT reproducibility: fixed
model revision + CPU inference is bit-stable; GPU inference can flip borderline
breakpoints across hardware — document that GT-feeding runs should pin the
chunking device.

### 3.3 Shared plumbing

- `Chunker.chunk_batch()` default + `iter_chunks()` record buffering (§2.5),
  buffer size from a new `pipeline.chunk_buffer_records` (default 64; 1 keeps
  today's exact behavior).
- Lineage columns (§2.6) stamped in `iter_chunks()` at fan-out time
  (`sources/base.py:154` is the single place records are created from pieces).
- Launch-time warning when `chunk_chars`/`max_chunk_chars` exceeds any entry's
  `max_length` (§2.2).
- Registration: imports in `chunkers/__init__.py`; `NON_SPLITTING_STRATEGIES`
  untouched (both new strategies are splitting, which is what the single-input
  validation should see).

## 4. Sequencing

Each step lands independently and keeps `master` green:

1. **`recursive`** — chunker + unit tests + `docs/embedding/overview.md` "Text
   splitting" section update. No plumbing dependencies; immediately usable.
2. **Lineage columns** — `iter_chunks()` stamping + collision validation +
   loading docs (`payload_fields` example). Benefits `fixed_char` and
   `recursive` right away.
3. **`chunk_batch` + buffering** — pure refactor, no behavior change for
   existing strategies (assert bit-identical output in tests).
4. **`semantic`** — sentence splitter, chunker, tests. Depends on (3) for
   throughput and reuses (1) for the oversize fallback.
5. **Smoke configs** — a small ms_marco embed config per strategy (mirroring the
   existing smoke-config convention) for end-to-end runs.

## 5. Validation

- **Unit** (pure, fast): determinism; ≥1 piece always; exact reconstruction for
  recursive with no overlap; size bounds honored; overlap content correct;
  degenerate inputs (empty string handled upstream, single char, no separators
  present, one giant unbroken token, unicode); preset contents; semantic on
  synthetic docs with an obvious topic shift (two clusters of near-duplicate
  sentences → exactly one breakpoint); `chunk_batch == [chunk(t) for t]` for
  every strategy including semantic (batching must not change boundaries).
- **Pipeline**: run the embed pipeline on a fixture with a splitting chunker;
  assert row fan-out counts, replicated columns, lineage column values, and
  parquet schema.
- **End-to-end smoke**: embed a few thousand ms_marco passages with `recursive`
  and with `semantic` → `nova load` → spot-check payloads carry
  `parent_doc_id`/`chunk_index` → `nova bf` GT run completes and ids line up.
  Also worth reporting from the smoke run: chunk-length histograms and
  chunks-per-doc distribution per strategy — the first sanity signal that
  parameters are reasonable for the corpus.

## 6. Explicitly deferred / out of scope

- **Token-budgeted splitting** (`length_unit: tokens`) — revisit if char budgets
  prove too sloppy (§2.2).
- **Static-embedding chunking model** (model2vec) — drop-in behind the existing
  `model:` knob if CPU MiniLM is the bottleneck.
- **Doc-level result aggregation/dedup** in nova-bf/nova-sweep (max-passage,
  first-chunk-wins, etc.) — retrieval-eval design of its own; unblocked by
  `parent_doc_id` but not part of chunking.
- **Chunk context enrichment** (prepending title/heading to each piece,
  "contextual retrieval"-style) — would need a post-chunk template
  (`render_columns` runs before splitting); worth a separate design if wanted.
- **Layout-aware chunking** (HTML/PDF structure) — corpora here are plain-text
  columns; the markdown preset covers the structured-text need for now.
- **Late chunking / multi-vector approaches** — different embedding contract
  entirely, not a `Chunker`.
