"""THE tokenizer for `match_text`/`match_text_from_query` — one definition,
used by both sides of every text match.

Semantics are Qdrant's `word` tokenizer with `lowercase: true`, including its
ORDER of operations: split the ORIGINAL text into maximal alphanumeric runs
first, then lowercase each token (split-then-lower — token boundaries never
depend on case mapping). A row matches a phrase iff every one of the phrase's
tokens is one of the row's tokens.

Both the corpus-column path (`filters._token_row_masks`) and the query-side
helpers here run the SAME Arrow kernels (`split_pattern_regex` on
`TOKEN_SPLIT_PATTERN`, then `utf8_lower`), so corpus and query tokens agree
BY CONSTRUCTION — there is no cross-implementation equivalence to maintain.
(An earlier draft tokenized queries with Python `str.lower`/`str.isalnum`;
Python and Arrow disagree on thousands of codepoints — Unicode-version skew
in what counts as alphanumeric, and full vs simple case mapping, e.g. Python
lowers word-final Greek Σ to ς where Arrow gives σ, and `İ` to `i`+U+0307
where Arrow gives `i` — so query tokens could silently never match corpus
tokens. Routing every string through the same kernels makes that entire
failure class unrepresentable.)

Known residual divergence vs Qdrant itself: Arrow's `utf8_lower` applies the
simple per-codepoint case mapping while Qdrant (Rust `to_lowercase`) applies
the full Unicode mapping — they differ on a handful of codepoints (Greek
word-final Σ, `İ`, and ~50 others). Tokens containing those characters can
match differently than a real Qdrant index; everything else is exact (see
the live-Qdrant parity harness results in the PR).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

# Maximal runs of anything OUTSIDE Unicode letters/numbers separate tokens —
# RE2's classes for the same "alphanumeric" definition Qdrant's word
# tokenizer splits on (underscores, punctuation, symbols, and marks all
# separate; splitting happens BEFORE lowercasing, see module docstring).
TOKEN_SPLIT_PATTERN = r"[^\p{L}\p{N}]+"


def tokenize_many(values) -> list[list[str] | None]:
    """Token lists for a whole sequence of strings in one set of Arrow kernel
    calls — per-string round-trips would dominate on query-log-scale inputs.
    A non-string entry (`None`, `nan`, anything else) yields `None` (the
    "matches nothing" convention for null per-query values); leading/trailing
    separators' empty-string split artifacts are dropped, so a string with no
    alphanumeric characters yields `[]`."""
    clean = [v if isinstance(v, str) else None for v in values]
    toks = pc.split_pattern_regex(
        pa.array(clean, type=pa.large_string()), pattern=TOKEN_SPLIT_PATTERN,
    )
    lowered = pc.utf8_lower(pc.list_flatten(toks)).to_pylist()
    lengths = pc.list_value_length(toks).to_pylist()
    out: list[list[str] | None] = []
    i = 0
    for n in lengths:
        if n is None:
            out.append(None)
        else:
            out.append([t for t in lowered[i:i + n] if t])
            i += n
    return out


def tokenize(text: str) -> list[str]:
    """Single-string convenience over `tokenize_many` — config validation and
    static `match_text` phrases; hot paths use the batched form."""
    toks = tokenize_many([text])[0]
    return toks if toks is not None else []
