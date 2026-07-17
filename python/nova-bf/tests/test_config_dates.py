"""Config-level tests for date_fields: static bound normalization + validators."""

from __future__ import annotations

import textwrap

import pytest

from nova_bf.config import load_config
from nova_bf.dates import parse_scalar_epoch_us


def _cfg(tmp_path, body: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return load_config(str(p))


BASE_STORES = """
      corpus:
        path: /tmp/corpus
        {corpus_extra}
      queries:
        path: /tmp/queries.parquet
        {queries_extra}
      output:
        path: /tmp/out
"""


def test_static_bound_string_parsed_to_epoch_us(tmp_path):
    cfg = _cfg(tmp_path, """
      corpus:
        path: /tmp/corpus
        date_fields: [published_at]
      queries:
        path: /tmp/queries.parquet
      output:
        path: /tmp/out
      searches:
        - name: recent
          filter:
            must:
              - field: published_at
                range:
                  gte: "2013-01-01T00:00:00Z"
                  lt: "2014-01-01T00:00:00Z"
    """)
    rng = cfg.searches[0].filter.must[0].range
    assert rng.gte == float(parse_scalar_epoch_us("2013-01-01T00:00:00Z"))
    assert rng.lt == float(parse_scalar_epoch_us("2014-01-01T00:00:00Z"))


def test_dict_form_with_custom_format(tmp_path):
    cfg = _cfg(tmp_path, """
      corpus:
        path: /tmp/corpus
        date_fields:
          crawl_day: "%Y%m%d"
      queries:
        path: /tmp/queries.parquet
      output:
        path: /tmp/out
      searches:
        - name: s
          filter:
            must:
              - field: crawl_day
                range: {gte: "20130101"}
    """)
    rng = cfg.searches[0].filter.must[0].range
    assert rng.gte == float(parse_scalar_epoch_us("20130101", "%Y%m%d"))


def test_string_bound_on_non_date_field_rejected(tmp_path):
    with pytest.raises(ValueError, match="not declared in corpus.date_fields"):
        _cfg(tmp_path, """
          corpus:
            path: /tmp/corpus
          queries:
            path: /tmp/queries.parquet
          output:
            path: /tmp/out
          searches:
            - name: s
              filter:
                must:
                  - field: published_at
                    range: {gte: "2013-01-01T00:00:00Z"}
        """)


def test_numeric_bound_on_date_field_passes_through(tmp_path):
    # A numeric literal is treated as already-epoch µs and left as-is.
    cfg = _cfg(tmp_path, """
      corpus:
        path: /tmp/corpus
        date_fields: [published_at]
      queries:
        path: /tmp/queries.parquet
      output:
        path: /tmp/out
      searches:
        - name: s
          filter:
            must:
              - field: published_at
                range: {gte: 1356998400000000}
    """)
    assert cfg.searches[0].filter.must[0].range.gte == 1356998400000000


def test_range_from_query_requires_declared_query_date_field(tmp_path):
    with pytest.raises(ValueError, match="not declared in queries.date_fields"):
        _cfg(tmp_path, """
          corpus:
            path: /tmp/corpus
            date_fields: [published_at]
          queries:
            path: /tmp/queries.parquet
          output:
            path: /tmp/out
          searches:
            - name: s
              filter:
                must:
                  - field: published_at
                    range_from_query: {gte: after}
        """)


def test_range_from_query_date_bound_on_non_date_corpus_field_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a\\s*\\ndate field|is not a date field"):
        _cfg(tmp_path, """
          corpus:
            path: /tmp/corpus
          queries:
            path: /tmp/queries.parquet
            date_fields: [after]
          output:
            path: /tmp/out
          searches:
            - name: s
              filter:
                must:
                  - field: some_number
                    range_from_query: {gte: after}
        """)


def test_range_from_query_consistent_declaration_ok(tmp_path):
    cfg = _cfg(tmp_path, """
      corpus:
        path: /tmp/corpus
        date_fields: [published_at]
      queries:
        path: /tmp/queries.parquet
        date_fields: [after]
      output:
        path: /tmp/out
      searches:
        - name: s
          filter:
            must:
              - field: published_at
                range_from_query: {gte: after}
    """)
    assert cfg.searches[0].filter.must[0].range_from_query.gte == "after"


def test_non_date_numeric_range_from_query_still_works(tmp_path):
    # No date_fields anywhere: the ordinary numeric range_from_query path is
    # completely unaffected by the new validator.
    cfg = _cfg(tmp_path, """
      corpus:
        path: /tmp/corpus
      queries:
        path: /tmp/queries.parquet
      output:
        path: /tmp/out
      searches:
        - name: s
          filter:
            must:
              - field: cost
                range_from_query: {lte: max_budget}
    """)
    assert cfg.searches[0].filter.must[0].range_from_query.lte == "max_budget"
