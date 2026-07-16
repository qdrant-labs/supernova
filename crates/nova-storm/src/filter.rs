//! Backend-agnostic payload/metadata filter — shaped like `nova-bf`'s own
//! `Filter`/`FilterCondition` (`python/nova-bf/src/nova_bf/config.py`), so a
//! filter authored for a `nova bf` ground-truth run and a `nova storm` load
//! test read the same way. Translating this into an actual wire-level filter
//! is backend-specific (see `targets::qdrant::to_qdrant_filter`) — this module
//! only owns the config shape and its validation.
//!
//! `must` = AND, `should` = OR-at-least-one, `must_not` = AND-NOT, same as
//! Qdrant's own filter and `nova-bf`'s.

use std::collections::BTreeSet;

use serde::Deserialize;

use crate::config::ConfigError;

/// A single scalar payload value, as it would appear in a corpus/collection
/// field. `Float` is kept for parity with `nova-bf`'s `MatchValue = str | int
/// | float | bool` even though Qdrant's own match condition has no float
/// variant — a backend that can't honour it (Qdrant, today) rejects it at
/// translation time with a clear error, not here at parse time, since this
/// type isn't inherently bound to one backend.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum MatchValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// `match`'s value: a scalar (equality) or a list (matches any of them —
/// Qdrant's `MatchAny`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum MatchSpec {
    One(MatchValue),
    Any(Vec<MatchValue>),
}

/// Numeric bounds, combinable (e.g. `gte` + `lt` together).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeCondition {
    #[serde(default)]
    pub gt: Option<f64>,
    #[serde(default)]
    pub gte: Option<f64>,
    #[serde(default)]
    pub lt: Option<f64>,
    #[serde(default)]
    pub lte: Option<f64>,
}

impl RangeCondition {
    fn has_bound(&self) -> bool {
        self.gt.is_some() || self.gte.is_some() || self.lt.is_some() || self.lte.is_some()
    }
}

/// Per-query numeric bounds — same shape as [`RangeCondition`], but each bound
/// names a QUERIES column supplying that query's own value for the bound,
/// instead of a literal number (e.g. `lt: max_budget` means "each query's own
/// ceiling comes from its own `max_budget` column").
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeFromQuery {
    #[serde(default)]
    pub gt: Option<String>,
    #[serde(default)]
    pub gte: Option<String>,
    #[serde(default)]
    pub lt: Option<String>,
    #[serde(default)]
    pub lte: Option<String>,
}

impl RangeFromQuery {
    fn has_bound(&self) -> bool {
        self.gt.is_some() || self.gte.is_some() || self.lt.is_some() || self.lte.is_some()
    }

    /// Every queries column named by a bound on this condition.
    fn columns(&self) -> impl Iterator<Item = &str> {
        [&self.gt, &self.gte, &self.lt, &self.lte].into_iter().filter_map(|c| c.as_deref())
    }
}

/// One field predicate: keyword-style equality (`match`), numeric bounds
/// (`range`), full-text (`match_text`), or a per-query variant of any of the
/// three (`match_from_query`/`range_from_query`/`match_text_from_query`),
/// which pulls its comparison value(s) from a column in the queries file
/// instead of a literal in this config — so two different queries in the same
/// run can each be restricted to a different subset (e.g. each scoped to its
/// own tenant, budget, or search phrase). Exactly one of the six must be set.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterCondition {
    /// The collection field this condition reads and matches against.
    pub field: String,
    #[serde(default, rename = "match")]
    pub r#match: Option<MatchSpec>,
    #[serde(default)]
    pub range: Option<RangeCondition>,
    #[serde(default)]
    pub match_text: Option<String>,
    /// Per-query variants — see this struct's docstring. Each names a QUERIES
    /// column (not a collection field); `field` above still names the
    /// collection field every one of these compares against.
    #[serde(default)]
    pub match_from_query: Option<String>,
    #[serde(default)]
    pub range_from_query: Option<RangeFromQuery>,
    #[serde(default)]
    pub match_text_from_query: Option<String>,
}

impl FilterCondition {
    fn validate(&self) -> Result<(), ConfigError> {
        let set_count = [
            self.r#match.is_some(),
            self.range.is_some(),
            self.match_text.is_some(),
            self.match_from_query.is_some(),
            self.range_from_query.is_some(),
            self.match_text_from_query.is_some(),
        ]
        .into_iter()
        .filter(|&s| s)
        .count();
        if set_count != 1 {
            return Err(ConfigError::FilterConditionNotExactlyOne { field: self.field.clone() });
        }
        if let Some(text) = &self.match_text
            && text.trim().is_empty()
        {
            return Err(ConfigError::FilterConditionBlankMatchText { field: self.field.clone() });
        }
        if let Some(range) = &self.range
            && !range.has_bound()
        {
            return Err(ConfigError::FilterConditionEmptyRange { field: self.field.clone() });
        }
        if let Some(range) = &self.range_from_query
            && !range.has_bound()
        {
            return Err(ConfigError::FilterConditionEmptyRange { field: self.field.clone() });
        }
        Ok(())
    }

    /// Does this ONE condition vary per query?
    pub fn is_per_query(&self) -> bool {
        self.match_from_query.is_some()
            || self.range_from_query.is_some()
            || self.match_text_from_query.is_some()
    }

    /// Every queries column this condition (if per-query) reads a comparison
    /// value from.
    fn query_columns(&self) -> Vec<&str> {
        let mut cols = Vec::new();
        if let Some(c) = &self.match_from_query {
            cols.push(c.as_str());
        }
        if let Some(r) = &self.range_from_query {
            cols.extend(r.columns());
        }
        if let Some(c) = &self.match_text_from_query {
            cols.push(c.as_str());
        }
        cols
    }
}

/// A payload filter, shaped like Qdrant's own filter (must/should/must_not).
/// Restricts which points are eligible neighbours for every query in the run
/// — it does not touch queries themselves, same as a Qdrant search filter.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    #[serde(default)]
    pub must: Vec<FilterCondition>,
    #[serde(default)]
    pub should: Vec<FilterCondition>,
    #[serde(default)]
    pub must_not: Vec<FilterCondition>,
}

impl Filter {
    /// Validate every condition in this filter. Called once at config-load
    /// time (`StormConfig::from_yaml`), the same place `top_k`/`batch_size`
    /// are checked — serde has no post-parse validator hook of its own, so
    /// this stands in for what `pydantic`'s `@model_validator` does for
    /// `nova-bf`'s equivalent config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.all_conditions().try_for_each(FilterCondition::validate)
    }

    /// Every condition in this filter, across all three groups.
    fn all_conditions(&self) -> impl Iterator<Item = &FilterCondition> {
        self.must.iter().chain(self.should.iter()).chain(self.must_not.iter())
    }

    /// Every QUERIES column referenced by a per-query condition anywhere in
    /// this filter — `nova storm` reads exactly (and only) the queries
    /// columns some per-query condition actually references (see
    /// `queries::load_query_vectors`).
    pub fn query_fields(&self) -> BTreeSet<&str> {
        self.all_conditions().flat_map(FilterCondition::query_columns).collect()
    }

    /// Does any condition in this filter vary per query?
    pub fn is_per_query(&self) -> bool {
        !self.query_fields().is_empty()
    }
}

/// One `_from_query` column's resolved value for a single query — the
/// per-query analog of a literal in [`FilterCondition`]. Populated once at
/// query-load time (`queries::load_query_vectors`) alongside each
/// [`QueryVector`](crate::queries::QueryVector), then read at request-build
/// time by a backend's translation (e.g.
/// `targets::qdrant::to_qdrant_condition`).
#[derive(Debug, Clone, PartialEq)]
pub enum FilterFieldValue {
    Text(String),
    Num(f64),
    TextList(Vec<String>),
    NumList(Vec<f64>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn de(yaml: &str) -> Filter {
        serde_yaml::from_str(yaml).expect("parses")
    }

    #[test]
    fn parses_static_match_range_and_text() {
        let f = de(
            r#"
must:
  - field: category
    match: shoes
  - field: price
    range:
      gte: 10.0
      lt: 100.0
should:
  - field: description
    match_text: "waterproof hiking"
must_not:
  - field: tag
    match: [discontinued, recalled]
"#,
        );
        assert_eq!(f.must.len(), 2);
        assert_eq!(f.should.len(), 1);
        assert_eq!(f.must_not.len(), 1);
        assert_eq!(f.must[0].r#match, Some(MatchSpec::One(MatchValue::Str("shoes".into()))));
        assert!(!f.is_per_query());
        assert!(f.query_fields().is_empty());
        f.validate().expect("valid");
    }

    #[test]
    fn parses_per_query_variants_and_collects_query_fields() {
        let f = de(
            r#"
must:
  - field: text
    match_text_from_query: keyword_phrase
  - field: tenant_id
    match_from_query: tenant_column
  - field: budget
    range_from_query:
      lt: max_budget
      gte: min_budget
"#,
        );
        assert!(f.is_per_query());
        assert_eq!(
            f.query_fields(),
            BTreeSet::from(["keyword_phrase", "tenant_column", "max_budget", "min_budget"])
        );
        f.validate().expect("valid");
    }

    #[test]
    fn rejects_condition_with_zero_kinds_set() {
        let f = de("must:\n  - field: category\n");
        assert!(matches!(
            f.validate().unwrap_err(),
            ConfigError::FilterConditionNotExactlyOne { field } if field == "category"
        ));
    }

    #[test]
    fn rejects_condition_with_multiple_kinds_set() {
        let f = de("must:\n  - field: category\n    match: shoes\n    match_text: shoes\n");
        assert!(matches!(
            f.validate().unwrap_err(),
            ConfigError::FilterConditionNotExactlyOne { field } if field == "category"
        ));
    }

    #[test]
    fn rejects_blank_match_text() {
        let f = de("must:\n  - field: description\n    match_text: \"   \"\n");
        assert!(matches!(
            f.validate().unwrap_err(),
            ConfigError::FilterConditionBlankMatchText { field } if field == "description"
        ));
    }

    #[test]
    fn rejects_range_with_no_bounds() {
        let f = de("must:\n  - field: price\n    range: {}\n");
        assert!(matches!(
            f.validate().unwrap_err(),
            ConfigError::FilterConditionEmptyRange { field } if field == "price"
        ));
    }

    #[test]
    fn rejects_range_from_query_with_no_bounds() {
        let f = de("must:\n  - field: price\n    range_from_query: {}\n");
        assert!(matches!(
            f.validate().unwrap_err(),
            ConfigError::FilterConditionEmptyRange { field } if field == "price"
        ));
    }

    #[test]
    fn deny_unknown_fields_rejects_typos() {
        let result: Result<Filter, _> = serde_yaml::from_str(
            "must:\n  - field: category\n    matches: shoes\n", // typo: "matches"
        );
        assert!(result.is_err());
    }
}
