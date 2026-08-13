//! Datetime parsing shared by the query loader and the qdrant target.
//!
//! One parser, two callers: `queries.rs` validates (and normalizes) datetime
//! range-bound values AT LOAD TIME — so a malformed value fails the run before
//! any load is offered, instead of as a full-duration per-dispatch error loop
//! — and `targets/qdrant.rs` converts the already-validated strings to the
//! gRPC `Timestamp` at dispatch.
//!
//! Accepted forms, all verified against real writers:
//! - full RFC3339: `2017-09-19T11:23:19Z`, offsets (`+02:00`), pre-1970
//!   dates, fractional seconds, lowercase `t`/`z`
//! - DuckDB's VARCHAR renderings: `2017-09-19 11:23:19` (TIMESTAMP — space
//!   separator, no offset, read as UTC) and `2017-09-19 09:23:19+00`
//!   (TIMESTAMPTZ — hour-only offset, minutes appended)
//! - date-only `2017-09-19` (midnight UTC), the shape a DATE column renders to

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// Parse one datetime string. `Err` carries the *underlying* parse error; the
/// callers wrap it with their own field/column context.
pub fn parse_datetime_utc(raw: &str) -> Result<OffsetDateTime, String> {
    let trimmed = raw.trim();

    // date-only: a DATE column's text form. Midnight UTC.
    let candidate = if is_date_only(trimmed) {
        format!("{trimmed}T00:00:00Z")
    } else {
        trimmed.to_string()
    };

    OffsetDateTime::parse(&candidate, &Rfc3339)
        .or_else(|first_err| {
            // Normalize the space-separated forms DuckDB renders:
            //   2017-09-19 11:23:19        (TIMESTAMP -> read as UTC)
            //   2017-09-19 09:23:19+00     (TIMESTAMPTZ -> hour-only offset)
            let mut c = candidate.replacen(' ', "T", 1);
            // The separator may be lowercase `t` (RFC3339-legal); the offset
            // scan below must not mistake the date's dashes for a sign.
            let sep = c.find(['T', 't']).unwrap_or(0);
            match c.rfind(['+', '-']).filter(|&i| i > sep) {
                None => c.push('Z'), // no offset at all -> UTC
                Some(i) => {
                    // hour-only offset (`+00`, `-05`): RFC3339 wants `+00:00`
                    if c.len() - i == 3 {
                        c.push_str(":00");
                    }
                }
            }
            OffsetDateTime::parse(&c, &Rfc3339).map_err(|_| first_err)
        })
        .map_err(|e| e.to_string())
}

/// Re-render a parsed datetime as canonical RFC3339 — what the loader stores,
/// so the dispatch-time re-parse can never fail on a value the load accepted.
pub fn to_rfc3339(t: OffsetDateTime) -> Result<String, String> {
    t.format(&Rfc3339).map_err(|e| e.to_string())
}

fn is_date_only(s: &str) -> bool {
    s.len() == 10
        && s.as_bytes()[4] == b'-'
        && s.as_bytes()[7] == b'-'
        && s.bytes().enumerate().all(|(i, b)| matches!(i, 4 | 7) || b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch(s: &str) -> i128 {
        parse_datetime_utc(s).expect(s).unix_timestamp_nanos()
    }

    const T1: i128 = 1_505_820_199 * 1_000_000_000; // 2017-09-19T11:23:19Z

    #[test]
    fn accepts_all_documented_forms() {
        assert_eq!(epoch("2017-09-19T11:23:19Z"), T1);
        assert_eq!(epoch("2017-09-19 11:23:19"), T1); // duckdb TIMESTAMP text
        assert_eq!(epoch("2017-09-19 13:23:19+02"), T1); // duckdb TIMESTAMPTZ text, hour-only
        assert_eq!(epoch("2017-09-19 09:23:19-02"), T1);
        assert_eq!(epoch("2017-09-19T13:23:19+02:00"), T1); // full offset
        assert_eq!(epoch("2017-09-19t11:23:19z"), T1); // lowercase separator
        assert_eq!(epoch("2017-09-19t11:23:19"), T1); // lowercase, no offset
        assert_eq!(epoch("2017-09-19 11:23:19z"), T1); // space form, lowercase z offset
        assert_eq!(epoch("  2017-09-19T11:23:19Z  "), T1); // padding
        assert_eq!(epoch("2017-09-19"), 1_505_779_200 * 1_000_000_000); // date-only
    }

    #[test]
    fn pre_1970_and_fractional() {
        assert_eq!(epoch("1969-12-31T23:59:59.5Z"), -500_000_000);
        assert_eq!(epoch("2017-09-19T11:23:19.123Z"), T1 + 123_000_000);
    }

    #[test]
    fn garbage_is_rejected() {
        for bad in ["", "n/a", "2013/05/18", "last tuesday", "123", "2017-13-45", "11:23:19"] {
            assert!(parse_datetime_utc(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn normalization_round_trips_through_rfc3339() {
        let t = parse_datetime_utc("2017-09-19 13:23:19+02").unwrap();
        let rendered = to_rfc3339(t).unwrap();
        assert_eq!(parse_datetime_utc(&rendered).unwrap(), t);
    }
}
