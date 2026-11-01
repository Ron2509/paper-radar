//! Keyword prefilter (stage 2 of relevance filtering).
//!
//! Cheap, deterministic, no LLM: a paper survives only if its title +
//! abstract ([`Paper::text_for_filter`]) contains at least one configured
//! keyword. Case-insensitive; matches must be whole terms — the characters on
//! both sides of a hit may not be `a-z` or `0-9`, so "rust" does not match
//! "trust" or "rustling", and "dpo" does not match "dopamine".

// TODO(wiring): remove once the digest pipeline calls the prefilter; until
// then keyword_filter is only exercised by tests.
#![allow(dead_code)]

use crate::arxiv::Paper;

/// Keeps papers whose title + abstract contain at least one keyword.
///
/// Matching is case-insensitive and whole-term: every character boundary
/// around a candidate match must not be an ASCII letter or digit. A keyword
/// that itself contains digits (e.g. "gpu") still matches, as long as it is
/// not embedded in a longer alphanumeric run (so "gpus" does not match
/// "gpu"; add plural forms to the config if wanted). Papers are returned in
/// input order.
pub fn keyword_filter(papers: Vec<Paper>, keywords: &[String]) -> Vec<Paper> {
    let needles: Vec<String> = keywords
        .iter()
        .map(|k| k.trim().to_lowercase())
        .filter(|k| !k.is_empty())
        .collect();

    papers
        .into_iter()
        .filter(|paper| {
            let haystack = paper.text_for_filter().to_lowercase();
            needles
                .iter()
                .any(|needle| contains_whole_term(&haystack, needle))
        })
        .collect()
}

/// Whether `haystack` (already lowercased) contains `needle` (already
/// lowercased, non-empty) with no ASCII alphanumeric character immediately
/// before or after the match.
fn contains_whole_term(haystack: &str, needle: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let at = start + pos;
        let end = at + needle.len();

        let boundary_before_ok = at == 0 || !is_word_byte(haystack.as_bytes()[at - 1]);
        let boundary_after_ok = end == haystack.len() || !is_word_byte(haystack.as_bytes()[end]);

        if boundary_before_ok && boundary_after_ok {
            return true;
        }
        // Overlapping candidates: step past this hit's first byte.
        start = at + 1;
    }
    false
}

/// ASCII letters and digits are word characters for boundary purposes.
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal paper for filter tests.
    fn paper(id: &str, title: &str, summary: &str) -> Paper {
        Paper {
            id: id.to_string(),
            url: format!("https://arxiv.org/abs/{id}"),
            title: title.to_string(),
            summary: summary.to_string(),
            authors: vec![],
            primary_category: Some("cs.CL".to_string()),
            categories: vec!["cs.CL".to_string()],
            published: None,
        }
    }

    fn keywords(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rust_does_not_match_trust_or_rusty_but_matches_rust() {
        let kws = keywords(&["rust"]);

        let trust = vec![paper(
            "2501.00001",
            "Trust management for federated learning",
            "We study trust in multi-party settings.",
        )];
        assert!(
            keyword_filter(trust, &kws).is_empty(),
            "\"trust\" must not match the keyword \"rust\""
        );

        let rusty = vec![paper(
            "2501.00002",
            "A rusty benchmark",
            "Corroded baselines everywhere.",
        )];
        assert!(
            keyword_filter(rusty, &kws).is_empty(),
            "\"rusty\" must not match the keyword \"rust\""
        );

        let real = vec![paper(
            "2501.00003",
            "Rust ML systems",
            "An inference engine written in Rust.",
        )];
        assert_eq!(keyword_filter(real, &kws).len(), 1);
    }

    #[test]
    fn matching_ignores_case() {
        let kws = keywords(&["kv cache"]);

        let papers = vec![paper(
            "2501.00004",
            "SGLANG SERVERS",
            "We improve KV CACHE eviction policies under load.",
        )];
        let kept = keyword_filter(papers, &kws);
        assert_eq!(kept.len(), 1, "uppercase \"KV CACHE\" must match");
    }

    #[test]
    fn non_matching_papers_are_removed_and_order_is_kept() {
        let kws = keywords(&["rlhf", "tokenizer"]);

        let papers = vec![
            paper(
                "2501.00005",
                "Reward modelling at scale",
                "RLHF from weak preferences.",
            ),
            paper(
                "2501.00006",
                "Quantum error correction",
                "Surface codes and thresholds.",
            ),
            paper(
                "2501.00007",
                "A tokenizer for low-resource languages",
                "Byte-level BPE for Indic scripts.",
            ),
        ];

        let kept = keyword_filter(papers, &kws);
        let ids: Vec<&str> = kept.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["2501.00005", "2501.00007"]);
    }
}
