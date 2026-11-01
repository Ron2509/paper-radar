#![allow(dead_code)]
//! Keyword prefilter (stage 2 of relevance filtering).
//!
//! Cheap, deterministic, no LLM: a paper survives only if its title +
//! abstract ([`Paper::text_for_filter`]) contains at least one configured
//! keyword. Case-insensitive; matches must be whole terms — the characters on
//! both sides of a hit may not be `a-z` or `0-9`, so "rust" does not match
//! "trust" or "rustling", and "dpo" does not match "dopamine".

// TODO(wiring): remove once the digest pipeline calls the prefilter; until
// then keyword_filter is only exercised by tests.

use crate::arxiv::Paper;

/// Keeps papers whose title + abstract contain at least one keyword.
///
/// Matching is case-insensitive and whole-term: every character boundary
/// around a candidate match must not be an ASCII letter or digit, except that
/// a single trailing plural "s" is accepted ("jailbreaks" matches
/// "jailbreak", "jailbreaking" does not). Papers are returned in input order.
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
/// before or after the match. A single plural "s" right after the match is
/// tolerated ("jailbreaks" matches "jailbreak"); anything longer is not
/// ("jailbreaking" does not match).
fn contains_whole_term(haystack: &str, needle: &str) -> bool {
    // Step past the needle's first CHARACTER, not its first byte: a
    // multi-byte first character (non-ASCII keyword) would otherwise leave
    // `start` mid-character and panic on the next slice.
    let step = needle.chars().next().map_or(1, char::len_utf8);
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let at = start + pos;
        let end = at + needle.len();

        let boundary_before_ok = at == 0 || !is_word_byte(haystack.as_bytes()[at - 1]);

        // Swallow one optional plural "s". `b's'` can never be a UTF-8
        // continuation byte, so the byte check is safe.
        let plural_end = if end < haystack.len() && haystack.as_bytes()[end] == b's' {
            end + 1
        } else {
            end
        };
        let boundary_after_ok =
            plural_end == haystack.len() || !is_word_byte(haystack.as_bytes()[plural_end]);

        if boundary_before_ok && boundary_after_ok {
            return true;
        }
        // Overlapping candidates: step past this hit's first character.
        start = at + step;
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

    #[test]
    fn a_single_trailing_s_is_accepted_but_longer_suffixes_are_not() {
        let kws = keywords(&["jailbreak", "tokenizer"]);

        // Plain plural: matches.
        assert!(contains_whole_term(
            "we study jailbreaks of aligned models",
            "jailbreak"
        ));
        // Derived word with more after the stem: no match.
        assert!(!contains_whole_term(
            "jailbreaking aligned models is easy",
            "jailbreak"
        ));
        // Plural of a multi-word keyword also matches.
        assert!(contains_whole_term(
            "better tokenizers for indic scripts",
            "tokenizer"
        ));
        // And through the public API, with case folding.
        let papers = vec![
            paper("2501.00008", "Jailbreaks Galore", "Many jailbreaks found."),
            paper("2501.00009", "Other work", "Unrelated content entirely."),
        ];
        let kept = keyword_filter(papers, &kws);
        let ids: Vec<&str> = kept.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["2501.00008"]);
    }

    #[test]
    fn non_ascii_needles_do_not_panic_and_still_match() {
        // A needle whose first character is multi-byte used to advance `start`
        // by one byte, landing mid-character and panicking on the next slice.
        // With per-character stepping it must simply work.
        let needle = "मॉडल"; // "model" in Hindi
        assert!(contains_whole_term("हम मॉडल बनाते हैं", needle));

        // Non-ASCII neighbours are boundaries (only a-z/0-9 are word
        // characters), so Devanagari suffixes do not block a match...
        assert!(contains_whole_term("मॉडलिंग करते हैं", needle));
        // ...but ASCII alphanumeric neighbours still do.
        assert!(!contains_whole_term("मॉडलx सिस्टम", needle));
        assert!(!contains_whole_term("xमॉडल सिस्टम", needle));

        // Overlap retry path with a multi-byte first character: the needle
        // appears embedded once (bad boundary), then standalone (good).
        assert!(contains_whole_term("xमॉडल और मॉडल", needle));
    }
}
