//! Configuration loading for paper-radar.
//!
//! Reads `config.toml` into typed structs, one per TOML section. Each section
//! is parsed independently so that a missing field produces an error that
//! names the section and the field, e.g.
//! `config error in "config.toml": [digest] missing field \`min_score\``.

// TODO(wiring): remove once `main` starts using the config; until then the
// structs are only consumed by tests.
#![allow(dead_code)]

use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use thiserror::Error;

/// Errors produced while loading or validating the configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("cannot read config file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// The file is not valid TOML.
    #[error("invalid TOML in config file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    /// A required section or field is missing, or a value is out of range.
    #[error("config error in {path}: {message}")]
    Invalid { path: String, message: String },
}

/// `[model]` — Soket model used for scoring and summaries.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Soket model id, e.g. "GLM-5.3".
    pub model: String,
}

/// `[arxiv]` — what to fetch and how politely.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArxivConfig {
    /// arXiv categories, e.g. ["cs.CL", "cs.LG"].
    pub categories: Vec<String>,
    /// Contact email for the arXiv User-Agent.
    pub contact_email: String,
    /// Minimum seconds between arXiv requests (arXiv asks for >= 3).
    pub request_interval_secs: u64,
}

/// `[topics]` — seed of the team interest profile.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopicsConfig {
    /// Team interest topics.
    pub topics: Vec<String>,
    /// Keyword prefilter applied before LLM scoring.
    pub keywords: Vec<String>,
}

/// `[digest]` — what makes it into the daily post.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DigestConfig {
    /// Minimum LLM relevance score (0-10) for a paper to be eligible.
    pub min_score: u8,
    /// Maximum papers posted per digest (1-5).
    pub max_papers: usize,
    /// Whether to post a one-line message when nothing passes the threshold.
    pub post_when_empty: bool,
}

/// `[slack]` — where and how to interact.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlackConfig {
    /// Channel ID of the team's papers channel (not the name).
    pub channel_id: String,
    /// Max "go deeper" requests per user per hour.
    pub deeper_rate_limit_per_hour: u32,
}

/// `[schedule]` — when the daily digest runs.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleConfig {
    /// Digest time of day, "HH:MM".
    pub time: String,
    /// IANA timezone name, e.g. "Asia/Kolkata".
    pub timezone: String,
    /// Run on weekdays only.
    pub weekdays_only: bool,
}

impl ScheduleConfig {
    /// The parsed timezone. Safe to unwrap: validated during load.
    pub fn tz(&self) -> chrono_tz::Tz {
        self.timezone
            .parse()
            .expect("timezone was validated during config load")
    }
}

/// Root of `config.toml`.
#[derive(Debug, Clone)]
pub struct Config {
    pub model: ModelConfig,
    pub arxiv: ArxivConfig,
    pub topics: TopicsConfig,
    pub digest: DigestConfig,
    pub slack: SlackConfig,
    pub schedule: ScheduleConfig,
}

impl Config {
    /// Loads config from the default location: `$PAPER_RADAR_CONFIG`, or
    /// `config.toml` in the current directory.
    pub fn load_default() -> Result<Self, ConfigError> {
        let path =
            std::env::var("PAPER_RADAR_CONFIG").unwrap_or_else(|_| "config.toml".to_string());
        Self::load(path)
    }

    /// Loads and validates config from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let origin = path.display().to_string();
        let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: origin.clone(),
            source,
        })?;
        Self::parse(&raw, &origin)
    }

    /// Parses config from a TOML string. Used by `load` and by tests.
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        Self::parse(raw, "<config string>")
    }

    fn parse(raw: &str, origin: &str) -> Result<Self, ConfigError> {
        let table: toml::Table = toml::from_str(raw).map_err(|source| ConfigError::Parse {
            path: origin.to_string(),
            source,
        })?;

        let config = Config {
            model: parse_section(&table, "model", origin)?,
            arxiv: parse_section(&table, "arxiv", origin)?,
            topics: parse_section(&table, "topics", origin)?,
            digest: parse_section(&table, "digest", origin)?,
            slack: parse_section(&table, "slack", origin)?,
            schedule: parse_section(&table, "schedule", origin)?,
        };
        config.validate(origin)?;
        Ok(config)
    }

    fn validate(&self, origin: &str) -> Result<(), ConfigError> {
        let err = |message: String| ConfigError::Invalid {
            path: origin.to_string(),
            message,
        };

        if self.model.model.trim().is_empty() {
            return Err(err("[model] model must not be empty".to_string()));
        }
        if self.arxiv.categories.is_empty() {
            return Err(err("[arxiv] categories must not be empty".to_string()));
        }
        if self.arxiv.contact_email.trim().is_empty() {
            return Err(err("[arxiv] contact_email must not be empty".to_string()));
        }
        if self.arxiv.request_interval_secs < 3 {
            return Err(err(
                "[arxiv] request_interval_secs must be at least 3 (arXiv etiquette)".to_string(),
            ));
        }
        if self.topics.topics.is_empty() {
            return Err(err("[topics] topics must not be empty".to_string()));
        }
        if self.topics.keywords.is_empty() {
            return Err(err("[topics] keywords must not be empty".to_string()));
        }
        if self.digest.min_score > 10 {
            return Err(err(format!(
                "[digest] min_score must be 0-10, got {}",
                self.digest.min_score
            )));
        }
        if self.digest.max_papers == 0 || self.digest.max_papers > 5 {
            return Err(err(format!(
                "[digest] max_papers must be between 1 and 5, got {}",
                self.digest.max_papers
            )));
        }
        if self.slack.channel_id.trim().is_empty() {
            return Err(err("[slack] channel_id must not be empty".to_string()));
        }
        if self.slack.deeper_rate_limit_per_hour == 0 {
            return Err(err(
                "[slack] deeper_rate_limit_per_hour must be at least 1".to_string()
            ));
        }
        if let Err(e) = parse_time_of_day(&self.schedule.time) {
            return Err(err(format!(
                "[schedule] invalid time {:?}: {e}",
                self.schedule.time
            )));
        }
        if self.schedule.timezone.parse::<chrono_tz::Tz>().is_err() {
            return Err(err(format!(
                "[schedule] unknown timezone {:?}",
                self.schedule.timezone
            )));
        }
        Ok(())
    }
}

/// Extracts and deserializes one `[section]` of the TOML table, with errors
/// prefixed by the section name.
fn parse_section<T>(table: &toml::Table, name: &str, origin: &str) -> Result<T, ConfigError>
where
    T: DeserializeOwned,
{
    let value = table.get(name).ok_or_else(|| ConfigError::Invalid {
        path: origin.to_string(),
        message: format!("missing required section [{name}]"),
    })?;
    value.clone().try_into().map_err(|e| ConfigError::Invalid {
        path: origin.to_string(),
        message: format!("[{name}] {e}"),
    })
}

/// Parses "HH:MM" (24h) into `(hour, minute)`.
fn parse_time_of_day(s: &str) -> Result<(u8, u8), String> {
    let (h, m) = s
        .split_once(':')
        .ok_or_else(|| "expected HH:MM".to_string())?;
    let hour: u8 = h.parse().map_err(|_| format!("invalid hour {h:?}"))?;
    let minute: u8 = m.parse().map_err(|_| format!("invalid minute {m:?}"))?;
    if hour > 23 {
        return Err(format!("hour {hour} out of range 0-23"));
    }
    if minute > 59 {
        return Err(format!("minute {minute} out of range 0-59"));
    }
    Ok((hour, minute))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_CONFIG: &str = r#"
[model]
model = "GLM-5.3"

[arxiv]
categories = ["cs.CL", "cs.LG", "cs.AI", "cs.DC", "cs.CR"]
contact_email = "team@example.com"
request_interval_secs = 3

[topics]
topics = ["RLHF", "KV cache", "Indic LLMs"]
keywords = ["rlhf", "kv cache", "indic"]

[digest]
min_score = 7
max_papers = 5
post_when_empty = true

[slack]
channel_id = "C0123456789"
deeper_rate_limit_per_hour = 5

[schedule]
time = "09:00"
timezone = "Asia/Kolkata"
weekdays_only = true
"#;

    #[test]
    fn loads_all_sections_from_valid_toml() {
        let cfg = Config::from_toml_str(FULL_CONFIG).expect("valid config should parse");

        assert_eq!(cfg.model.model, "GLM-5.3");
        assert_eq!(cfg.arxiv.categories.len(), 5);
        assert_eq!(cfg.arxiv.categories[0], "cs.CL");
        assert_eq!(cfg.arxiv.contact_email, "team@example.com");
        assert_eq!(cfg.arxiv.request_interval_secs, 3);
        assert_eq!(cfg.topics.topics.len(), 3);
        assert_eq!(cfg.topics.keywords, vec!["rlhf", "kv cache", "indic"]);
        assert_eq!(cfg.digest.min_score, 7);
        assert_eq!(cfg.digest.max_papers, 5);
        assert!(cfg.digest.post_when_empty);
        assert_eq!(cfg.slack.channel_id, "C0123456789");
        assert_eq!(cfg.slack.deeper_rate_limit_per_hour, 5);
        assert_eq!(cfg.schedule.time, "09:00");
        assert_eq!(cfg.schedule.tz().name(), "Asia/Kolkata");
        assert!(cfg.schedule.weekdays_only);
    }

    #[test]
    fn missing_required_field_gives_clear_error() {
        // Drop `min_score` from [digest]; the error must name both.
        let bad = FULL_CONFIG.replacen("min_score = 7\n", "", 1);

        let err = Config::from_toml_str(&bad).expect_err("config with missing field must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("digest"),
            "error should name the section: {msg}"
        );
        assert!(
            msg.contains("min_score"),
            "error should name the field: {msg}"
        );
    }

    #[test]
    fn unknown_field_gives_error_naming_it() {
        // A typo in a key must be rejected, not silently ignored.
        let bad = FULL_CONFIG.replacen("[slack]\n", "[slack]\nchanel_id = \"C0123456789\"\n", 1);

        let err = Config::from_toml_str(&bad).expect_err("unknown field must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("slack"),
            "error should name the section: {msg}"
        );
        assert!(
            msg.contains("chanel_id"),
            "error should name the typo'd field: {msg}"
        );
    }
}
