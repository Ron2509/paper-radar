//! Secrets handling for paper-radar.
//!
//! Loads `SOKET_API_KEY`, `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` from the
//! environment, with an optional `.env` file applied first via dotenvy (real
//! environment variables always win). Each value is wrapped in [`Secret`],
//! whose `Debug` and `Display` print `[REDACTED]`, so a secret can never leak
//! into logs or error messages. Call [`Secret::expose`] only at the exact
//! place the real value is needed.

// TODO(wiring): remove once `main` starts using secrets; until then these
// types are only consumed by tests.
#![allow(dead_code)]

use std::fmt;

use thiserror::Error;

/// Name of the Soket API key environment variable.
pub const SOKET_API_KEY_VAR: &str = "SOKET_API_KEY";
/// Name of the Slack bot token environment variable.
pub const SLACK_BOT_TOKEN_VAR: &str = "SLACK_BOT_TOKEN";
/// Name of the Slack app (Socket Mode) token environment variable.
pub const SLACK_APP_TOKEN_VAR: &str = "SLACK_APP_TOKEN";

/// Errors produced while loading secrets.
#[derive(Debug, Error)]
pub enum SecretsError {
    /// A required variable is not set (or is empty/whitespace).
    #[error(
        "required environment variable {var} is not set; add it to .env or your shell environment"
    )]
    Missing { var: &'static str },
    /// A token is set but does not have the expected prefix.
    #[error("environment variable {var} must start with {expected}")]
    BadPrefix {
        var: &'static str,
        expected: &'static str,
    },
}

/// A secret value that refuses to print itself.
///
/// `Debug` and `Display` both render as `[REDACTED]`; the value is only
/// available through [`Secret::expose`].
#[derive(Clone)]
pub struct Secret {
    value: String,
}

impl Secret {
    /// Wraps a value; nothing is validated here.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }

    /// The real value. Use only where the secret is actually consumed.
    pub fn expose(&self) -> &str {
        &self.value
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

/// All secrets paper-radar needs, loaded and validated.
#[derive(Debug, Clone)]
pub struct Secrets {
    /// Key for the Soket LLM API.
    pub soket_api_key: Secret,
    /// Slack bot token (`xoxb-...`).
    pub slack_bot_token: Secret,
    /// Slack app token for Socket Mode (`xapp-...`).
    pub slack_app_token: Secret,
}

impl Secrets {
    /// Loads secrets from the process environment, applying `.env` first if
    /// present. A missing or unreadable `.env` is not an error as long as the
    /// variables come from the shell.
    pub fn load() -> Result<Self, SecretsError> {
        let _ = dotenvy::dotenv();
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Loads secrets from a lookup function instead of the real environment.
    /// Used by `load` and by tests.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, SecretsError> {
        let soket_api_key = Secret::new(require(&lookup, SOKET_API_KEY_VAR)?);

        let bot_token = require(&lookup, SLACK_BOT_TOKEN_VAR)?;
        ensure_prefix(SLACK_BOT_TOKEN_VAR, &bot_token, "xoxb-")?;

        let app_token = require(&lookup, SLACK_APP_TOKEN_VAR)?;
        ensure_prefix(SLACK_APP_TOKEN_VAR, &app_token, "xapp-")?;

        Ok(Self {
            soket_api_key,
            slack_bot_token: Secret::new(bot_token),
            slack_app_token: Secret::new(app_token),
        })
    }
}

/// Fetches a non-empty variable, trimming surrounding whitespace.
fn require(
    lookup: &dyn Fn(&str) -> Option<String>,
    var: &'static str,
) -> Result<String, SecretsError> {
    match lookup(var).map(|v| v.trim().to_string()) {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(SecretsError::Missing { var }),
    }
}

/// Checks a token's prefix without ever including the token in the error.
fn ensure_prefix(
    var: &'static str,
    value: &str,
    expected: &'static str,
) -> Result<(), SecretsError> {
    if value.starts_with(expected) {
        Ok(())
    } else {
        Err(SecretsError::BadPrefix { var, expected })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a lookup from fixed pairs: no real `.env`, no real environment.
    fn lookup_from<'a>(
        pairs: &'a [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn loads_all_secrets_and_redacts_them_in_debug_and_display() {
        let vars = [
            (SOKET_API_KEY_VAR, "sk-test-soket-key"),
            (SLACK_BOT_TOKEN_VAR, "xoxb-1234567890-abcdef"),
            (SLACK_APP_TOKEN_VAR, "xapp-1-A1B2C3D4E5"),
        ];
        let secrets =
            Secrets::from_lookup(lookup_from(&vars)).expect("valid variables should load");

        // Real values are available only through expose().
        assert_eq!(secrets.soket_api_key.expose(), "sk-test-soket-key");
        assert_eq!(secrets.slack_bot_token.expose(), "xoxb-1234567890-abcdef");
        assert_eq!(secrets.slack_app_token.expose(), "xapp-1-A1B2C3D4E5");

        // Debug and Display never reveal values, individually or nested.
        assert_eq!(format!("{:?}", secrets.soket_api_key), "[REDACTED]");
        assert_eq!(format!("{}", secrets.slack_bot_token), "[REDACTED]");
        let whole = format!("{secrets:?}");
        assert!(
            whole.contains("[REDACTED]"),
            "struct Debug should redact: {whole}"
        );
        assert!(!whole.contains("sk-test-soket-key"));
        assert!(!whole.contains("xoxb-"));
        assert!(!whole.contains("xapp-"));
    }

    #[test]
    fn missing_or_malformed_variable_gives_clear_error() {
        // SLACK_APP_TOKEN absent.
        let vars = [
            (SOKET_API_KEY_VAR, "sk-test-soket-key"),
            (SLACK_BOT_TOKEN_VAR, "xoxb-1234567890-abcdef"),
        ];
        let err = Secrets::from_lookup(lookup_from(&vars)).expect_err("missing variable must fail");
        assert!(
            err.to_string().contains("SLACK_APP_TOKEN"),
            "error should name the missing variable: {err}"
        );

        // SLACK_APP_TOKEN set but empty counts as missing too.
        let vars = [
            (SOKET_API_KEY_VAR, "sk-test-soket-key"),
            (SLACK_BOT_TOKEN_VAR, "xoxb-1234567890-abcdef"),
            (SLACK_APP_TOKEN_VAR, "   "),
        ];
        let err = Secrets::from_lookup(lookup_from(&vars)).expect_err("empty variable must fail");
        assert!(
            err.to_string().contains("SLACK_APP_TOKEN"),
            "error should name the empty variable: {err}"
        );

        // Bot token with the wrong prefix: error names the variable and the
        // expected prefix, never the token value.
        let vars = [
            (SOKET_API_KEY_VAR, "sk-test-soket-key"),
            (SLACK_BOT_TOKEN_VAR, "xoxp-not-a-bot-token"),
            (SLACK_APP_TOKEN_VAR, "xapp-1-A1B2C3D4E5"),
        ];
        let err = Secrets::from_lookup(lookup_from(&vars)).expect_err("wrong prefix must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("SLACK_BOT_TOKEN"),
            "should name variable: {msg}"
        );
        assert!(msg.contains("xoxb-"), "should name expected prefix: {msg}");
        assert!(
            !msg.contains("xoxp-not-a-bot-token"),
            "must not echo the token"
        );
    }
}
