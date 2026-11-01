#![allow(dead_code)]

//! Posting messages to Slack via the Web API.

use anyhow::{anyhow, Context};

/// Escape a string for Slack and neutralise broadcast mentions.
///
/// `&`, `<` and `>` are HTML-escaped first (Slack's markup treats `<...>` as
/// entities), then the literal `@channel`, `@here` and `@everyone` triggers
/// lose their `@`. The result is capped at 12,000 characters on a char
/// boundary.
pub fn sanitize_for_slack(text: &str) -> String {
    let escaped = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");

    let no_mentions = escaped
        .replace("@channel", "channel")
        .replace("@here", "here")
        .replace("@everyone", "everyone");

    truncate_chars(&no_mentions, 12_000)
}

/// Truncate to at most `max_chars` characters, on a char boundary.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect()
}

/// Post a message to a Slack channel with `chat.postMessage`.
///
/// Link unfurling and name expansion are disabled; `thread_ts` posts into a
/// thread when given. On success returns the new message's `ts`. Errors
/// never contain the bot token.
pub async fn post_message(
    bot_token: &str,
    channel: &str,
    text: &str,
    thread_ts: Option<&str>,
) -> anyhow::Result<String> {
    const URL: &str = "https://slack.com/api/chat.postMessage";

    let mut body = serde_json::json!({
        "channel": channel,
        "text": text,
        "unfurl_links": false,
        "unfurl_media": false,
        "link_names": false,
    });
    if let Some(thread_ts) = thread_ts {
        body["thread_ts"] = serde_json::json!(thread_ts);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("building Slack HTTP client")?;

    let response = client
        .post(URL)
        .header("Authorization", format!("Bearer {bot_token}"))
        .header("Content-Type", "application/json; charset=utf-8")
        .body(serde_json::to_string(&body).context("serializing Slack message body")?)
        .send()
        .await
        .context("sending Slack chat.postMessage request")?;

    let reply_text = response
        .text()
        .await
        .context("reading Slack chat.postMessage reply")?;
    let reply: serde_json::Value =
        serde_json::from_str(&reply_text).context("parsing Slack chat.postMessage reply")?;

    if reply.get("ok") != Some(&serde_json::Value::Bool(true)) {
        let error = reply
            .get("error")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown_error");
        // Deliberately only Slack's error field: never echo the token.
        return Err(anyhow!("slack error: {error}"));
    }

    let ts = reply
        .get("ts")
        .and_then(|value| value.as_str())
        .context("Slack reply missing ts")?
        .to_string();
    Ok(ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutralises_channel_broadcast() {
        let sanitized = sanitize_for_slack("<!channel> hello");

        assert_eq!(sanitized, "&lt;!channel&gt; hello");
    }

    #[test]
    fn here_mention_loses_its_at() {
        let sanitized = sanitize_for_slack("attention @here please");

        assert_eq!(sanitized, "attention here please");
    }
}
