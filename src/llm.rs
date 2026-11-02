#![allow(dead_code)]
//! LLM clients for paper-radar.
//!
//! All paper-radar prompts go through [`LlmClient`], so tests can substitute
//! [`FakeLlm`] and nothing outside this module talks to a provider directly.
//!
//! Security posture (see AGENTS.md): every prompt is plain text we construct
//! ourselves; untrusted paper text is delimited by the caller before it
//! reaches here. The model has no tools, runs no commands and fetches no
//! URLs — our code does all I/O.
//!
//! [`ChatLlm`] speaks any OpenAI-compatible chat API over plain HTTP:
//! `POST {base_url}/chat/completions` with a bearer token, one system and
//! one user message, no streaming, no `temperature`, no tools. Redirects are
//! never followed, so the API key can only ever reach the configured host.
//! [`SoketLlm`] is the older client built on the Loop harness, kept until
//! `main` migrates to `ChatLlm`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::config::ModelConfig;
use crate::secrets::Secret;

use loop_agent::harness::{
    create_in_memory_session_store, create_session_repository, AgentHarness, AgentHarnessOptions,
    HostExecutionEnv, SandboxMode,
};
use loop_ai::providers::{soket_provider, SOKET_PROVIDER_ID};
use loop_ai::{AssistantContent, Credential, Models, ModelsRefreshOptions};

/// Output limit for Soket models via the Loop harness. GLM-5.3 always reasons
/// before answering and reasoning counts toward the limit, so the catalog
/// default of 16,384 can be exhausted by reasoning alone.
const MAX_OUTPUT_TOKENS: u64 = 32_768;

/// Hard cap on one response body (AGENTS.md: max response size 5 MB).
const MAX_RESPONSE_BYTES: usize = 5 * 1024 * 1024;
/// Cap on how much of an error body is put into an [`LlmError::Call`].
const MAX_ERROR_BODY_CHARS: usize = 300;
/// Overall request timeout for [`ChatLlm`].
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// TCP/TLS connect timeout for [`ChatLlm`].
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Failures from an [`LlmClient`] call.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The client or model could not be set up (bad model id, catalog
    /// refresh failure, HTTP client construction error).
    #[error("failed to set up the LLM client: {0}")]
    Setup(String),
    /// The prompt ran but the turn failed.
    #[error("LLM call failed: {0}")]
    Call(String),
    /// The model finished without producing any text — e.g. it spent the
    /// whole output limit on reasoning and stopped with reason `length`.
    /// Callers should treat this as a failure, not an empty string.
    #[error("LLM reply was empty: the model returned no text")]
    EmptyReply,
}

/// A one-shot text completion client: send `system` + `prompt`, get text back.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Runs one completion. Returns [`LlmError::EmptyReply`] rather than an
    /// empty string when the model produces no text.
    async fn complete(&self, system: &str, prompt: &str) -> Result<String, LlmError>;

    /// Starts a fresh conversation, dropping prior context. Call this between
    /// independent calls (e.g. one ranking batch must not see another's
    /// papers); leave it out when follow-ups should keep context.
    async fn reset(&self) -> Result<(), LlmError>;
}

/// [`LlmClient`] backed by any OpenAI-compatible chat API.
///
/// Built from `[model]` in config.toml: `base_url`, `model`, `max_tokens`
/// and `max_tokens_field`. The client never follows redirects (the API key
/// stays on the configured host), applies timeouts, and caps response bodies
/// at 5 MB. Each [`complete`](LlmClient::complete) call stands alone — there
/// is no conversation state, and `reset` is a no-op.
pub struct ChatLlm {
    client: reqwest::Client,
    url: String,
    api_key: Secret,
    model: String,
    max_tokens: u32,
    max_tokens_field: String,
}

impl ChatLlm {
    /// Builds the client from `[model]` config plus the provider API key.
    ///
    /// Fails only if the HTTP client cannot be constructed (e.g. a TLS
    /// backend problem); scheme/host validation of `base_url` already
    /// happened during config load.
    pub fn new(config: &ModelConfig, api_key: Secret) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(|e| LlmError::Setup(format!("could not build HTTP client: {e}")))?;

        // base_url may or may not end in "/"; normalise so we always POST to
        // exactly "{base}/chat/completions".
        let base = config.base_url.trim().trim_end_matches('/');
        let url = format!("{base}/chat/completions");

        Ok(Self {
            client,
            url,
            api_key,
            model: config.model.clone(),
            max_tokens: config.max_tokens,
            max_tokens_field: config.max_tokens_field.clone(),
        })
    }

    /// The request body for one completion: only `model`, the output limit
    /// field, and two messages (one system, one user). Nothing else — some
    /// providers reject extras like `temperature` with a 400.
    fn build_body(&self, system: &str, prompt: &str) -> Value {
        build_body(
            &self.model,
            &self.max_tokens_field,
            self.max_tokens,
            system,
            prompt,
        )
    }
}

/// Assembles the chat/completions JSON body. Free function so tests can call
/// it without a client.
fn build_body(
    model: &str,
    max_tokens_field: &str,
    max_tokens: u32,
    system: &str,
    prompt: &str,
) -> Value {
    json!({
        "model": model,
        max_tokens_field: max_tokens,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": prompt },
        ],
    })
}

/// Extracts the answer text from a chat/completions response body.
///
/// Reads `choices[0].message.content`. A missing, null or blank value is
/// [`LlmError::EmptyReply`] (e.g. reasoning ate the whole output limit and
/// the model stopped with `finish_reason: length`). A body that is not valid
/// JSON is [`LlmError::Call`]. The raw body is never included in an error.
fn parse_reply(body: &str) -> Result<String, LlmError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|_| LlmError::Call("response was not valid JSON".to_string()))?;

    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str);

    match content.map(str::trim) {
        Some(text) if !text.is_empty() => Ok(text.to_string()),
        _ => Err(LlmError::EmptyReply),
    }
}

#[async_trait]
impl LlmClient for ChatLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String, LlmError> {
        let body = serde_json::to_string(&self.build_body(system, prompt))
            .map_err(|e| LlmError::Call(format!("could not serialise request body: {e}")))?;

        let response = self
            .client
            .post(&self.url)
            .bearer_auth(self.api_key.expose())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| LlmError::Call(format!("request failed: {e}")))?;

        let status = response.status();
        let body = read_capped(response).await?;

        if !status.is_success() {
            return Err(LlmError::Call(format!(
                "provider returned {}: {}",
                status,
                redact(&truncate_chars(&body, MAX_ERROR_BODY_CHARS), &self.api_key)
            )));
        }
        parse_reply(&body)
    }

    async fn reset(&self) -> Result<(), LlmError> {
        // Each call stands alone; there is no conversation to reset.
        Ok(())
    }
}

/// Reads a response body chunk by chunk, failing once it exceeds
/// [`MAX_RESPONSE_BYTES`].
async fn read_capped(mut response: reqwest::Response) -> Result<String, LlmError> {
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| LlmError::Call(format!("could not read response body: {e}")))?
    {
        if bytes.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(LlmError::Call(format!(
                "response body exceeded {} bytes",
                MAX_RESPONSE_BYTES
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes)
        .map_err(|_| LlmError::Call("response body was not valid UTF-8".to_string()))
}

/// Truncates to at most `max` characters on a char boundary.
fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Replaces the API key with `[REDACTED]` if a provider echoes it back in an
/// error body. The key never appears in any error we surface.
fn redact(s: &str, api_key: &Secret) -> String {
    let exposed = api_key.expose();
    if exposed.is_empty() {
        s.to_string()
    } else {
        s.replace(exposed, "[REDACTED]")
    }
}

/// [`LlmClient`] backed by the Soket provider through the Loop harness.
///
/// Kept until `main` migrates to [`ChatLlm`]; new code should use `ChatLlm`.
/// The harness has no tools and no sandbox; it is purely a chat completion
/// with a system prompt. Sessions are in-memory — one [`SoketLlm`] value
/// holds one conversation, so "go deeper" follow-ups on a digest can reuse
/// the same instance and keep context.
pub struct SoketLlm {
    harness: AgentHarness,
}

impl SoketLlm {
    /// Builds the client: registers the Soket provider with the given API
    /// key, refreshes the model catalog, picks `model_id` (e.g. `GLM-5.3`)
    /// and raises its output limit to [`MAX_OUTPUT_TOKENS`].
    pub async fn new(api_key: &str, model_id: &str) -> Result<Self, LlmError> {
        let models = Arc::new(Models::new());
        models.set_provider(soket_provider());
        models.credentials().set(
            SOKET_PROVIDER_ID,
            Credential::ApiKey {
                key: api_key.to_string(),
            },
        );

        // Refresh the catalog so `model_id` resolves against the live Soket
        // list. On failure the provider keeps its seed list; if the model is
        // missing either way, `get_model` below reports it.
        let refreshed = models
            .refresh(ModelsRefreshOptions {
                provider_id: Some(SOKET_PROVIDER_ID.to_string()),
                ..Default::default()
            })
            .await;

        let mut model = models
            .get_model(SOKET_PROVIDER_ID, model_id)
            .ok_or_else(|| {
                let mut reason = format!("model {SOKET_PROVIDER_ID}/{model_id} is not available");
                if let Some(err) = refreshed.errors.get(SOKET_PROVIDER_ID) {
                    reason.push_str(&format!(" (catalog refresh failed: {err})"));
                }
                LlmError::Setup(reason)
            })?;
        model.max_tokens = MAX_OUTPUT_TOKENS;

        // In-memory session: nothing persists between runs yet.
        let store = create_in_memory_session_store();
        let repository = create_session_repository(store, None);
        let session = repository
            .create(None, Some("paper-radar".into()))
            .await
            .map_err(|e| LlmError::Setup(format!("could not create LLM session: {e}")))?;

        let harness = AgentHarness::new(AgentHarnessOptions {
            models,
            model: Some(model),
            session,
            host_env: Arc::new(HostExecutionEnv::new(std::env::temp_dir())),
            tools: vec![],
            system_prompt: String::new(),
            sandbox: SandboxMode::Disabled,
            resources: Default::default(),
        });

        Ok(Self { harness })
    }
}

#[async_trait]
impl LlmClient for SoketLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String, LlmError> {
        self.harness.set_system_prompt(system).await;
        let message = self
            .harness
            .prompt(prompt)
            .await
            .map_err(|e| LlmError::Call(e.to_string()))?;

        let assistant = message.as_assistant().ok_or_else(|| {
            LlmError::Call("model turn produced no assistant message".to_string())
        })?;
        if let Some(failure) = assistant.failure() {
            return Err(LlmError::Call(failure));
        }

        // Only text blocks carry the answer; thinking blocks are reasoning.
        let text: String = assistant
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        if text.is_empty() {
            // Most often a `length` stop after reasoning ate the budget.
            return Err(LlmError::EmptyReply);
        }
        Ok(text)
    }

    async fn reset(&self) -> Result<(), LlmError> {
        self.harness
            .start_new_session(None, Some("paper-radar".into()))
            .await
            .map(|_| ())
            .map_err(|e| LlmError::Call(e.to_string()))
    }
}

/// Scripted [`LlmClient`] for tests: returns queued replies in order and
/// records every `(system, prompt)` pair it was given.
pub struct FakeLlm {
    replies: Mutex<VecDeque<String>>,
    requests: Mutex<Vec<(String, String)>>,
    resets: Mutex<u32>,
}

impl FakeLlm {
    /// Creates a fake that answers with `replies`, in order.
    pub fn new(replies: Vec<String>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            resets: Mutex::new(0),
        }
    }

    /// The `(system, prompt)` pairs seen so far, in order.
    pub fn requests(&self) -> Vec<(String, String)> {
        self.requests.lock().unwrap().clone()
    }

    /// How many times [`LlmClient::reset`] was called.
    pub fn reset_count(&self) -> u32 {
        *self.resets.lock().unwrap()
    }
}

#[async_trait]
impl LlmClient for FakeLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String, LlmError> {
        self.requests
            .lock()
            .unwrap()
            .push((system.to_string(), prompt.to_string()));

        let reply =
            self.replies.lock().unwrap().pop_front().ok_or_else(|| {
                LlmError::Call("fake LLM ran out of scripted replies".to_string())
            })?;

        // Honour the trait contract the way SoketLlm does.
        if reply.trim().is_empty() {
            return Err(LlmError::EmptyReply);
        }
        Ok(reply)
    }

    async fn reset(&self) -> Result<(), LlmError> {
        *self.resets.lock().unwrap() += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_llm_answers_in_order_and_records_requests() {
        let fake = FakeLlm::new(vec!["first".to_string(), "second".to_string()]);

        let ranking = fake
            .complete("you rank papers", "score these abstracts")
            .await
            .expect("first scripted reply");
        let deeper = fake
            .complete("you summarise papers", "go deeper on #3")
            .await
            .expect("second scripted reply");

        assert_eq!(ranking, "first");
        assert_eq!(deeper, "second");
        assert_eq!(
            fake.requests(),
            vec![
                (
                    "you rank papers".to_string(),
                    "score these abstracts".to_string()
                ),
                (
                    "you summarise papers".to_string(),
                    "go deeper on #3".to_string()
                ),
            ]
        );
    }

    #[tokio::test]
    async fn blank_reply_is_reported_as_empty_reply_not_an_empty_string() {
        let fake = FakeLlm::new(vec!["   ".to_string()]);

        let err = fake
            .complete("system", "prompt")
            .await
            .expect_err("blank reply must be an error");

        assert!(matches!(err, LlmError::EmptyReply));
    }

    #[tokio::test]
    async fn reset_succeeds_and_is_counted() {
        let fake = FakeLlm::new(vec![]);

        assert_eq!(fake.reset_count(), 0);
        fake.reset().await.expect("reset should succeed");
        fake.reset().await.expect("reset should succeed");
        assert_eq!(fake.reset_count(), 2);

        // Resetting does not touch the scripted replies or recorded requests.
        assert!(fake.requests().is_empty());
    }

    #[test]
    fn build_body_uses_max_tokens_field_with_both_known_names() {
        for field in ["max_tokens", "max_completion_tokens"] {
            let body = build_body("GLM-5.3", field, 32_768, "be brief", "score these");

            assert_eq!(body["model"], "GLM-5.3");
            assert_eq!(body[field], 32_768, "limit must sit under {field:?}");
            // The other known limit field must NOT be present.
            let other = if field == "max_tokens" {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            assert!(
                body.get(other).is_none(),
                "{other:?} must not appear when using {field:?}"
            );
        }
    }

    #[test]
    fn build_body_has_only_model_limit_and_two_messages() {
        let body = build_body(
            "GLM-5.3",
            "max_tokens",
            1024,
            "you rank papers",
            "score these abstracts",
        );

        // Exactly the allowed keys: no temperature, top_p, stream, tools...
        let mut keys: Vec<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["max_tokens", "messages", "model"]);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "you rank papers");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "score these abstracts");
    }

    #[test]
    fn parse_reply_reads_first_choice_content() {
        let body = r#"{
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "  7 | relevant to KV cache  "},
                    "finish_reason": "stop"
                }
            ]
        }"#;
        let text = parse_reply(body).expect("a normal answer should parse");
        assert_eq!(text, "7 | relevant to KV cache");
    }

    #[test]
    fn parse_reply_null_or_blank_content_is_empty_reply() {
        // Empty and whitespace-only content strings.
        for content in [serde_json::json!(""), serde_json::json!("   ")] {
            let body = json!({"choices":[{"message":{"content":content}}]}).to_string();
            let err =
                parse_reply(&body).expect_err("blank content must be EmptyReply, not Ok(\"\")");
            assert!(
                matches!(err, LlmError::EmptyReply),
                "blank {content:?} gave {err:?}"
            );
        }

        // Explicit null content, e.g. after reasoning ate the output limit.
        let body = r#"{"choices":[{"message":{"content":null}}]}"#;
        let err = parse_reply(body).expect_err("null content must be EmptyReply");
        assert!(matches!(err, LlmError::EmptyReply));
    }

    #[test]
    fn parse_reply_missing_choices_is_empty_reply() {
        // A well-formed JSON body with no choices array at all: nothing to
        // read an answer from, so it is an empty reply rather than a crash.
        let body = r#"{"id":"chatcmpl-1","object":"chat.completion"}"#;
        let err = parse_reply(body).expect_err("missing choices must fail");
        assert!(matches!(err, LlmError::EmptyReply));

        // Same for an empty choices array.
        let body = r#"{"choices":[]}"#;
        let err = parse_reply(body).expect_err("empty choices must fail");
        assert!(matches!(err, LlmError::EmptyReply));
    }

    #[test]
    fn parse_reply_invalid_json_is_call_error_without_the_body() {
        for body in ["not json at all", "{truncated", ""] {
            let err = parse_reply(body).expect_err("invalid JSON must fail");
            assert!(
                matches!(err, LlmError::Call(_)),
                "body {body:?} gave {err:?}"
            );
            // The raw body must never end up inside the error message.
            let msg = err.to_string();
            if !body.trim().is_empty() {
                assert!(!msg.contains(body.trim()), "error leaked body: {msg}");
            }
        }
    }

    #[test]
    fn chat_llm_new_normalises_url_and_keeps_config() {
        let config = ModelConfig {
            base_url: "https://api.soket.ai/v1/".to_string(),
            model: "GLM-5.3".to_string(),
            max_tokens: 4096,
            max_tokens_field: "max_completion_tokens".to_string(),
        };
        let llm = ChatLlm::new(&config, Secret::new("sk-test"))
            .expect("valid config should build a client");

        assert_eq!(llm.url, "https://api.soket.ai/v1/chat/completions");
        assert_eq!(llm.model, "GLM-5.3");
        assert_eq!(llm.max_tokens, 4096);
        assert_eq!(llm.max_tokens_field, "max_completion_tokens");
        // The key is stored redacted and never printed by Debug.
        assert_eq!(format!("{:?}", llm.api_key), "[REDACTED]");
    }

    #[test]
    fn chat_llm_new_strips_multiple_trailing_slashes() {
        let config = ModelConfig {
            base_url: "http://localhost:8000/v1//".to_string(),
            model: "GLM-5.3".to_string(),
            max_tokens: 32768,
            max_tokens_field: "max_tokens".to_string(),
        };
        let llm = ChatLlm::new(&config, Secret::new("sk-test")).expect("local http is allowed");
        assert_eq!(llm.url, "http://localhost:8000/v1/chat/completions");
    }

    #[test]
    fn redact_replaces_the_api_key_but_not_other_text() {
        let key = Secret::new("sk-secret-key-123");
        let body = "Unauthorized: sk-secret-key-123 is invalid";
        assert_eq!(redact(body, &key), "Unauthorized: [REDACTED] is invalid");
        assert_eq!(redact("no key here", &key), "no key here");
    }

    #[test]
    fn truncate_chars_cuts_on_char_boundaries() {
        assert_eq!(truncate_chars("abcdefghij", 300), "abcdefghij");
        assert_eq!(truncate_chars("abcdefghij", 4), "abcd");
        // Multi-byte characters are not split mid-character.
        assert_eq!(truncate_chars("hindi word: नमस्ते", 13), "hindi word: न");
    }
}
