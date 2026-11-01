#![allow(dead_code)]
//! Thin wrapper around the Loop agent harness for LLM calls.
//!
//! All paper-radar prompts go through [`LlmClient`], so tests can substitute
//! [`FakeLlm`] and nothing outside this module talks to Loop or Soket
//! directly.
//!
//! Security posture (see AGENTS.md): the harness is created with NO tools and
//! [`SandboxMode::Disabled`], so the model can never run commands, touch
//! files or fetch URLs. Every prompt is plain text we construct ourselves;
//! untrusted paper text is delimited by the caller before it reaches here.

// TODO(wiring): remove once the digest pipeline calls the LLM; until then
// SoketLlm is only exercised manually.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use loop_agent::harness::{
    create_in_memory_session_store, create_session_repository, AgentHarness, AgentHarnessOptions,
    HostExecutionEnv, SandboxMode,
};
use loop_ai::providers::{soket_provider, SOKET_PROVIDER_ID};
use loop_ai::{AssistantContent, Credential, Models, ModelsRefreshOptions};

/// Output limit for Soket models. GLM-5.3 always reasons before answering and
/// reasoning counts toward the limit, so the catalog default of 16,384 can be
/// exhausted by reasoning alone, leaving an empty answer.
const MAX_OUTPUT_TOKENS: u64 = 32_768;

/// Failures from an [`LlmClient`] call.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The harness or model could not be set up (bad model id, catalog
    /// refresh failure, session error).
    #[error("failed to set up the LLM harness: {0}")]
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

/// [`LlmClient`] backed by the Soket provider through the Loop harness.
///
/// The harness has no tools and no sandbox; it is purely a chat completion
/// with a system prompt. Sessions are in-memory for now — one
/// [`SoketLlm`] value holds one conversation, so "go deeper" follow-ups on a
/// digest can reuse the same instance and keep context.
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
}
