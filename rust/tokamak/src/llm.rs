//! OpenAI-compatible HTTP client with bounded concurrency and retry.
//!
//! One backend only — `POST {endpoint}/chat/completions`. Works against
//! vLLM, TGI, SGLang, Together, Anyscale, OpenAI, llama.cpp's server mode,
//! Ollama, Anthropic-OpenAI-compat shims, etc. Anything that speaks the
//! OpenAI chat-completions schema.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub endpoint:    String,
    pub api_key:     String,
    pub model:       String,
    pub max_tokens:  u32,
    pub temperature: f32,
    pub timeout_s:   f64,
    pub retries:     u32,
}

impl LlmConfig {
    pub fn from_env_or(endpoint: Option<String>, model: Option<String>) -> Self {
        let endpoint = endpoint
            .or_else(|| std::env::var("TOKAMAK_ENDPOINT").ok())
            .or_else(|| std::env::var("TOKAMAK_OPENAI_BASE_URL").ok())
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "http://localhost:8000/v1".to_string());
        let api_key = std::env::var("TOKAMAK_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_else(|_| "dummy".to_string());
        let model = model
            .or_else(|| std::env::var("TOKAMAK_MODEL").ok())
            .unwrap_or_else(|| "gpt-4o-mini".to_string());
        let max_tokens = std::env::var("TOKAMAK_MAX_TOKENS").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(4096);
        let temperature: f32 = std::env::var("TOKAMAK_TEMPERATURE").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let timeout_s = std::env::var("TOKAMAK_TIMEOUT").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(180.0);
        let retries = std::env::var("TOKAMAK_RETRIES").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(2);
        Self { endpoint, api_key, model, max_tokens, temperature, timeout_s, retries }
    }
}

#[derive(Clone)]
pub struct LlmClient {
    cfg:    LlmConfig,
    client: Client,
}

impl LlmClient {
    pub fn new(cfg: LlmConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(cfg.timeout_s))
            .build()
            .context("build reqwest client")?;
        Ok(Self { cfg, client })
    }

    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        self.chat_with_retry(system, user).await
    }

    async fn chat_with_retry(&self, system: &str, user: &str) -> Result<String> {
        let mut attempt: u32 = 0;
        loop {
            match self.chat_once(system, user).await {
                Ok(s)  => return Ok(s),
                Err(e) => {
                    if attempt >= self.cfg.retries {
                        return Err(e);
                    }
                    let backoff = std::cmp::min(1u64 << attempt, 30);
                    tracing::warn!("llm transient error (attempt {}/{}): {} — retry in {}s",
                                   attempt+1, self.cfg.retries+1, e, backoff);
                    tokio::time::sleep(Duration::from_secs(backoff)).await;
                    attempt += 1;
                }
            }
        }
    }

    async fn chat_once(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.cfg.endpoint.trim_end_matches('/'));
        let body = ChatRequest {
            model: &self.cfg.model,
            messages: vec![
                ChatMessage { role: "system", content: system },
                ChatMessage { role: "user",   content: user },
            ],
            max_tokens:  self.cfg.max_tokens,
            temperature: self.cfg.temperature,
        };
        let resp = self.client
            .post(&url)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send().await
            .context("POST /chat/completions")?
            .error_for_status()?;
        let parsed: ChatResponse = resp.json().await.context("decode chat response")?;
        let text = parsed.choices.into_iter().next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();
        Ok(text.trim().to_string())
    }

    /// Run a batch of (system, user) chats with semaphore-bounded concurrency.
    /// Order preserved. None ⇒ permanent failure for that item (caller falls back).
    pub async fn batch_chat(self, items: Vec<(String, String)>, max_workers: usize) -> Vec<Option<String>> {
        if items.is_empty() { return Vec::new(); }
        let sem = Arc::new(Semaphore::new(max_workers.max(1)));
        let me = Arc::new(self);
        let mut handles = Vec::with_capacity(items.len());
        for (system, user) in items.into_iter() {
            let permit = Arc::clone(&sem);
            let me_clone = Arc::clone(&me);
            handles.push(tokio::spawn(async move {
                let _p = permit.acquire_owned().await.expect("semaphore");
                me_clone.chat(&system, &user).await.ok()
            }));
        }
        let mut out = Vec::with_capacity(handles.len());
        for h in handles { out.push(h.await.unwrap_or(None)); }
        out
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
}
#[derive(Serialize)]
struct ChatMessage<'a> { role: &'a str, content: &'a str }

#[derive(Deserialize)]
struct ChatResponse { choices: Vec<ChatChoice> }
#[derive(Deserialize)]
struct ChatChoice  { message: ChatChoiceMsg }
#[derive(Deserialize)]
struct ChatChoiceMsg { #[serde(default)] content: Option<String> }
