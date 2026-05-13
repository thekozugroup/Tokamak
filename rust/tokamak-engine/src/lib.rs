//! Tokamak's concurrent row-scoped orchestrator.
//!
//! One purpose: drive a large batch of short-lived agent calls (compress /
//! invert / validate) against an OpenAI-compatible endpoint with a hard cap
//! on concurrent in-flight requests. Each row is its own short-lived,
//! row-scoped agent — there is no shared mutable state between rows, so
//! Rust's ownership model gives us memory-safe parallel processing for free.
//!
//! Exposed to Python via PyO3:
//!
//! ```python
//! import tokamak_engine
//! results = tokamak_engine.batch_chat(
//!     base_url   = "http://localhost:8000/v1",
//!     api_key    = "dummy",
//!     model      = "Qwen/Qwen2.5-7B-Instruct",
//!     items      = [{"system": "...", "user": "..."}],
//!     max_workers= 64,
//!     timeout_s  = 180.0,
//!     retries    = 2,
//!     max_tokens = 4096,
//!     temperature= 0.0,
//! )
//! ```
//!
//! `results[i]` is either the assistant string for `items[i]`, or `None` if
//! the call failed after all retries. The caller decides fallback policy
//! (Python pipeline falls back to the original uncompressed text).

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Clone, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatResponseChoiceMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponseChoice {
    message: ChatResponseChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatResponseChoice>,
}

#[derive(Debug, Clone)]
struct Item {
    system: String,
    user: String,
}

async fn call_once(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    item: &Item,
    max_tokens: u32,
    temperature: f32,
) -> anyhow::Result<String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = ChatRequest {
        model,
        messages: vec![
            ChatMessage { role: "system", content: &item.system },
            ChatMessage { role: "user",   content: &item.user },
        ],
        max_tokens,
        temperature,
    };
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    let parsed: ChatResponse = resp.json().await?;
    let text = parsed
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();
    Ok(text.trim().to_string())
}

async fn call_with_retry(
    client: Arc<Client>,
    base_url: Arc<String>,
    api_key: Arc<String>,
    model: Arc<String>,
    item: Item,
    max_tokens: u32,
    temperature: f32,
    retries: u32,
) -> Option<String> {
    let mut attempt: u32 = 0;
    loop {
        match call_once(
            &client, &base_url, &api_key, &model, &item, max_tokens, temperature,
        )
        .await
        {
            Ok(s) => return Some(s),
            Err(e) => {
                if attempt >= retries {
                    tracing::warn!("call failed after {} retries: {:?}", retries, e);
                    return None;
                }
                let backoff = std::cmp::min(1u64 << attempt, 30);
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                attempt += 1;
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    base_url,
    api_key,
    model,
    items,
    max_workers = 8,
    timeout_s = 180.0,
    retries = 2,
    max_tokens = 4096,
    temperature = 0.0,
))]
#[allow(clippy::too_many_arguments)]
fn batch_chat(
    py: Python<'_>,
    base_url: String,
    api_key: String,
    model: String,
    items: &Bound<'_, PyList>,
    max_workers: usize,
    timeout_s: f64,
    retries: u32,
    max_tokens: u32,
    temperature: f32,
) -> PyResult<PyObject> {
    // Convert PyList[dict] -> Vec<Item> while we still hold the GIL.
    let parsed: PyResult<Vec<Item>> = items
        .iter()
        .map(|obj| {
            let d = obj.downcast::<PyDict>()?;
            let system: String = d
                .get_item("system")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing key: system"))?
                .extract()?;
            let user: String = d
                .get_item("user")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing key: user"))?
                .extract()?;
            Ok(Item { system, user })
        })
        .collect();
    let items = parsed?;

    // Release the GIL while we run the async batch.
    let results: Vec<Option<String>> = py.allow_threads(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(std::cmp::max(1, max_workers))
            .enable_all()
            .build()
            .expect("build tokio runtime");

        runtime.block_on(async move {
            let client = Arc::new(
                Client::builder()
                    .timeout(Duration::from_secs_f64(timeout_s))
                    .build()
                    .expect("build reqwest client"),
            );
            let base_url = Arc::new(base_url);
            let api_key = Arc::new(api_key);
            let model = Arc::new(model);
            let sem = Arc::new(Semaphore::new(std::cmp::max(1, max_workers)));

            let mut handles = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                let permit = Arc::clone(&sem);
                let client = Arc::clone(&client);
                let base_url = Arc::clone(&base_url);
                let api_key = Arc::clone(&api_key);
                let model = Arc::clone(&model);
                handles.push(tokio::spawn(async move {
                    let _p = permit.acquire_owned().await.expect("semaphore");
                    call_with_retry(
                        client, base_url, api_key, model, item,
                        max_tokens, temperature, retries,
                    )
                    .await
                }));
            }

            let mut out = Vec::with_capacity(handles.len());
            for h in handles {
                out.push(h.await.unwrap_or(None));
            }
            out
        })
    });

    Python::with_gil(|py| {
        let list = PyList::empty_bound(py);
        for r in results {
            match r {
                Some(s) => list.append(s)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list.into())
    })
}

#[pymodule]
fn tokamak_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_chat, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
