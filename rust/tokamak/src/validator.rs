//! QAQC validator agent. Three-axis grade (logical/content/safety) on 0..1,
//! signal = worst axis. Mirrored validators take worst-judge across N.

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::llm::LlmClient;
use crate::processor::SpanInput;
use crate::prompts;

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub signal: f64,
    pub logical_fidelity: f64,
    pub content_fidelity: f64,
    pub safety: f64,
    pub notes: String,
    pub per_judge: Vec<f64>,
}

static JSON_OBJ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\{.*\}").unwrap());

pub fn parse_score(raw: &str) -> ValidationResult {
    let Some(m) = JSON_OBJ_RE.find(raw) else {
        return ValidationResult {
            signal: 0.0, logical_fidelity: 0.0, content_fidelity: 0.0, safety: 0.0,
            notes: "parse_error".into(), per_judge: vec![],
        };
    };
    let Ok(v): Result<Value, _> = serde_json::from_str(m.as_str()) else {
        return ValidationResult {
            signal: 0.0, logical_fidelity: 0.0, content_fidelity: 0.0, safety: 0.0,
            notes: "parse_error".into(), per_judge: vec![],
        };
    };

    let clamp = |x: f64| if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
    let f = |k: &str| -> f64 {
        v.get(k).and_then(|x| x.as_f64()).map(clamp).unwrap_or(0.0)
    };

    let logical = f("logical_fidelity");
    let content = f("content_fidelity");
    let safety  = f("safety");
    let signal = if v.get("signal").is_some() {
        f("signal")
    } else {
        logical.min(content).min(safety)
    };
    let notes = v.get("notes").and_then(|x| x.as_str()).unwrap_or("").to_string();

    ValidationResult { signal, logical_fidelity: logical, content_fidelity: content, safety, notes, per_judge: vec![] }
}

fn format_validate_user(s: &SpanInput, processed: &str, mode: &str) -> String {
    format!(
        "PROBLEM:\n{}\n\nANSWER:\n{}\n\nMODE: {}\n\nORIGINAL:\n{}\n\nPROCESSED:\n{}",
        if s.problem.trim().is_empty() { "(none)" } else { s.problem.trim() },
        if s.answer.trim().is_empty()  { "(none)" } else { s.answer.trim()  },
        mode, s.reasoning, processed,
    )
}

pub struct Validator { pub mirrors: usize }

impl Validator {
    pub fn new(mirrors: usize) -> Self { Self { mirrors: mirrors.max(1) } }

    /// Validate every (span, processed) pair concurrently. The total
    /// concurrency cap is `max_workers`; each item may further consume up
    /// to `mirrors` slots (one per mirrored judge).
    pub async fn validate_batch(
        &self,
        client: LlmClient,
        items: Vec<(SpanInput, String, &'static str)>,   // span, processed, mode_name
        max_workers: usize,
    ) -> Vec<ValidationResult> {
        if items.is_empty() { return Vec::new(); }
        let sem = Arc::new(Semaphore::new(max_workers.max(1)));
        let me = Arc::new(client);
        let mirrors = self.mirrors;

        let mut item_handles = Vec::with_capacity(items.len());
        for (span, processed, mode_name) in items.into_iter() {
            let sem = Arc::clone(&sem);
            let me  = Arc::clone(&me);
            item_handles.push(tokio::spawn(async move {
                grade_one(me, sem, mirrors, span, processed, mode_name).await
            }));
        }
        let mut out = Vec::with_capacity(item_handles.len());
        for h in item_handles {
            out.push(h.await.unwrap_or_else(|_| ValidationResult {
                signal: 0.0, logical_fidelity: 0.0, content_fidelity: 0.0, safety: 0.0,
                notes: "join_error".into(), per_judge: vec![],
            }));
        }
        out
    }
}

async fn grade_one(
    client: Arc<LlmClient>,
    sem: Arc<Semaphore>,
    mirrors: usize,
    span: SpanInput,
    processed: String,
    mode_name: &'static str,
) -> ValidationResult {
    let system = prompts::validate_prompt().to_string();
    let user = format_validate_user(&span, &processed, mode_name);
    let mut judge_handles = Vec::with_capacity(mirrors);
    for _ in 0..mirrors {
        let sem = Arc::clone(&sem);
        let client = Arc::clone(&client);
        let s = system.clone();
        let u = user.clone();
        judge_handles.push(tokio::spawn(async move {
            let _p = sem.acquire_owned().await.expect("semaphore");
            let raw = client.chat(&s, &u).await.unwrap_or_default();
            parse_score(&raw)
        }));
    }
    let mut judges: Vec<ValidationResult> = Vec::with_capacity(judge_handles.len());
    for h in judge_handles {
        if let Ok(r) = h.await { judges.push(r); }
    }
    if judges.is_empty() {
        return ValidationResult { signal: 0.0, logical_fidelity: 0.0, content_fidelity: 0.0, safety: 0.0,
                                  notes: "all_judges_failed".into(), per_judge: vec![] };
    }
    let per: Vec<f64> = judges.iter().map(|r| r.signal).collect();
    let worst = judges.into_iter().min_by(|a, b| a.signal.partial_cmp(&b.signal).unwrap()).unwrap();
    ValidationResult { per_judge: per, ..worst }
}
