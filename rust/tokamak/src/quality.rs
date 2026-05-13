//! Six-dimension quality scoring. Reported per row; never used as a filter.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

pub const THINKING_TAGS: &[&str] = &["<thinking>", "<reasoning>", "<thought>", "<analyze>", "<scratchpad>"];
pub const TOOL_CALL_TAGS: &[&str] = &["<tool_call>", "<function_calls>", "<invoke>"];

static REFUSAL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)i\s+can'?t?\s+(?:help|do|assist)|i'?m\s+sorry|i\s+(?:don't|do not)\s+know|unable\s+to|not\s+(?:able|allowed)\s+to"
).unwrap());

pub fn extract_messages(trace: &Value) -> Vec<&Map<String, Value>> {
    let mut out = Vec::new();
    if let Some(Value::Array(arr)) = trace.get("messages") {
        for v in arr { if let Value::Object(m) = v { out.push(m); } }
        return out;
    }
    if let Some(Value::Array(arr)) = trace.get("conversations") {
        for v in arr { if let Value::Object(m) = v { out.push(m); } }
    }
    out
}

pub fn content_string(msg: &Map<String, Value>) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks.iter().filter_map(|b| {
            if let Value::Object(o) = b {
                o.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())
            } else { Some(b.to_string()) }
        }).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

fn all_text(trace: &Value) -> String {
    serde_json::to_string(trace).unwrap_or_default()
}

pub struct Score {
    pub composite: f64,
    pub reasoning_depth: f64,
    pub structure: f64,
    pub tool_calls: f64,
    pub coherence: f64,
    pub length: f64,
    pub refusal: f64,
}

pub fn score(trace: &Value) -> Score {
    let s = Score {
        reasoning_depth: score_reasoning_depth(trace),
        structure:       score_structure(trace),
        tool_calls:      score_tool_calls(trace),
        coherence:       score_coherence(trace),
        length:          score_length(trace),
        refusal:         score_refusal(trace),
        composite: 0.0,
    };
    let composite = 0.20*s.reasoning_depth + 0.20*s.structure + 0.15*s.tool_calls
                  + 0.15*s.coherence + 0.15*s.length + 0.15*s.refusal;
    Score { composite, ..s }
}

fn score_reasoning_depth(trace: &Value) -> f64 {
    let text = all_text(trace).to_lowercase();
    let msgs = extract_messages(trace);
    let thinking_hits = THINKING_TAGS.iter().filter(|t| text.contains(*t)).count() as f64;
    let substantial   = msgs.iter().filter(|m| content_string(m).len() > 100).count() as f64;
    let raw = 0.4 * (thinking_hits / 2.0).min(1.0) + 0.6 * (substantial / 4.0).min(1.0);
    raw.min(1.0)
}

fn score_structure(trace: &Value) -> f64 {
    let msgs = extract_messages(trace);
    if msgs.is_empty() { return 0.0; }
    let roles: Vec<&str> = msgs.iter().map(|m| m.get("role").and_then(|v| v.as_str()).unwrap_or("")).collect();
    let has_user = roles.iter().any(|r| matches!(*r, "user" | "human"));
    let has_asst = roles.iter().any(|r| matches!(*r, "assistant" | "gpt"));
    if !(has_user && has_asst) { return 0.0; }
    0.5 + 0.5 * (msgs.len() as f64 / 4.0).min(1.0)
}

fn score_tool_calls(trace: &Value) -> f64 {
    let text = all_text(trace).to_lowercase();
    if TOOL_CALL_TAGS.iter().any(|t| text.contains(*t)) { 1.0 } else { 0.5 }
}

fn score_coherence(trace: &Value) -> f64 {
    let msgs = extract_messages(trace);
    if msgs.len() < 2 { return 0.5; }
    let asst: Vec<String> = msgs.iter()
        .filter(|m| matches!(m.get("role").and_then(|v| v.as_str()).unwrap_or(""), "assistant" | "gpt"))
        .map(|m| content_string(m)).collect();
    if asst.is_empty() { return 0.4; }
    let unique = asst.iter().collect::<std::collections::HashSet<_>>().len() as f64;
    (unique / asst.len().max(1) as f64).min(1.0)
}

fn score_length(trace: &Value) -> f64 {
    let n = all_text(trace).len();
    if n < 256 { 0.2 } else if n > 100_000 { 0.5 } else { 1.0 }
}

fn score_refusal(trace: &Value) -> f64 {
    let text = all_text(trace);
    if REFUSAL_RE.is_match(&text) { 0.4 } else { 1.0 }
}
