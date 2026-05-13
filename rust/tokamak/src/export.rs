//! Export to Axolotl / ShareGPT / Unsloth formats. Carries the `signal`
//! column through when set.

use serde_json::{json, Map, Value};

use crate::quality::{content_string, extract_messages};

pub const DEFAULT_SYSTEM: &str = "You are a careful, terse reasoning model. Internal reasoning is rendered in terse style: no filler, no hedging, all technical substance preserved.";

fn normalize(messages: Vec<&Map<String, Value>>) -> Vec<Value> {
    messages.into_iter().map(|m| {
        let mut role = m.get("role").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if role == "human" { role = "user".into(); }
        if role == "gpt"   { role = "assistant".into(); }
        json!({ "role": role, "content": content_string(m) })
    }).collect()
}

fn ensure_system(mut msgs: Vec<Value>) -> Vec<Value> {
    let has_system = msgs.iter().any(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"));
    if !has_system {
        msgs.insert(0, json!({"role":"system","content":DEFAULT_SYSTEM}));
    }
    msgs
}

fn signal_of(trace: &Value) -> Option<f64> {
    if let Some(md) = trace.get("metadata") {
        if let Some(s) = md.get("signal").and_then(|v| v.as_f64()) { return Some(s); }
    }
    trace.get("signal").and_then(|v| v.as_f64())
}

pub fn axolotl(trace: &Value, score: f64, trace_id: &str, error_class: &str) -> Value {
    let msgs = ensure_system(normalize(extract_messages(trace)));
    let mut metadata = json!({
        "quality_score": (score * 10000.0).round() / 10000.0,
        "error_class": error_class,
    });
    if let Some(s) = signal_of(trace) {
        metadata["signal"] = json!(s);
    }
    json!({ "id": trace_id, "messages": msgs, "metadata": metadata })
}

pub fn sharegpt(trace: &Value) -> Value {
    let msgs = ensure_system(normalize(extract_messages(trace)));
    let convos: Vec<Value> = msgs.iter().map(|m| {
        let role = m["role"].as_str().unwrap_or("");
        let mapped = match role { "user"=>"human", "assistant"=>"gpt", _=>role };
        json!({ "from": mapped, "value": m["content"] })
    }).collect();
    json!({ "conversations": convos })
}

pub fn unsloth(trace: &Value, score: f64, trace_id: &str, error_class: &str) -> Value {
    let msgs = ensure_system(normalize(extract_messages(trace)));
    let mut v = json!({
        "id": trace_id,
        "messages": msgs,
        "score": (score * 10000.0).round() / 10000.0,
        "error_class": error_class,
    });
    if let Some(s) = signal_of(trace) {
        v["signal"] = json!(s);
    }
    v
}
