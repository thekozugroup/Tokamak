//! Lexical deduplication. 5-gram shingle Jaccard over a normalised
//! signature that strips boilerplate tool-call JSON.

use std::collections::HashSet;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::quality::{content_string, extract_messages};

static TOOL_BLOCK: Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?is)<(tool_call|function_calls|tool_result|output)>.*?</(tool_call|function_calls|tool_result|output)>"
).unwrap());

static WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

fn signature(trace: &Value) -> String {
    let mut parts = Vec::new();
    for m in extract_messages(trace) {
        let mut text = content_string(m);
        text = TOOL_BLOCK.replace_all(&text, "").into_owned();
        text = WS_RE.replace_all(&text, " ").trim().to_string();
        if text.is_empty() { continue; }
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("?");
        parts.push(format!("{role}:{text}"));
    }
    parts.join("\n")
}

fn shingles(text: &str, k: usize) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < k {
        if words.is_empty() { return HashSet::new(); }
        return HashSet::from([words.join(" ")]);
    }
    (0..=words.len() - k)
        .map(|i| words[i..i+k].join(" "))
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

/// Returns (kept, dropped_count).
pub fn dedup(traces: Vec<Value>, max_similarity: f64) -> (Vec<Value>, usize) {
    let mut kept: Vec<Value> = Vec::with_capacity(traces.len());
    let mut shingle_sets: Vec<HashSet<String>> = Vec::with_capacity(traces.len());
    let mut dropped = 0usize;

    for trace in traces {
        let sig = signature(&trace);
        let sh = shingles(&sig, 5);
        let dup = shingle_sets.iter().any(|existing| jaccard(&sh, existing) >= max_similarity);
        if dup { dropped += 1; continue; }
        kept.push(trace);
        shingle_sets.push(sh);
    }
    (kept, dropped)
}
