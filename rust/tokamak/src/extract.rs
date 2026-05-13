//! Locate reasoning spans across heterogeneous trace formats and capture
//! the surrounding **problem + answer** context the agents need as rails.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

/// Each pattern matches a single tag pair. The closing tag is duplicated
/// inline since Rust's `regex` crate does not support backreferences.
static TAG_RES: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    let tags = ["thinking", "think", "reasoning", "thought", "analyze", "scratchpad"];
    tags.into_iter()
        .map(|t| {
            let pat = format!(r"(?is)<{tag}>(.*?)</{tag}>", tag = t);
            (t, Regex::new(&pat).unwrap())
        })
        .collect()
});

/// A reasoning span located in a trace, with handles to rewrite it back.
///
/// We don't carry closures — instead we record the JSON pointer-style path
/// to the host string and the original outer text to splice over. The
/// pipeline calls `apply()` to perform the rewrite.
#[derive(Debug, Clone)]
pub struct Span {
    pub source: String,        // "tag:thinking" / "block:thinking" / "field:reasoning"
    pub text: String,          // current reasoning text
    pub problem: String,
    pub answer: String,

    // Where to write the result back:
    pub path: Vec<PathStep>,   // path from trace root to the host string field
    pub kind: SpanKind,
}

#[derive(Debug, Clone)]
pub enum PathStep { Key(String), Idx(usize) }

#[derive(Debug, Clone)]
pub enum SpanKind {
    /// The reasoning lives inside `<tag>...</tag>` of the host string;
    /// rewrite by replacing the outer tag-and-body.
    TagBody { tag: String, outer_original: String },
    /// The host field is itself the reasoning text; rewrite by replacing it.
    Field,
}

/// Walk a trace and yield every editable reasoning span we can find.
pub fn find_reasoning_spans(trace: &Value) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let Value::Object(root) = trace else { return out; };
    let Some(Value::Array(messages)) = root.get("messages") else {
        return walk_fallback(trace);
    };

    for (ai, msg) in messages.iter().enumerate() {
        let Value::Object(m) = msg else { continue; };
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(role, "assistant" | "model") { continue; }

        let problem = gather_problem(messages, ai);
        let answer  = extract_answer(m);

        // Case A: content is a tagged string.
        if let Some(Value::String(s)) = m.get("content") {
            for (tag, re) in TAG_RES.iter() {
                for cap in re.captures_iter(s) {
                    let outer = cap.get(0).unwrap().as_str().to_string();
                    let inner = cap.get(1).unwrap().as_str().to_string();
                    out.push(Span {
                        source:  format!("tag:{tag}"),
                        text:    inner,
                        problem: problem.clone(),
                        answer:  answer.clone(),
                        path: vec![
                            PathStep::Key("messages".into()),
                            PathStep::Idx(ai),
                            PathStep::Key("content".into()),
                        ],
                        kind: SpanKind::TagBody {
                            tag: (*tag).to_string(),
                            outer_original: outer,
                        },
                    });
                }
            }
            continue;
        }

        // Case B: content is a content-block list.
        if let Some(Value::Array(blocks)) = m.get("content") {
            for (bi, b) in blocks.iter().enumerate() {
                let Value::Object(block) = b else { continue; };

                // Anthropic-style thinking block
                if block.get("type").and_then(|v| v.as_str()) == Some("thinking") {
                    if let Some(Value::String(thinking)) = block.get("thinking") {
                        out.push(Span {
                            source: "block:thinking".into(),
                            text: thinking.clone(),
                            problem: problem.clone(),
                            answer:  answer.clone(),
                            path: vec![
                                PathStep::Key("messages".into()),
                                PathStep::Idx(ai),
                                PathStep::Key("content".into()),
                                PathStep::Idx(bi),
                                PathStep::Key("thinking".into()),
                            ],
                            kind: SpanKind::Field,
                        });
                    }
                }

                for fld in ["reasoning", "thought", "analysis"] {
                    if let Some(Value::String(val)) = block.get(fld) {
                        if !val.trim().is_empty() {
                            out.push(Span {
                                source: format!("field:{fld}"),
                                text: val.clone(),
                                problem: problem.clone(),
                                answer:  answer.clone(),
                                path: vec![
                                    PathStep::Key("messages".into()),
                                    PathStep::Idx(ai),
                                    PathStep::Key("content".into()),
                                    PathStep::Idx(bi),
                                    PathStep::Key(fld.into()),
                                ],
                                kind: SpanKind::Field,
                            });
                        }
                    }
                }
            }
        }

        // Case C: OpenAI o1-style `reasoning` field on the message itself.
        for fld in ["reasoning", "thinking", "analysis", "thought"] {
            if let Some(Value::String(val)) = m.get(fld) {
                if !val.trim().is_empty() {
                    out.push(Span {
                        source: format!("field:{fld}"),
                        text: val.clone(),
                        problem: problem.clone(),
                        answer:  answer.clone(),
                        path: vec![
                            PathStep::Key("messages".into()),
                            PathStep::Idx(ai),
                            PathStep::Key(fld.into()),
                        ],
                        kind: SpanKind::Field,
                    });
                }
            }
        }
    }

    out
}

/// Fallback walker for traces without a top-level `messages` array.
fn walk_fallback(trace: &Value) -> Vec<Span> {
    let mut out = Vec::new();
    walk(trace, &mut Vec::new(), &mut out);
    out
}

fn walk(node: &Value, path: &mut Vec<PathStep>, out: &mut Vec<Span>) {
    match node {
        Value::String(s) => {
            for (tag, re) in TAG_RES.iter() {
                for cap in re.captures_iter(s) {
                    let outer = cap.get(0).unwrap().as_str().to_string();
                    let inner = cap.get(1).unwrap().as_str().to_string();
                    out.push(Span {
                        source: format!("tag:{tag}"),
                        text: inner,
                        problem: String::new(),
                        answer:  String::new(),
                        path: path.clone(),
                        kind: SpanKind::TagBody {
                            tag: (*tag).to_string(),
                            outer_original: outer,
                        },
                    });
                }
            }
        }
        Value::Object(obj) => {
            for (k, v) in obj.iter() {
                path.push(PathStep::Key(k.clone()));
                walk(v, path, out);
                path.pop();
            }
        }
        Value::Array(arr) => {
            for (i, v) in arr.iter().enumerate() {
                path.push(PathStep::Idx(i));
                walk(v, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

// ---- Context helpers ----

fn gather_problem(messages: &[Value], assistant_idx: usize) -> String {
    let mut parts = Vec::new();
    for m in &messages[..assistant_idx] {
        let Value::Object(o) = m else { continue; };
        let role = o.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(role, "user" | "system" | "human") { continue; }
        let txt = content_string(o);
        if !txt.trim().is_empty() { parts.push(txt.trim().to_string()); }
    }
    parts.join("\n\n")
}

fn extract_answer(m: &Map<String, Value>) -> String {
    match m.get("content") {
        Some(Value::String(s)) => strip_reasoning(s).trim().to_string(),
        Some(Value::Array(blocks)) => {
            let chunks: Vec<String> = blocks.iter().filter_map(|b| {
                let Value::Object(block) = b else { return None; };
                if block.get("type").and_then(|v| v.as_str()) != Some("text") {
                    return None;
                }
                block.get("text").and_then(|v| v.as_str())
                    .map(|t| strip_reasoning(t))
            }).collect();
            chunks.join("\n\n").trim().to_string()
        }
        _ => String::new(),
    }
}

fn content_string(m: &Map<String, Value>) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            blocks.iter().filter_map(|b| {
                let Value::Object(block) = b else { return None; };
                block.get("text").and_then(|v| v.as_str()).map(|s| s.to_string())
            }).collect::<Vec<_>>().join("\n")
        }
        _ => String::new(),
    }
}

fn strip_reasoning(text: &str) -> String {
    let mut out = text.to_string();
    for (_tag, re) in TAG_RES.iter() {
        out = re.replace_all(&out, "").into_owned();
    }
    out
}

// ---- Apply replacement back into the trace ----

pub fn apply_replacement(trace: &mut Value, span: &Span, new_inner: &str) {
    // Walk to the parent value.
    let mut cursor: &mut Value = trace;
    for step in &span.path[..span.path.len().saturating_sub(1)] {
        cursor = match (step, cursor) {
            (PathStep::Key(k), Value::Object(o)) => o.get_mut(k).unwrap(),
            (PathStep::Idx(i), Value::Array(a))  => a.get_mut(*i).unwrap(),
            _ => return,
        };
    }
    let Some(last) = span.path.last() else { return; };
    let target: &mut Value = match (last, cursor) {
        (PathStep::Key(k), Value::Object(o)) => o.get_mut(k).unwrap(),
        (PathStep::Idx(i), Value::Array(a))  => a.get_mut(*i).unwrap(),
        _ => return,
    };

    match &span.kind {
        SpanKind::Field => {
            *target = Value::String(new_inner.to_string());
        }
        SpanKind::TagBody { tag, outer_original } => {
            if let Value::String(s) = target {
                let replacement = format!("<{tag}>{new_inner}</{tag}>");
                *s = s.replacen(outer_original, &replacement, 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finds_thinking_tag() {
        let t = json!({"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<thinking>internal step</thinking>\nAnswer."},
        ]});
        let spans = find_reasoning_spans(&t);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].source, "tag:thinking");
        assert!(spans[0].text.contains("internal step"));
        assert!(spans[0].problem.contains("Hi"));
        assert!(spans[0].answer.contains("Answer."));
    }

    #[test]
    fn apply_rewrites_in_place() {
        let mut t = json!({"messages": [
            {"role": "assistant", "content": "<thinking>orig</thinking>\nA"}
        ]});
        let spans = find_reasoning_spans(&t);
        apply_replacement(&mut t, &spans[0], "compressed");
        let s = t["messages"][0]["content"].as_str().unwrap();
        assert!(s.contains("<thinking>compressed</thinking>"));
    }

    #[test]
    fn anthropic_thinking_block() {
        let t = json!({"messages":[{"role":"assistant","content":[
            {"type":"thinking","thinking":"hidden chain"},
            {"type":"text","text":"visible answer"},
        ]}]});
        let spans = find_reasoning_spans(&t);
        assert!(spans.iter().any(|s| s.source == "block:thinking"));
    }
}
