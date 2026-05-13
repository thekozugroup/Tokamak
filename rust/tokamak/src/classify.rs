//! 5-factor error taxonomy.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

pub const ERROR_CLASSES: &[&str] = &[
    "tool_failure", "syntax_error", "reasoning_error",
    "safety_refusal", "timeout_stall", "none",
];

static TOOL_FAIL: Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)\b(?:tool\s+(?:failed|error|timeout)|exit\s+code\s+[1-9]|command\s+not\s+found|error\s*:\s*\w+|traceback)"
).unwrap());
static SYNTAX:    Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)\b(syntaxerror|parseerror|unexpected\s+token|invalid\s+syntax|unmatched\s+(?:bracket|paren|brace))"
).unwrap());
static REASONING: Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)\b(i\s+made\s+an?\s+(?:mistake|error)|that\s+was\s+wrong|let\s+me\s+reconsider|i\s+was\s+(?:wrong|incorrect)|actually,?\s+(?:that's|that\s+is)\s+(?:wrong|incorrect))"
).unwrap());
static SAFETY:    Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)\b(i\s+can'?t\s+(?:help|assist|do)|i'?m\s+(?:not\s+able|unable)|i\s+(?:must|cannot)\s+decline|against\s+my\s+guidelines)"
).unwrap());
static TIMEOUT:   Lazy<Regex> = Lazy::new(|| Regex::new(
    r"(?i)\b(timeout|timed\s+out|deadline\s+exceeded|hung|stalled|no\s+response|connection\s+reset)"
).unwrap());

pub fn classify(trace: &Value) -> &'static str {
    let text = serde_json::to_string(trace).unwrap_or_default();
    if TOOL_FAIL.is_match(&text)  { return "tool_failure"; }
    if SYNTAX.is_match(&text)     { return "syntax_error"; }
    if SAFETY.is_match(&text)     { return "safety_refusal"; }
    if TIMEOUT.is_match(&text)    { return "timeout_stall"; }
    if REASONING.is_match(&text)  { return "reasoning_error"; }
    "none"
}
