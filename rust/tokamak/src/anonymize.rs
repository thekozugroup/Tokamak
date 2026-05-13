//! Regex + entropy PII redaction. Walks the trace JSON and rewrites every
//! string in place.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

struct Pat { re: Regex, repl: &'static str }

static PATTERNS: Lazy<Vec<Pat>> = Lazy::new(|| {
    let new = |p: &str, r: &'static str| Pat { re: Regex::new(p).unwrap(), repl: r };
    vec![
        new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",         "<EMAIL>"),
        new(r"sk-(?:ant|live|test)?[_-]?[a-zA-Z0-9_-]{20,}",            "<API_KEY>"),
        new(r"AKIA[0-9A-Z]{16}",                                        "<AWS_KEY>"),
        new(r"ghp_[A-Za-z0-9]{36,}",                                    "<GH_TOKEN>"),
        new(r"\b\d{3}-\d{2}-\d{4}\b",                                   "<SSN>"),
        new(r"\b(?:\d[ -]?){13,19}\b",                                  "<CARD>"),
        new(r"/(?:home|Users)/[A-Za-z0-9_.-]+",                         "<HOME>"),
        new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                             "<IP>"),
        new(r"(?s)-----BEGIN [A-Z ]+ KEY-----.*?-----END [A-Z ]+ KEY-----", "<PRIVATE_KEY>"),
    ]
});

static TOKEN_LIKE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9_\-]{24,}").unwrap());

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Level { Standard, Strict }

impl Level {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "standard" => Ok(Self::Standard),
            "strict"   => Ok(Self::Strict),
            other      => anyhow::bail!("unknown anonymize level: {other:?}"),
        }
    }
}

fn entropy(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let n = s.len() as f64;
    let mut counts = std::collections::HashMap::<char, usize>::new();
    for c in s.chars() { *counts.entry(c).or_insert(0) += 1; }
    counts.values()
        .map(|&c| { let p = c as f64 / n; -p * p.log2() })
        .sum()
}

pub fn redact(text: &str, level: Level) -> (String, usize) {
    if text.is_empty() { return (text.into(), 0); }
    let mut t = text.to_string();
    let mut n = 0usize;

    for pat in PATTERNS.iter() {
        let matches: Vec<_> = pat.re.find_iter(&t).collect();
        let count = matches.len();
        if count > 0 {
            n += count;
            t = pat.re.replace_all(&t, pat.repl).into_owned();
        }
    }

    if level == Level::Strict {
        let (t2, n2) = entropy_redact(&t);
        t = t2; n += n2;
    }
    (t, n)
}

fn entropy_redact(text: &str) -> (String, usize) {
    let mut hits = 0usize;
    let out = TOKEN_LIKE.replace_all(text, |c: &regex::Captures| {
        let tok = c.get(0).unwrap().as_str();
        // Skip obvious code identifiers (snake_case, no digits).
        if tok.contains('_') && !tok.chars().any(|x| x.is_ascii_digit()) {
            return tok.to_string();
        }
        if entropy(tok) >= 4.0 {
            hits += 1;
            "<TOKEN>".to_string()
        } else {
            tok.to_string()
        }
    });
    (out.into_owned(), hits)
}

/// Recursively redact every string field in the trace JSON.
pub fn redact_trace(trace: &mut Value, level: Level) -> usize {
    let mut total = 0usize;
    walk(trace, level, &mut total);
    total
}

fn walk(v: &mut Value, level: Level, total: &mut usize) {
    match v {
        Value::String(s) => {
            let (new, n) = redact(s, level);
            *s = new;
            *total += n;
        }
        Value::Object(o) => { for (_, vv) in o.iter_mut() { walk(vv, level, total); } }
        Value::Array(a)  => { for vv in a.iter_mut() { walk(vv, level, total); } }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_email_and_paths() {
        let (out, n) = redact("contact alice@example.com at /Users/alice/secrets", Level::Standard);
        assert!(out.contains("<EMAIL>"));
        assert!(out.contains("<HOME>"));
        assert!(n >= 2);
    }
}
