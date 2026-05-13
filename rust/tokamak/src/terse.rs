//! Rule-based terse rewrite. Free, fast, ~10–25% reduction.
//!
//! Mirrors the Python regex pipeline in `src/tokamak/caveman.py`:
//! 1. Mask "protected" spans (code, paths, URLs, quoted strings, numbers).
//! 2. Strip filler / pleasantries / hedging.
//! 3. Verbose-phrase rewrites ("in order to" → "to").
//! 4. (full only) drop articles + arrow causality.
//! 5. Collapse whitespace.
//! 6. Restore protected spans byte-for-byte.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::prompts::Level;

/// Token estimate used everywhere. ~4 chars/token English, 1 char/token CJK.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() { return 0; }
    let mut cjk = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        let c = ch as u32;
        if (0x4E00..=0x9FFF).contains(&c) { cjk += 1; } else { other += 1; }
    }
    cjk + (other / 4).max(1)
}

#[derive(Debug, Clone)]
pub struct CompressResult {
    pub original: String,
    pub compressed: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
}

// ---- Protected span patterns ----

static PROTECT_RE: Lazy<Vec<Regex>> = Lazy::new(|| vec![
    Regex::new(r"(?s)```.*?```").unwrap(),
    Regex::new(r"(?s)~~~.*?~~~").unwrap(),
    Regex::new(r"`[^`\n]+`").unwrap(),
    Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap(),
    Regex::new(r"(?s)<function_calls>.*?</function_calls>").unwrap(),
    Regex::new(r"(?s)<tool_result>.*?</tool_result>").unwrap(),
    Regex::new(r"(?s)<output>.*?</output>").unwrap(),
    Regex::new(r"https?://\S+").unwrap(),
    Regex::new(r"/[\w./\-]+").unwrap(),
    Regex::new(r"[A-Za-z]:\\[\w.\\\-]+").unwrap(),
    Regex::new(r#""[^"\n]{0,200}""#).unwrap(),
    Regex::new(r"'[^'\n]{0,200}'").unwrap(),
    Regex::new(r"\b\d+(?:\.\d+)*\b").unwrap(),
]);

fn protect(text: &str) -> (String, Vec<String>) {
    let mut out = text.to_string();
    let mut originals: Vec<String> = Vec::new();
    for re in PROTECT_RE.iter() {
        let mut result = String::with_capacity(out.len());
        let mut last = 0usize;
        for m in re.find_iter(&out) {
            result.push_str(&out[last..m.start()]);
            let placeholder = format!("\x00P{}\x00", originals.len());
            originals.push(out[m.start()..m.end()].to_string());
            result.push_str(&placeholder);
            last = m.end();
        }
        result.push_str(&out[last..]);
        out = result;
    }
    (out, originals)
}

fn restore(text: &str, originals: &[String]) -> String {
    static PLACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\x00P(\d+)\x00").unwrap());
    PLACE_RE
        .replace_all(text, |c: &regex::Captures| {
            let idx: usize = c[1].parse().unwrap();
            originals.get(idx).cloned().unwrap_or_default()
        })
        .into_owned()
}

// ---- Wordlists ----

const FILLER: &[&str] = &[
    "just", "really", "basically", "actually", "simply", "literally",
    "obviously", "clearly", "essentially", "fundamentally", "honestly",
    "indeed", "very", "quite", "rather", "somewhat", "kind of", "sort of",
];

const PLEASANTRIES: &[&str] = &[
    "of course", "sure thing", "no problem", "happy to help",
    "let me think", "let me see", "let's see", "okay so", "ok so",
    "alright so", "alright", "great", "perfect", "awesome",
    "first off", "first of all", "to begin with",
    "i'll start by", "i will start by", "let me start by",
    "i want to", "i'm going to", "i am going to", "i'll go ahead and",
    "now i", "now let me", "now let's", "next i'll", "next let me",
];

const HEDGES: &[&str] = &[
    "i think", "i believe", "i suppose", "i guess", "i'd say",
    "it seems", "it appears", "it would seem", "perhaps", "maybe",
    "possibly", "probably", "i'm not sure but", "if i recall correctly",
];

fn build_wordlist_pattern(words: &[&str]) -> Regex {
    let mut sorted: Vec<&&str> = words.iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let alts = sorted.iter().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|");
    Regex::new(&format!(r"(?i)\b(?:{alts})\b[ \t]*,?\s*")).unwrap()
}

static FILLER_RE:        Lazy<Regex> = Lazy::new(|| build_wordlist_pattern(FILLER));
static PLEASANTRIES_RE:  Lazy<Regex> = Lazy::new(|| build_wordlist_pattern(PLEASANTRIES));
static HEDGES_RE:        Lazy<Regex> = Lazy::new(|| build_wordlist_pattern(HEDGES));

static OPENER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?im)(?:^|\n|(?:[.!?]\s))\s*(?:sure|okay|ok|right|well|yeah|yep|hmm)[\s,!.\-:]+").unwrap()
});

struct Repl { pat: Regex, repl: &'static str }

static REPLACEMENTS: Lazy<Vec<Repl>> = Lazy::new(|| vec![
    Repl { pat: Regex::new(r"(?i)\bin order to\b").unwrap(),                repl: "to" },
    Repl { pat: Regex::new(r"(?i)\bdue to the fact that\b").unwrap(),       repl: "because" },
    Repl { pat: Regex::new(r"(?i)\bas a result of\b").unwrap(),             repl: "due to" },
    Repl { pat: Regex::new(r"(?i)\bat this point in time\b").unwrap(),      repl: "now" },
    Repl { pat: Regex::new(r"(?i)\bat this moment\b").unwrap(),             repl: "now" },
    Repl { pat: Regex::new(r"(?i)\bin the event that\b").unwrap(),          repl: "if" },
    Repl { pat: Regex::new(r"(?i)\bmake use of\b").unwrap(),                repl: "use" },
    Repl { pat: Regex::new(r"(?i)\btake into account\b").unwrap(),          repl: "consider" },
    Repl { pat: Regex::new(r"(?i)\bgive consideration to\b").unwrap(),      repl: "consider" },
    Repl { pat: Regex::new(r"(?i)\bhas the ability to\b").unwrap(),         repl: "can" },
    Repl { pat: Regex::new(r"(?i)\bis able to\b").unwrap(),                 repl: "can" },
    Repl { pat: Regex::new(r"(?i)\bin spite of the fact that\b").unwrap(),  repl: "though" },
    Repl { pat: Regex::new(r"(?i)\bwith regard to\b").unwrap(),             repl: "re" },
    Repl { pat: Regex::new(r"(?i)\bwith respect to\b").unwrap(),            repl: "re" },
    Repl { pat: Regex::new(r"(?i)\bin terms of\b").unwrap(),                repl: "for" },
    Repl { pat: Regex::new(r"(?i)\ba large number of\b").unwrap(),          repl: "many" },
    Repl { pat: Regex::new(r"(?i)\ba small number of\b").unwrap(),          repl: "few" },
    Repl { pat: Regex::new(r"(?i)\bthe vast majority of\b").unwrap(),       repl: "most" },
    Repl { pat: Regex::new(r"(?i)\bimplement a solution for\b").unwrap(),   repl: "fix" },
    Repl { pat: Regex::new(r"(?i)\bverify that\b").unwrap(),                repl: "check" },
    Repl { pat: Regex::new(r"(?i)\bdetermine whether\b").unwrap(),          repl: "check if" },
    Repl { pat: Regex::new(r"(?i)\bnotwithstanding\b").unwrap(),            repl: "though" },
]);

static ARTICLE_RE:   Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)(?<=\S )\b(?:a|an|the)\b\s+").unwrap_or_else(|_| {
    // regex crate doesn't support lookbehind; fall back to a simpler pattern.
    Regex::new(r"(?i)\b(?:a|an|the)\b\s+").unwrap()
}));
static CAUSAL_RE:    Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\b(?:so that|so)\b\s+").unwrap());

static MULTI_WS_RE:  Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]+").unwrap());
static NL_WS_RE:     Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]*\n[ \t]*").unwrap());
static MULTI_NL_RE:  Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static PUNCT_WS_RE:  Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+([.,;:!?])").unwrap());
static ANY_WS_RE:    Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

fn collapse_whitespace(s: &str) -> String {
    let s = MULTI_WS_RE.replace_all(s, " ");
    let s = NL_WS_RE.replace_all(&s, "\n");
    let s = MULTI_NL_RE.replace_all(&s, "\n\n");
    let s = PUNCT_WS_RE.replace_all(&s, "$1");
    let s = ANY_WS_RE.replace_all(&s, " ");
    s.trim().to_string()
}

pub fn compress_rules(text: &str, level: Level) -> CompressResult {
    if text.trim().is_empty() {
        return CompressResult {
            original: text.to_string(), compressed: text.to_string(),
            original_tokens: 0, compressed_tokens: 0,
        };
    }

    let (masked, originals) = protect(text);
    let mut out = OPENER_RE.replace_all(&masked, "").into_owned();
    out = PLEASANTRIES_RE.replace_all(&out, "").into_owned();
    out = HEDGES_RE.replace_all(&out, "").into_owned();
    out = FILLER_RE.replace_all(&out, "").into_owned();
    for r in REPLACEMENTS.iter() {
        out = r.pat.replace_all(&out, r.repl).into_owned();
    }
    if level == Level::Full {
        out = ARTICLE_RE.replace_all(&out, "").into_owned();
        out = CAUSAL_RE.replace_all(&out, "-> ").into_owned();
    }
    out = collapse_whitespace(&out);
    let restored = restore(&out, &originals);

    CompressResult {
        original_tokens: estimate_tokens(text),
        compressed_tokens: estimate_tokens(&restored),
        original: text.to_string(),
        compressed: restored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_filler_and_pleasantries() {
        let r = compress_rules(
            "Sure, let me think. Basically, the bug is just an off-by-one in the loop.",
            Level::Lite,
        );
        let low = r.compressed.to_lowercase();
        for w in ["sure", "let me think", "basically", "just"] {
            assert!(!low.contains(w), "expected {w:?} dropped: {}", r.compressed);
        }
        assert!(r.compressed.contains("off-by-one"));
        assert!(r.compressed.contains("loop"));
    }

    #[test]
    fn preserves_code_blocks() {
        let src = "Okay so basically wrap it in useMemo:\n```js\nconst x = useMemo(() => ({ a: 1 }), []);\n```\nThat should fix it actually.";
        let r = compress_rules(src, Level::Lite);
        assert!(r.compressed.contains("useMemo(() => ({ a: 1 }), [])"));
        let low = r.compressed.to_lowercase();
        assert!(!low.contains("okay so"));
        assert!(!low.contains("basically"));
        assert!(!low.contains("actually"));
    }

    #[test]
    fn token_estimate_drops() {
        let src = "Sure! I'd be happy to help you with that. The reason your component is re-rendering is basically that you're creating a new object reference each render. I think you should just wrap it in useMemo.";
        let r = compress_rules(src, Level::Lite);
        assert!(r.compressed_tokens < r.original_tokens);
    }
}
