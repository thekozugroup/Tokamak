//! Dataset card and per-trace processing report renderers.

use std::collections::BTreeMap;
use serde::Serialize;

#[derive(Debug, Default, Serialize)]
pub struct CardStats {
    pub traces_in: usize,
    pub traces_out: usize,
    pub spans: usize,
    pub original_tokens: usize,
    pub processed_tokens: usize,
    pub redactions: usize,
    pub avg_signal: f64,
}

pub fn render_card(stats: &CardStats, scores: &[f64], errors: &BTreeMap<&str, usize>, repo_id: &str) -> String {
    let n = scores.len() as f64;
    let avg = if n > 0.0 { scores.iter().sum::<f64>() / n } else { 0.0 };
    let err_lines = if errors.is_empty() {
        "- none".to_string()
    } else {
        errors.iter().map(|(k,v)| format!("- `{k}`: {v}")).collect::<Vec<_>>().join("\n")
    };
    let saved = if stats.original_tokens > 0 {
        100.0 * (1.0 - (stats.processed_tokens as f64 / stats.original_tokens as f64))
    } else { 0.0 };
    let signal_line = if stats.avg_signal > 0.0 {
        format!("| Mean QAQC signal | {:.3} |", stats.avg_signal)
    } else {
        "| Mean QAQC signal | _not run_ |".to_string()
    };

    format!(r#"# Tokamak — Terse-Reasoning Trace Dataset

{repo_line}

A reasoning-trace dataset whose internal `<thinking>` / `reasoning` channels
were processed — either compressed (terse rewrite, every step preserved) or
inverted (compressed skeleton expanded into a fuller trace). Final answers,
tool calls, code, and user inputs are byte-identical to the source traces.

## Pipeline

1. Ingest raw JSONL traces
2. Extract reasoning spans + surrounding problem / answer context
3. Process reasoning (compress, invert, or both)
4. QAQC: independent validator grades each span on 0..1
5. Anonymize (regex + entropy)
6. Quality score (6-dim, reported never filtered)
7. Classify error taxonomy
8. Deduplicate
9. Triple export — Axolotl + ShareGPT + Unsloth

## Processing results

| Metric | Value |
|--------|-------|
| Traces in | {ti} |
| Traces out (post-dedup) | {to} |
| Reasoning spans rewritten | {sp} |
| Reasoning tokens before | {ob} |
| Reasoning tokens after | {ot} |
| **Tokens saved** | **{saved:.1}%** |
| PII redactions | {rd} |
{signal_line}

## Quality

- Mean composite score: {avg:.3}

## Error taxonomy

{err_lines}

## Formats

- `data.jsonl` — Axolotl `messages`
- `sharegpt.jsonl` — ShareGPT `conversations`
- `unsloth.jsonl` — Unsloth `messages`
- `prompts/` — system prompts used by each agent
"#,
        repo_line = if repo_id.is_empty() { String::new() } else { format!("Repo: `{repo_id}`\n") },
        ti = stats.traces_in, to = stats.traces_out, sp = stats.spans,
        ob = stats.original_tokens, ot = stats.processed_tokens, rd = stats.redactions,
    )
}

#[derive(Debug, Serialize)]
pub struct PerTraceRow {
    pub index: usize,
    pub spans: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub redactions: usize,
    pub signal: f64,
}

pub fn render_report(rows: &[PerTraceRow]) -> String {
    let mut out = String::from("# Processing report\n\n| trace | spans | tokens_before | tokens_after | saved % | signal |\n|-------|-------|---------------|--------------|---------|--------|\n");
    for row in rows {
        let saved = if row.tokens_before > 0 {
            100.0 * (1.0 - (row.tokens_after as f64 / row.tokens_before as f64))
        } else { 0.0 };
        let sig = if row.signal < 0.0 { "—".to_string() } else { format!("{:.3}", row.signal) };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {} |\n",
            row.index, row.spans, row.tokens_before, row.tokens_after, saved, sig
        ));
    }
    out
}
