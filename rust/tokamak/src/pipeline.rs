//! Top-level orchestrator. Loads JSONL, finds spans, runs processor stages,
//! optionally validates, anonymizes, scores, dedups, exports.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::anonymize::{self, Level as AnonLevel};
use crate::card::{render_card, render_report, CardStats, PerTraceRow};
use crate::classify;
use crate::dedup;
use crate::export;
use crate::extract::{self, Span};
use crate::llm::{LlmClient, LlmConfig};
use crate::processor::{Mode, Processor, ProcessResult, SpanInput};
use crate::prompts::{self, Level};
use crate::quality;
use crate::terse::estimate_tokens;
use crate::validator::Validator;

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub input_dir:       Option<PathBuf>,
    pub input_file:      Option<PathBuf>,
    pub output_dir:      PathBuf,
    pub mode:            String,         // "rules" | "compress" | "invert" | "both" | "noop"
    pub level:           Level,
    pub anonymize_level: AnonLevel,
    pub max_similarity:  f64,
    pub seqs:            usize,
    pub qaqc:            bool,
    pub qaqc_mirrors:    usize,
    pub endpoint:        Option<String>,
    pub model:           Option<String>,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct RunStats {
    pub traces_in:        usize,
    pub traces_out:       usize,
    pub spans:            usize,
    pub original_tokens:  usize,
    pub compressed_tokens:usize,
    pub redactions:       usize,
    pub avg_signal:       f64,
}

pub fn load_traces(opts: &RunOptions) -> Result<Vec<Value>> {
    let mut traces = Vec::new();
    if let Some(p) = &opts.input_file { traces.extend(load_jsonl(p)?); }
    if let Some(d) = &opts.input_dir  {
        for entry in walk_jsonl(d)? { traces.extend(load_jsonl(&entry)?); }
    }
    Ok(traces)
}

fn load_jsonl(path: &Path) -> Result<Vec<Value>> {
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        match serde_json::from_str::<Value>(&line) {
            Ok(v) => out.push(v),
            Err(e) => tracing::warn!("{}:{} bad JSON: {}", path.display(), i+1, e),
        }
    }
    Ok(out)
}

fn walk_jsonl(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() { out.extend(walk_jsonl(&path)?); }
        else if path.extension().and_then(|x| x.to_str()) == Some("jsonl") { out.push(path); }
    }
    out.sort();
    Ok(out)
}

fn stages_for(mode: &str) -> Result<Vec<Mode>> {
    match mode {
        "rules"    => Ok(vec![Mode::Rules]),
        "compress" => Ok(vec![Mode::Compress]),
        "invert"   => Ok(vec![Mode::Invert]),
        "both"     => Ok(vec![Mode::Compress, Mode::Invert]),
        "noop"     => Ok(vec![Mode::Noop]),
        other      => anyhow::bail!("unknown mode: {other:?}"),
    }
}

#[tokio::main]
pub async fn run(opts: RunOptions) -> Result<RunStats> {
    let traces = load_traces(&opts)?;
    if traces.is_empty() { anyhow::bail!("no traces loaded"); }

    fs::create_dir_all(&opts.output_dir)?;
    fs::create_dir_all(opts.output_dir.join("prompts"))?;
    write_prompts(&opts.output_dir)?;

    let stages = stages_for(&opts.mode)?;
    let needs_llm = stages.iter().any(|m| matches!(m, Mode::Compress | Mode::Invert)) || opts.qaqc;

    // Build client once, shared across stages.
    let llm = if needs_llm {
        Some(LlmClient::new(LlmConfig::from_env_or(opts.endpoint.clone(), opts.model.clone()))
            .context("build llm client")?)
    } else { None };

    // Mutable trace store + flat span list.
    let mut traces = traces;
    let (flat_spans, owner) = collect_spans(&traces);

    // Capture originals before any rewrite (needed for QAQC).
    let originals: Vec<SpanInput> = flat_spans.iter()
        .map(|s| SpanInput { problem: s.problem.clone(), answer: s.answer.clone(), reasoning: s.text.clone() })
        .collect();

    // Run each stage in order. Each stage produces new reasoning text for
    // every span; the next stage feeds it as input.
    let mut current_reasoning: Vec<String> = flat_spans.iter().map(|s| s.text.clone()).collect();
    let mut final_results: Vec<ProcessResult> = Vec::new();
    let mut final_mode = Mode::Noop;

    for stage_mode in stages.iter().copied() {
        let proc = Processor::new(stage_mode, opts.level);
        let inputs: Vec<SpanInput> = flat_spans.iter().enumerate().map(|(i, s)| SpanInput {
            problem: s.problem.clone(), answer: s.answer.clone(),
            reasoning: current_reasoning[i].clone(),
        }).collect();

        let results = match stage_mode {
            Mode::Rules | Mode::Noop => inputs.iter().map(|s| proc.process_local(s)).collect(),
            Mode::Compress | Mode::Invert => {
                proc.process_batch_llm(llm.clone().expect("llm client"), &inputs, opts.seqs.max(1)).await
            }
        };

        for (i, r) in results.iter().enumerate() {
            current_reasoning[i] = r.processed.clone();
        }
        final_results = results;
        final_mode = stage_mode;
    }

    // QAQC pass.
    let mut signals: Vec<Option<f64>> = vec![None; flat_spans.len()];
    let mut avg_signal_sum = 0.0;
    let mut avg_signal_count = 0usize;
    if opts.qaqc && !flat_spans.is_empty() {
        let v = Validator::new(opts.qaqc_mirrors);
        let mode_name: &'static str = match final_mode {
            Mode::Invert => "invert",
            _            => "compress",
        };
        let items: Vec<(SpanInput, String, &'static str)> = originals.iter().enumerate()
            .map(|(i, orig)| (orig.clone(), final_results[i].processed.clone(), mode_name))
            .collect();
        let grades = v.validate_batch(llm.clone().expect("llm client"), items, opts.seqs.max(1)).await;
        for (i, g) in grades.into_iter().enumerate() {
            signals[i] = Some(g.signal);
            avg_signal_sum += g.signal;
            avg_signal_count += 1;
        }
    }

    // Apply final processed text back into the trace JSON.
    for (i, span) in flat_spans.iter().enumerate() {
        extract::apply_replacement(&mut traces[owner[i]], span, &final_results[i].processed);
    }

    // Per-trace rollup.
    let mut per_trace_rows: Vec<PerTraceRow> = (0..traces.len()).map(|i| PerTraceRow {
        index: i, spans: 0, tokens_before: 0, tokens_after: 0, redactions: 0, signal: -1.0,
    }).collect();
    let mut per_trace_signal_sum: Vec<f64> = vec![0.0; traces.len()];
    let mut per_trace_signal_n: Vec<usize> = vec![0; traces.len()];
    for (i, _span) in flat_spans.iter().enumerate() {
        let ti = owner[i];
        per_trace_rows[ti].spans += 1;
        per_trace_rows[ti].tokens_before += estimate_tokens(&originals[i].reasoning);
        per_trace_rows[ti].tokens_after  += final_results[i].processed_tokens;
        if let Some(s) = signals[i] {
            per_trace_signal_sum[ti] += s;
            per_trace_signal_n[ti] += 1;
        }
    }

    // Anonymize each trace, stamp signal, accumulate redactions.
    let mut redactions = 0usize;
    for (ti, trace) in traces.iter_mut().enumerate() {
        let n = anonymize::redact_trace(trace, opts.anonymize_level);
        redactions += n;
        per_trace_rows[ti].redactions = n;
        let signal = if per_trace_signal_n[ti] > 0 {
            per_trace_signal_sum[ti] / per_trace_signal_n[ti] as f64
        } else { -1.0 };
        per_trace_rows[ti].signal = signal;
        stamp_signal(trace, signal);
    }

    // Dedup.
    let (kept, dropped) = dedup::dedup(traces, opts.max_similarity);
    tracing::info!("dedup dropped {} / {}", dropped, kept.len() + dropped);

    // Score + classify + export.
    let mut score_values: Vec<f64> = Vec::with_capacity(kept.len());
    let mut error_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let axolotl_path  = opts.output_dir.join("data.jsonl");
    let sharegpt_path = opts.output_dir.join("sharegpt.jsonl");
    let unsloth_path  = opts.output_dir.join("unsloth.jsonl");
    let mut axo_f = fs::File::create(&axolotl_path)?;
    let mut shg_f = fs::File::create(&sharegpt_path)?;
    let mut uns_f = fs::File::create(&unsloth_path)?;

    for (i, trace) in kept.iter().enumerate() {
        let sc = quality::score(trace);
        let err = classify::classify(trace);
        let tid = trace_id(trace, i);
        score_values.push(sc.composite);
        *error_counts.entry(err).or_insert(0) += 1;

        writeln!(axo_f, "{}", export::axolotl(trace, sc.composite, &tid, err))?;
        writeln!(shg_f, "{}", export::sharegpt(trace))?;
        writeln!(uns_f, "{}", export::unsloth(trace, sc.composite, &tid, err))?;
    }

    let stats = RunStats {
        traces_in:        per_trace_rows.len(),
        traces_out:       kept.len(),
        spans:            flat_spans.len(),
        original_tokens:  per_trace_rows.iter().map(|r| r.tokens_before).sum(),
        compressed_tokens:per_trace_rows.iter().map(|r| r.tokens_after).sum(),
        redactions,
        avg_signal: if avg_signal_count > 0 { avg_signal_sum / avg_signal_count as f64 } else { 0.0 },
    };

    let card_stats = CardStats {
        traces_in: stats.traces_in, traces_out: stats.traces_out, spans: stats.spans,
        original_tokens: stats.original_tokens, processed_tokens: stats.compressed_tokens,
        redactions: stats.redactions, avg_signal: stats.avg_signal,
    };
    fs::write(opts.output_dir.join("dataset_card.md"),
              render_card(&card_stats, &score_values, &error_counts, ""))?;
    fs::write(opts.output_dir.join("compression_report.md"),
              render_report(&per_trace_rows))?;

    Ok(stats)
}

fn stamp_signal(trace: &mut Value, signal: f64) {
    if let Value::Object(o) = trace {
        let md = o.entry("metadata".to_string()).or_insert_with(|| json!({}));
        if let Value::Object(mdo) = md {
            mdo.insert("signal".into(), json!((signal * 10000.0).round() / 10000.0));
        }
        o.insert("signal".into(), json!((signal * 10000.0).round() / 10000.0));
    }
}

fn collect_spans(traces: &[Value]) -> (Vec<Span>, Vec<usize>) {
    let mut flat = Vec::new();
    let mut owner = Vec::new();
    for (ti, t) in traces.iter().enumerate() {
        for s in extract::find_reasoning_spans(t) {
            flat.push(s);
            owner.push(ti);
        }
    }
    (flat, owner)
}

fn write_prompts(dir: &Path) -> Result<()> {
    fs::write(dir.join("prompts/compress_lite.md"), prompts::compress_prompt(Level::Lite))?;
    fs::write(dir.join("prompts/compress_full.md"), prompts::compress_prompt(Level::Full))?;
    fs::write(dir.join("prompts/invert.md"),        prompts::invert_prompt())?;
    fs::write(dir.join("prompts/validate.md"),      prompts::validate_prompt())?;
    Ok(())
}

fn trace_id(trace: &Value, idx: usize) -> String {
    let raw = serde_json::to_string(trace).unwrap_or_default();
    let mut hasher = sha1_smol::Sha1::new();
    hasher.update(raw.as_bytes());
    let hex = hasher.digest().to_string();
    format!("t{:06}_{}", idx, &hex[..10])
}
