//! Per-span processing agent. Compress / invert / rules / noop.

use crate::llm::LlmClient;
use crate::prompts::{self, Level};
use crate::terse::{compress_rules, estimate_tokens};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode { Rules, Compress, Invert, Noop }

impl Mode {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "rules"    => Ok(Self::Rules),
            "compress" => Ok(Self::Compress),
            "invert"   => Ok(Self::Invert),
            "noop"     => Ok(Self::Noop),
            other      => anyhow::bail!("unknown processor mode: {other:?}"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Rules => "rules", Self::Compress => "compress",
            Self::Invert => "invert", Self::Noop => "noop",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpanInput {
    pub problem: String,
    pub answer:  String,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub mode: Mode,
    pub original: String,
    pub processed: String,
    pub original_tokens: usize,
    pub processed_tokens: usize,
}

pub fn format_compress_user(s: &SpanInput) -> String {
    format!(
        "PROBLEM:\n{}\n\nANSWER:\n{}\n\nREASONING:\n{}",
        if s.problem.trim().is_empty() { "(no surrounding problem context provided)" } else { s.problem.trim() },
        if s.answer.trim().is_empty()  { "(no final answer provided)" }              else { s.answer.trim()  },
        s.reasoning,
    )
}

pub fn format_invert_user(s: &SpanInput) -> String {
    format!(
        "Problem:\n{}\n\nModel's final answer:\n{}\n\nReasoning Bubbles:\n{}\n\nReconstruct the full reasoning trace.",
        if s.problem.trim().is_empty() { "(no surrounding problem context provided)" } else { s.problem.trim() },
        if s.answer.trim().is_empty()  { "(no final answer provided)" }              else { s.answer.trim()  },
        s.reasoning,
    )
}

pub struct Processor { pub mode: Mode, pub level: Level }

impl Processor {
    pub fn new(mode: Mode, level: Level) -> Self { Self { mode, level } }

    /// Local (no-LLM) processing for rules/noop modes.
    pub fn process_local(&self, s: &SpanInput) -> ProcessResult {
        let text = &s.reasoning;
        if text.trim().is_empty() {
            return ProcessResult {
                mode: self.mode, original: text.clone(), processed: text.clone(),
                original_tokens: 0, processed_tokens: 0,
            };
        }
        match self.mode {
            Mode::Noop => ProcessResult {
                mode: self.mode, original: text.clone(), processed: text.clone(),
                original_tokens: estimate_tokens(text), processed_tokens: estimate_tokens(text),
            },
            Mode::Rules => {
                let r = compress_rules(text, self.level);
                ProcessResult {
                    mode: self.mode,
                    original: r.original, processed: r.compressed,
                    original_tokens: r.original_tokens, processed_tokens: r.compressed_tokens,
                }
            }
            _ => panic!("process_local called for LLM mode"),
        }
    }

    pub fn system(&self) -> &'static str {
        match self.mode {
            Mode::Compress => prompts::compress_prompt(self.level),
            Mode::Invert   => prompts::invert_prompt(),
            _              => "",
        }
    }

    pub fn format_user(&self, s: &SpanInput) -> String {
        match self.mode {
            Mode::Compress => format_compress_user(s),
            Mode::Invert   => format_invert_user(s),
            _ => s.reasoning.clone(),
        }
    }

    /// Process a batch of spans concurrently against an LLM endpoint.
    pub async fn process_batch_llm(
        &self,
        client: LlmClient,
        spans: &[SpanInput],
        max_workers: usize,
    ) -> Vec<ProcessResult> {
        let system = self.system().to_string();
        let items: Vec<(String, String)> = spans.iter()
            .map(|s| (system.clone(), self.format_user(s))).collect();
        let outs = client.batch_chat(items, max_workers).await;

        spans.iter().zip(outs).map(|(span, out)| {
            let text = &span.reasoning;
            let processed = match out {
                Some(s) if !s.trim().is_empty() => s,
                _ => text.clone(), // fall back to original on permanent failure
            };
            ProcessResult {
                mode: self.mode,
                original: text.clone(),
                original_tokens: estimate_tokens(text),
                processed_tokens: estimate_tokens(&processed),
                processed,
            }
        }).collect()
    }
}
