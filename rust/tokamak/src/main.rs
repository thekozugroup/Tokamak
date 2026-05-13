//! `tokamak` CLI — single-binary entry point.

use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgAction, Parser, ValueEnum};

use tokamak::anonymize::Level as AnonLevel;
use tokamak::pipeline::{run, RunOptions};
use tokamak::prompts::{self, Level as PromptLevel};

#[derive(Parser, Debug)]
#[command(
    name = "tokamak",
    version,
    about = "Terse-reasoning trace processor and curator.",
    long_about = "Process the reasoning channel of agent traces (compress or invert) \
                  with full PROBLEM + ANSWER context as rails, optional mirrored QAQC \
                  judging, and triple-format export (Axolotl / ShareGPT / Unsloth)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Parser, Debug)]
enum Cmd {
    /// Run the full processing + curation pipeline.
    Run(RunArgs),
    /// Print one of the bundled system prompts.
    Prompt {
        #[arg(value_enum)]
        which: WhichPrompt,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum WhichPrompt { Lite, Full, Invert, Validate }

#[derive(Parser, Debug)]
struct RunArgs {
    /// Directory of .jsonl files.
    #[arg(long)]
    input_dir: Option<PathBuf>,
    /// Single .jsonl file.
    #[arg(long)]
    input_file: Option<PathBuf>,
    /// Output directory.
    #[arg(long, default_value = "./curated")]
    output_dir: PathBuf,

    /// Processing mode.
    #[arg(long, default_value = "rules", value_parser = ["rules","compress","invert","both","noop"])]
    mode: String,
    /// Compression intensity (lite or full).
    #[arg(long, default_value = "lite", value_parser = ["lite","full"])]
    level: String,
    /// Concurrent agents per stage.
    #[arg(long, default_value_t = 8)]
    seqs: usize,

    /// Enable mirrored QAQC validation.
    #[arg(long, action = ArgAction::SetTrue)]
    qaqc: bool,
    /// Number of mirrored judges per span (worst-judge rule).
    #[arg(long, default_value_t = 1)]
    qaqc_mirrors: usize,

    /// OpenAI-compatible base URL (e.g. http://localhost:8000/v1).
    /// Defaults to env TOKAMAK_ENDPOINT / OPENAI_BASE_URL.
    #[arg(long)]
    endpoint: Option<String>,
    /// Model id. Defaults to env TOKAMAK_MODEL.
    #[arg(long)]
    model: Option<String>,

    #[arg(long, default_value = "strict", value_parser = ["standard","strict"])]
    anonymize_level: String,
    /// Jaccard threshold for dedup (1.0 disables).
    #[arg(long, default_value_t = 0.92)]
    max_similarity: f64,
}

fn init_tracing() {
    let fmt = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("TOKAMAK_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false);
    let _ = fmt.try_init();
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.cmd.unwrap_or(Cmd::Run(RunArgs {
        input_dir: None, input_file: None, output_dir: PathBuf::from("./curated"),
        mode: "rules".into(), level: "lite".into(), seqs: 8,
        qaqc: false, qaqc_mirrors: 1, endpoint: None, model: None,
        anonymize_level: "strict".into(), max_similarity: 0.92,
    })) {
        Cmd::Run(args) => cmd_run(args),
        Cmd::Prompt { which } => { cmd_prompt(which); Ok(()) }
    }
}

fn cmd_prompt(which: WhichPrompt) {
    let s = match which {
        WhichPrompt::Lite     => prompts::compress_prompt(PromptLevel::Lite),
        WhichPrompt::Full     => prompts::compress_prompt(PromptLevel::Full),
        WhichPrompt::Invert   => prompts::invert_prompt(),
        WhichPrompt::Validate => prompts::validate_prompt(),
    };
    println!("{s}");
}

fn cmd_run(a: RunArgs) -> Result<()> {
    if a.input_dir.is_none() && a.input_file.is_none() {
        anyhow::bail!("--input-dir or --input-file is required");
    }
    let opts = RunOptions {
        input_dir:       a.input_dir,
        input_file:      a.input_file,
        output_dir:      a.output_dir.clone(),
        mode:            a.mode,
        level:           PromptLevel::from_str(&a.level)?,
        anonymize_level: AnonLevel::from_str(&a.anonymize_level)?,
        max_similarity:  a.max_similarity,
        seqs:            a.seqs.max(1),
        qaqc:            a.qaqc,
        qaqc_mirrors:    a.qaqc_mirrors.max(1),
        endpoint:        a.endpoint,
        model:           a.model,
    };

    let stats = run(opts)?;
    let saved = if stats.original_tokens > 0 {
        100.0 * (1.0 - stats.compressed_tokens as f64 / stats.original_tokens as f64)
    } else { 0.0 };

    println!();
    println!("  traces in        : {}", stats.traces_in);
    println!("  traces out       : {}", stats.traces_out);
    println!("  reasoning spans  : {}", stats.spans);
    println!("  tokens before    : {}", stats.original_tokens);
    println!("  tokens after     : {}", stats.compressed_tokens);
    println!("  tokens saved     : {:.1}%", saved);
    println!("  PII redactions   : {}", stats.redactions);
    if stats.avg_signal > 0.0 {
        println!("  avg signal       : {:.3}", stats.avg_signal);
    }
    println!("  output dir       : {}", a.output_dir.canonicalize().unwrap_or(a.output_dir).display());

    // Also print machine-readable stats on the final line for the skill agent.
    println!("STATS_JSON={}", serde_json::to_string(&stats)?);
    Ok(())
}
