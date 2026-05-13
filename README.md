Tokamak processes the internal reasoning channel of agent traces — either compressing it into a terse rewrite (same logical steps, roughly half the tokens) or inverting a compressed skeleton into a fuller, more explicit trace — then runs a full curation pipeline that emits training-ready SFT data. Every span is graded 0..1 by an independent QAQC agent and the score is written to a `signal` column so downstream training can filter or weight by quality.

Reasoning traces dominate the cost of building "thinking" datasets. A single multi-step trajectory can run 4–10× the tokens of the final answer. Tokamak rewrites only that channel, leaves user turns, tool calls, code, and final answers byte-identical, and emits clean JSONL ready for full fine-tuning frameworks.

## Screenshots

![Tokamak — toroidal plasma confinement, the namesake metaphor for shaped reasoning](./docs/screenshot.png)

## How it works

A reasoning span is any `<thinking>`, `<reasoning>`, `<thought>`, `<analyze>`, or `<scratchpad>` block, plus OpenAI o1-style `reasoning` fields and Anthropic `{type: "thinking"}` content blocks. Tokamak walks every trace, locates each span, and processes it in place — leaving code fences, file paths, URLs, quoted strings, error messages, and tool-call JSON byte-for-byte intact.

Every span is processed with its full conversational context — the surrounding **problem** and the assistant's **final answer** are fed to the LLM as reference rails, never rewritten and never cited in the output. The processing agent only sees and only modifies the reasoning trace itself. The rails exist so the model never invents steps that fit a different answer, and so it preserves every transition the trace actually uses to reach the given conclusion.

### Two processing modes

- **Compress** — terse rewrite. Drop filler, pleasantries, hedging. Replace verbose phrasing with short synonyms. Preserve every reasoning step in order, every formula, every code byte. Typically 50–75% token reduction with no information loss.
- **Invert** — expand a compressed reasoning skeleton into a fuller, more explicit trace. The skeleton is treated as a structural outline: each bullet or fragment is expanded into coherent prose, no new methods or assumptions are introduced, the conclusion must match the given answer. This mode is inspired by [`Jackrong/Trace-Inverter-4B`](https://huggingface.co/Jackrong/Trace-Inverter-4B) and is the inverse operation of compression — useful for upsampling concise traces from a stronger teacher model into supervision data for a smaller student.

Modes can be run in isolation (`--mode compress` or `--mode invert`) or chained (`--mode both` runs compress → invert). A `--mode rules` flag applies a regex-only compression pass with no LLM calls, and `--mode noop` is a passthrough for baselines.

### Mirrored QAQC and the `signal` column

`--qaqc` enables an independent validation agent that grades each processed span against its original on three axes — logical fidelity (same steps, same order, same conclusion), content fidelity (every formula, code byte, error string, identifier preserved), and safety (no hallucinated methods or new assumptions). The overall `signal` score is the minimum of the three axes (worst-axis rule — one severe failure dominates).

`--qaqc-mirrors N` spins up N short-lived judge agents in parallel against the same span and keeps the worst score (worst-judge rule). Each mirror is scoped to a single row — there is no shared state between rows.

The resulting score is written to a `signal` column on every output row, so trainers can filter (`signal >= 0.7`), weight (`loss *= signal`), or sample (`p(row) ∝ signal`).

### Processing pipeline per row

```
row ──►  ┌─────────────────┐
         │ processing agent│  ──► compressed-or-inverted reasoning
         └─────────────────┘                                │
              ▲                                             ▼
     (problem + answer rails)         ┌─────────────────────────────┐
                                      │ mirrored validator agents   │
                                      │ (N short-lived judges)      │
                                      └─────────────┬───────────────┘
                                                    ▼
                                            signal ∈ [0, 1]
```

### Curation

After processing each trace flows through: regex + entropy-based PII anonymization, six-dimension quality scoring (reported, never used to filter), five-factor error classification, lexical deduplication that ignores boilerplate tool-call JSON, and simultaneous export to Axolotl, ShareGPT, and Unsloth formats. A dataset card is generated with processing statistics and the mean `signal`, and an optional `--push-to-hub` flag ships the result to the Hugging Face Hub.

## Usage

```bash
# Rules-only compression — free, no LLM, ~15% reduction
tokamak --input-file traces.jsonl --output-dir ./out --mode rules

# LLM-driven terse compression with 32 concurrent agents
tokamak --input-file traces.jsonl --output-dir ./out \
        --mode compress --level lite --seqs 32

# Invert compressed traces back into fuller reasoning
tokamak --input-file compressed.jsonl --output-dir ./out --mode invert --seqs 32

# Compress then invert, with 3 mirrored validators per span
tokamak --input-file traces.jsonl --output-dir ./out \
        --mode both --seqs 32 --qaqc --qaqc-mirrors 3
```

LLM backend resolution is environment-driven:

- `TOKAMAK_OPENAI_BASE_URL` → routes every agent (compress, invert, validate) at an OpenAI-compatible endpoint (vLLM, TGI, SGLang, Together, …).
- `ANTHROPIC_API_KEY` → Anthropic SDK.
- Otherwise the `claude` CLI is used.

## Optional Rust core

A companion Rust crate, [`rust/tokamak-engine`](./rust/tokamak-engine), provides a memory-safe `tokio` + `reqwest` orchestrator for OpenAI-compatible endpoints. When built and importable, the Python pipeline routes batch agent calls through it for semaphore-bounded concurrency without a GIL bottleneck. Build it with `maturin develop --release` inside `rust/tokamak-engine/`; without it, the pipeline falls back to a Python `ThreadPoolExecutor`.

## Stack

- Python 3.10+, zero required runtime dependencies for the rules-only path
- Optional: Anthropic SDK (LLM agents), `openai` (OpenAI-compatible endpoint), `scikit-learn` (dedup), `sentence-transformers` (semantic dedup), `huggingface_hub` (publish)
- Optional Rust: `tokamak-engine` PyO3 module for high-throughput concurrent orchestration
- Pytest regression suite

## Status

Active
