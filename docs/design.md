# Tokamak — Caveman Reasoning Curator

> Merger of [caveman](https://github.com/JuliusBrussee/caveman) (token-compressed
> language) and [TALOS-trace-curator](https://github.com/DJLougen/TALOS-trace-curator)
> (Hermes trace → fine-tune dataset pipeline). Captures caveman-style **internal
> reasoning** so each training step costs ~50–75% fewer tokens, letting the same
> token budget cover more reasoning steps.

## Problem

Reasoning traces dominate the cost of building "thinking" datasets. A single
multi-step trajectory can run 4–10× the tokens of the final answer. caveman
already proved you can compress LLM *output* ~75% with no accuracy loss. TALOS
already proved you can curate raw agent sessions into clean SFT/RL data. Nobody
has wired them together to attack the *reasoning channel specifically*.

## Goal

Produce a single tool that:

1. Reads raw agent traces (Hermes JSONL, Claude/OpenAI message JSONL).
2. Locates **reasoning content only** — `<thinking>`, `<reasoning>`, `analysis`
   blocks, assistant scratchpads — and rewrites them in **caveman lite**
   (concise, no filler, no hedging, full technical accuracy).
3. Leaves user turns, tool calls, tool results, code blocks, and final assistant
   answers byte-for-byte unchanged so trained behaviour stays faithful.
4. Runs the full TALOS pipeline (anonymize → quality score → error classify →
   dedup → triple export → dataset card → optional HF push) on the rewritten
   traces.
5. Reports per-trace and corpus-level **compression ratio** so the user can
   prove the dataset packs more reasoning per token.

## Non-Goals

- Not a runtime skill. Caveman ships as a runtime skill; Tokamak ships as a
  curation pipeline. The system prompts are exposed so users *can* generate new
  caveman-native data, but Tokamak itself does not run agents.
- Not a TALOS replacement. Quality scoring / error taxonomy / formatters are
  ported faithfully, not reinvented.
- Not opinionated about training stack. Output is Axolotl + ShareGPT + Unsloth,
  same as TALOS.

## Architecture

```
                        ┌──────────────────────┐
   raw .jsonl ─────────▶│  ingest              │
   (Hermes/Claude/OAI)  └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  reasoning extractor │  finds <thinking> /
                        │                      │  <reasoning> / scratchpad
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  caveman compressor  │  rules engine (default) or
                        │   (lite + concise)   │  Claude API rewrite
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  TALOS pipeline      │  anonymize → quality
                        │                      │  → classify → dedup
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  triple export       │  axolotl / sharegpt /
                        │  + dataset card      │  unsloth + card.md
                        └──────────┬───────────┘
                                   ▼
                              curated/ + HF push
```

## Components

| Module                 | Source           | Responsibility                                    |
|------------------------|------------------|---------------------------------------------------|
| `tokamak.prompts`      | new              | Caveman lite + concise system prompts             |
| `tokamak.extract`      | new              | Locate reasoning spans inside traces              |
| `tokamak.caveman`      | port of compress | Rules-based + LLM-based reasoning compressor      |
| `tokamak.anonymize`    | TALOS port       | Regex + entropy PII redaction                     |
| `tokamak.quality`      | TALOS port       | 6-dim quality score, reported never filtered      |
| `tokamak.classify`     | TALOS port       | 5-factor error taxonomy                           |
| `tokamak.dedup`        | TALOS port       | Lexical (TF-IDF) dedup with optional semantic     |
| `tokamak.export`       | TALOS port       | Axolotl / ShareGPT / Unsloth formatters           |
| `tokamak.card`         | TALOS port       | dataset_card.md with compression stats added      |
| `tokamak.cli`          | new              | Orchestrator + flags                              |

## Compression Modes

1. **`rules` (default, free, fast)** — apply caveman lite/full transforms with
   regex + a small replacement dictionary. Drops articles/filler/pleasantries,
   normalizes hedging, collapses common verbose patterns.
2. **`llm` (paid, accurate)** — call Claude (or `claude --print`) with the
   caveman lite system prompt to rewrite each reasoning block. Slower, better.
3. **`prompt-only`** — emit the system prompt for use during *future* trace
   generation. No rewriting; user generates born-compressed data.

## What Gets Compressed

A reasoning span = any one of:
- text inside `<thinking>`, `<reasoning>`, `<thought>`, `<analyze>`, `<scratchpad>`
- assistant messages whose `role == "assistant"` and which contain no tool
  calls and are immediately followed by another assistant message (i.e.
  internal thought turns, not user-facing answers)
- a `reasoning` field on a message (OpenAI o1-style)

Everything else is preserved verbatim. Tool-call JSON, code fences, error
strings, file paths, version numbers, and quoted inputs are **always**
preserved verbatim even inside reasoning spans.

## Data Flow Per Trace

```
raw_trace
  └─▶ extract_reasoning_spans(trace)        →  [(path, original_text), …]
        └─▶ for each span:
              if mode == rules:  caveman.rules.compress(text)
              if mode == llm:    caveman.llm.compress(text, level="lite")
        └─▶ patch trace in place, record (orig_tokens, new_tokens)
  └─▶ anonymize.redact(trace)
  └─▶ quality.score(trace)              →  composite + breakdown
  └─▶ classify.error_class(trace)       →  one of 5 + none
  └─▶ dedup pool
  └─▶ formatters → axolotl/sharegpt/unsloth records
```

## Output

```
out/
├── data.jsonl            # Axolotl messages format
├── sharegpt.jsonl        # ShareGPT conversations format
├── unsloth.jsonl         # Unsloth messages format
├── dataset_card.md       # stats + per-stage compression ratio
├── compression_report.md # per-trace before/after token counts
└── prompts/
    ├── caveman_lite.md   # the system prompt used
    └── caveman_full.md
```

## Testing

- `tests/test_extract.py` — span detection across Hermes / Claude / OAI shapes
- `tests/test_caveman_rules.py` — rule transforms preserve code/paths/errors
- `tests/test_pipeline.py` — end-to-end on the bundled `examples/sample_trace.jsonl`

## Deployment

User runs:

```
tokamak --input-dir ./raw --output-dir ./curated --compress-mode rules
tokamak --input-file my.jsonl --compress-mode llm --push-to-hub --repo-id me/cave-reason
```

Then trains in Axolotl / Unsloth with the resulting `data.jsonl`.
