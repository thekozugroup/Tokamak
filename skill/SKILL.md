---
name: tokamak
description: Process the reasoning channel of agent traces — compress (terse rewrite) or invert (skeleton → fuller trace) — and emit Axolotl / ShareGPT / Unsloth training JSONL with an optional QAQC `signal` column. Use when the user has reasoning-trace JSONL and wants to build SFT data, save tokens on `<thinking>` blocks, or quality-grade processed traces.
---

# Tokamak — terse-reasoning trace processor

Single-binary Rust pipeline. Each row's reasoning is rewritten by a short-lived agent (compress / invert / both / rules / noop) with the surrounding **PROBLEM** and **ANSWER** fed as reference rails. An optional QAQC validator grades every span on 0..1 and the score lands in the row's `signal` column.

## When to invoke this skill

- User has `.jsonl` traces with `<thinking>` / `<reasoning>` blocks and wants to compress them.
- User wants to expand a compressed trace into a fuller skeleton using inversion (`Jackrong/Trace-Inverter-4B` style).
- User wants to QAQC-grade processed traces with mirrored judges.
- User wants Axolotl / ShareGPT / Unsloth SFT JSONL out the other side.

If the input doesn't have reasoning blocks, do not use this skill.

## Setup (one-time, idempotent)

Before the first run, ensure the binary is installed:

```bash
which tokamak >/dev/null 2>&1 || bash "${SKILL_DIR}/install.sh"
tokamak --version
```

`install.sh` detects the host platform, picks the matching binary out of `${SKILL_DIR}/bin/`, and symlinks it to `~/.local/bin/tokamak`. It also adds that directory to `PATH` for the current shell if needed. No Rust toolchain, no Python, no other deps required.

## Output location

`--output-dir` is **optional**. Tokamak writes everything under `./curated` *relative to the current working directory* — for skill invocations that means the agent's workspace folder. Do **not** ask the user for an output directory unless they explicitly want one elsewhere; the default Just Works.

If you need a different location, pass `--output-dir <path>` (absolute or relative). The final stdout line `STATS_JSON=...` is followed by an explicit `output dir : <absolute path>` line so you can confirm where files landed.

## Required user inputs

Before running, ask the user the following with `AskUserQuestion`. Skip any question the user has already answered earlier in the session — never re-ask.

| Question | Required when | Default |
|----------|---------------|---------|
| Input file or directory | always | _none_ |
| Output directory | optional | `./curated` (relative to your workspace) |
| Mode (rules / compress / invert / both / noop) | always | `rules` |
| Compression intensity (`lite` / `full`) | mode ∈ {compress, both} | `lite` |
| OpenAI-compatible endpoint URL | mode ∈ {compress, invert, both} OR qaqc enabled | env `TOKAMAK_ENDPOINT` |
| Model id | same as endpoint | env `TOKAMAK_MODEL` |
| Concurrent agents (`--seqs`) | mode is LLM-driven | `8` |
| QAQC on? | always (default off) | `false` |
| QAQC mirrors | qaqc enabled | `1` |

When the user picks `compress`, `invert`, or `both`, they MUST provide an OpenAI-compatible endpoint and model. Tokamak does not bundle an LLM — it expects an existing endpoint (vLLM, TGI, SGLang, Together, OpenAI, llama.cpp server, Ollama in OpenAI-compat mode, etc.).

Recommended endpoint phrasing:
> The endpoint must accept `POST /chat/completions` with the OpenAI schema. Examples: `http://localhost:8000/v1` for a local vLLM, `https://api.openai.com/v1`, `https://api.together.xyz/v1`. If you have a separate API key, set `TOKAMAK_API_KEY` in your environment.

## Recommended flow

1. **Probe the input.** Read the first line of the input file and confirm at least one of `<thinking>`, `<reasoning>`, `<thought>`, `<analyze>`, `<scratchpad>`, or an Anthropic-style `{type: "thinking"}` block is present. If absent, surface that to the user before running.
2. **Elicit settings.** Use `AskUserQuestion` for the table above. Group related questions (mode + level + seqs in one call, qaqc + mirrors in another).
3. **Dry-run rules first** unless the user explicitly asks for an LLM mode. Rules mode is free, fast, gives a 10–25 % token reduction, and surfaces any input-shape problems before LLM spend.
4. **Run.** Execute from the user's workspace directory so `./curated` lands where they expect:

   ```bash
   tokamak run \
       --input-file "$INPUT" \
       --mode "$MODE" \
       --level "$LEVEL" \
       --seqs "$SEQS" \
       --endpoint "$ENDPOINT" \
       --model "$MODEL" \
       ${OUT:+--output-dir "$OUT"} \
       ${QAQC:+--qaqc} \
       ${QAQC_MIRRORS:+--qaqc-mirrors "$QAQC_MIRRORS"}
   ```

   The final stdout line is `STATS_JSON={...}`. Parse it for the structured run summary. The line immediately above it (`output dir : ...`) gives the absolute path of the written artifacts.
5. **Show the user**: tokens saved, mean signal (if QAQC ran), output directory, and three exported files (`data.jsonl`, `sharegpt.jsonl`, `unsloth.jsonl`).

## Failure modes

- **No reasoning spans found** — surface a sample of the first record and ask the user whether they want to add custom tag names (not yet supported via CLI; flag for a feature request).
- **HTTP errors from the endpoint** — Tokamak retries twice with exponential backoff, then falls back to the original reasoning text for that span. You'll see `STATS_JSON.compressed_tokens` close to `STATS_JSON.original_tokens` and warnings in the logs.
- **`signal` is 0.0 across the board** — usually a validator parse error; the judge model is producing non-JSON output. Lower the temperature or use a stronger model for QAQC.

## Output

After a successful run:

```
$OUT/
  data.jsonl              # Axolotl
  sharegpt.jsonl          # ShareGPT
  unsloth.jsonl           # Unsloth
  dataset_card.md         # mean signal + per-axis stats
  compression_report.md   # per-trace table with signal column
  prompts/
    compress_lite.md
    compress_full.md
    invert.md
    validate.md
```

Each row in `data.jsonl` and `unsloth.jsonl` carries `metadata.signal` (or `signal` at top-level for Unsloth) when QAQC ran. Trainers can filter (`signal >= 0.7`), weight, or sample by it.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `TOKAMAK_ENDPOINT` | Default OpenAI-compatible base URL |
| `TOKAMAK_API_KEY` | Bearer token for the endpoint |
| `TOKAMAK_MODEL` | Default model id |
| `TOKAMAK_MAX_TOKENS` | Per-agent max output tokens (default 4096) |
| `TOKAMAK_TEMPERATURE` | Sampling temperature (default 0.0) |
| `TOKAMAK_TIMEOUT` | Per-request timeout seconds (default 180) |
| `TOKAMAK_RETRIES` | Retry attempts on transient errors (default 2) |
| `TOKAMAK_LOG` | Tracing log filter, e.g. `info`, `tokamak=debug` |

## Provenance

Source: <https://github.com/thekozugroup/Tokamak>. Inversion mode is adapted from [`Jackrong/Trace-Inverter-4B`](https://huggingface.co/Jackrong/Trace-Inverter-4B).
