"""Dataset card generator with reasoning-processing metrics."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List


def render(stats: Dict, scores: List[float], errors: Counter, repo_id: str = "") -> str:
    n = len(scores)
    avg = sum(scores) / n if n else 0.0
    err_lines = "\n".join(f"- `{k}`: {v}" for k, v in sorted(errors.items()))
    saved_pct = 0.0
    if stats["original_tokens"]:
        saved_pct = 100.0 * (1 - stats["compressed_tokens"] / stats["original_tokens"])
    signal = stats.get("avg_signal", 0.0) or 0.0
    signal_line = (
        f"| Mean QAQC signal | {signal:.3f} |"
        if signal > 0 else
        "| Mean QAQC signal | _not run_ |"
    )
    return f"""# Tokamak — Terse-Reasoning Trace Dataset

{f'Repo: `{repo_id}`' if repo_id else ''}

A reasoning-trace dataset whose internal `<thinking>` / `reasoning` channels
were processed — either compressed (terse rewrite, every step preserved) or
inverted (compressed skeleton expanded into a fuller trace). Final answers,
tool calls, code, and user inputs are byte-identical to the source traces.

## Pipeline

1. Ingest raw JSONL traces
2. Extract reasoning spans + surrounding problem / answer context
3. Process reasoning (compress, invert, or both) — full prompt + answer
   are fed as rails so the processor never invents steps that fit a
   different answer
4. QAQC: independent validator grades each span on 0..1; row gets a
   `signal` column (worst-judge rule when mirrored)
5. Anonymize (regex + entropy)
6. Quality score (6-dim, reported never filtered)
7. Classify error taxonomy (5-factor + none)
8. Deduplicate
9. Triple export — Axolotl + ShareGPT + Unsloth

## Processing results

| Metric | Value |
|--------|-------|
| Traces in | {stats['traces_in']} |
| Traces out (post-dedup) | {stats['traces_out']} |
| Reasoning spans rewritten | {stats['spans']} |
| Reasoning tokens before | {stats['original_tokens']:,} |
| Reasoning tokens after | {stats['compressed_tokens']:,} |
| **Tokens saved** | **{saved_pct:.1f}%** |
| PII redactions | {stats['redactions']} |
{signal_line}

Same reasoning steps. Fewer tokens (compress) or richer expansion (invert).

## Quality

- Mean composite score: {avg:.3f}
- Six-dimension score breakdown: reasoning depth, structure, tool calls,
  coherence, length, refusal — equally weighted at 1/6 each. Reported per
  trace, never used as a filter.

## Error taxonomy

{err_lines or '- none'}

## Formats

- `data.jsonl` — Axolotl `messages`
- `sharegpt.jsonl` — ShareGPT `conversations`
- `unsloth.jsonl` — Unsloth `messages`
- `prompts/compress_lite.md`, `compress_full.md`, `invert.md`, `validate.md`
  — system prompts used by each agent

## Provenance

Built with [Tokamak](https://github.com/thekozugroup/Tokamak) — terse-style
processing applied only to reasoning channels.
"""
