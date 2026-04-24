"""Dataset card generator with caveman compression metrics."""

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
    return f"""# Tokamak — Caveman-Compressed Reasoning Dataset

{f'Repo: `{repo_id}`' if repo_id else ''}

A reasoning-trace dataset whose internal `<thinking>` / `reasoning` channels
were rewritten in **caveman lite** style. Final answers, tool calls, code, and
user inputs are byte-identical to the source traces.

## Pipeline (one command, eight stages)

1. Ingest raw JSONL traces
2. Extract reasoning spans
3. Compress reasoning (caveman lite)
4. Anonymize (regex + entropy)
5. Quality score (6-dim, reported never filtered)
6. Classify error taxonomy (5-factor + none)
7. Deduplicate
8. Triple export — Axolotl + ShareGPT + Unsloth

## Compression results

| Metric | Value |
|--------|-------|
| Traces in | {stats['traces_in']} |
| Traces out (post-dedup) | {stats['traces_out']} |
| Reasoning spans rewritten | {stats['spans']} |
| Reasoning tokens before | {stats['original_tokens']:,} |
| Reasoning tokens after | {stats['compressed_tokens']:,} |
| **Tokens saved** | **{saved_pct:.1f}%** |
| PII redactions | {stats['redactions']} |

Same reasoning steps. Far less tokens. More steps fit per training example.

## Quality

- Mean composite score: {avg:.3f}
- Score breakdown follows TALOS Ornstein v2 (reasoning depth, structure,
  tool calls, coherence, length, refusal — equally weighted at 1/6 each).

## Error taxonomy

{err_lines or '- none'}

## Formats

- `data.jsonl` — Axolotl `messages`
- `sharegpt.jsonl` — ShareGPT `conversations`
- `unsloth.jsonl` — Unsloth `messages`
- `prompts/caveman_lite.md` — system prompt used during compression

## Provenance

- Compression spec: [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman)
- Curation pipeline: [DJLougen/TALOS-trace-curator](https://github.com/DJLougen/TALOS-trace-curator)
- Built with [tokamak](https://github.com/) — caveman + TALOS merged for
  reasoning-channel compression.
"""
