"""End-to-end orchestration. CLI is a thin wrapper over `run()`."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import anonymize, card, caveman, classify, dedup, export, extract, prompts

logger = logging.getLogger("tokamak")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d bad JSON: %s", path.name, i, exc)
    return out


def load_traces(input_dir: Optional[Path], input_file: Optional[Path]) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    if input_file:
        traces.extend(_load_jsonl(input_file))
    if input_dir:
        for p in sorted(input_dir.rglob("*.jsonl")):
            traces.extend(_load_jsonl(p))
    return traces


# ---------------------------------------------------------------------------
# Compression of one trace
# ---------------------------------------------------------------------------

@dataclass
class TraceStats:
    spans: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0


@dataclass
class RunStats:
    traces_in: int = 0
    traces_out: int = 0
    spans: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    redactions: int = 0
    per_trace: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "traces_in": self.traces_in,
            "traces_out": self.traces_out,
            "spans": self.spans,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "redactions": self.redactions,
        }


def compress_trace(trace: Dict[str, Any], compressor) -> TraceStats:
    spans = extract.find_reasoning_spans(trace)
    s = TraceStats()
    for span in spans:
        result = compressor(span.text)
        span.apply(result.compressed)
        s.spans += 1
        s.original_tokens += result.original_tokens
        s.compressed_tokens += result.compressed_tokens
    return s


def compress_traces_concurrent(
    traces: List[Dict[str, Any]], compressor, max_workers: int
) -> List[TraceStats]:
    """Compress reasoning spans across many traces concurrently.

    Collects every span from every trace, submits them as a single batch to
    `compressor.compress_batch`, then applies results back in order. This lets
    a self-hosted LLM endpoint (vLLM etc.) saturate its scheduler across a
    whole dataset instead of one trace at a time.

    Requires `compressor` to expose `compress_batch(texts, max_workers)`.
    Falls back to sequential compression for traces with zero spans.
    """
    per_trace_spans: List[List[extract.Span]] = [
        extract.find_reasoning_spans(t) for t in traces
    ]

    # Flatten: list of (trace_index, span) pairs.
    flat: List[tuple] = []
    for ti, spans in enumerate(per_trace_spans):
        for sp in spans:
            flat.append((ti, sp))

    if not flat:
        return [TraceStats() for _ in traces]

    texts = [sp.text for _, sp in flat]
    results = compressor.compress_batch(texts, max_workers=max_workers)

    # Apply results back; collect per-trace stats.
    stats = [TraceStats() for _ in traces]
    for (ti, sp), result in zip(flat, results):
        sp.apply(result.compressed)
        stats[ti].spans += 1
        stats[ti].original_tokens += result.original_tokens
        stats[ti].compressed_tokens += result.compressed_tokens
    return stats


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _trace_id(trace: Dict[str, Any], index: int) -> str:
    raw = json.dumps(trace, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"t{index:06d}_{hashlib.sha1(raw).hexdigest()[:10]}"


def run(
    *,
    input_dir: Optional[Path],
    input_file: Optional[Path],
    output_dir: Path,
    compress_mode: str = "rules",
    caveman_level: str = "lite",
    anonymize_level: str = "strict",
    max_similarity: float = 0.92,
    model: Optional[str] = None,
    llm_concurrency: int = 1,
) -> RunStats:
    traces = load_traces(input_dir, input_file)
    if not traces:
        raise SystemExit("no traces loaded")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts").mkdir(exist_ok=True)
    (output_dir / "prompts" / "caveman_lite.md").write_text(prompts.system_prompt("lite"))
    (output_dir / "prompts" / "caveman_full.md").write_text(prompts.system_prompt("full"))

    compressor = caveman.get_compressor(compress_mode, level=caveman_level, model=model)

    stats = RunStats(traces_in=len(traces))
    processed: List[Dict[str, Any]] = []

    # Concurrent path: only meaningful when the compressor exposes compress_batch
    # (currently LLMCompressor). Rules / noop are cheap enough that the overhead
    # of cross-trace batching doesn't pay off.
    use_batch = (
        llm_concurrency > 1
        and compress_mode == "llm"
        and hasattr(compressor, "compress_batch")
    )

    if use_batch:
        trace_stats = compress_traces_concurrent(
            traces, compressor, max_workers=llm_concurrency
        )
        for i, (trace, ts) in enumerate(zip(traces, trace_stats)):
            stats.spans += ts.spans
            stats.original_tokens += ts.original_tokens
            stats.compressed_tokens += ts.compressed_tokens
            trace, n_redactions = anonymize.redact_trace(trace, level=anonymize_level)
            stats.redactions += n_redactions
            stats.per_trace.append({
                "index": i,
                "spans": ts.spans,
                "tokens_before": ts.original_tokens,
                "tokens_after": ts.compressed_tokens,
                "redactions": n_redactions,
            })
            processed.append(trace)
    else:
        for i, trace in enumerate(traces):
            # 1. Compress reasoning spans (mutates trace in place).
            ts = compress_trace(trace, compressor)
            stats.spans += ts.spans
            stats.original_tokens += ts.original_tokens
            stats.compressed_tokens += ts.compressed_tokens

            # 2. Anonymize.
            trace, n_redactions = anonymize.redact_trace(trace, level=anonymize_level)
            stats.redactions += n_redactions

            stats.per_trace.append({
                "index": i,
                "spans": ts.spans,
                "tokens_before": ts.original_tokens,
                "tokens_after": ts.compressed_tokens,
                "redactions": n_redactions,
            })
            processed.append(trace)

    # 3. Dedup.
    deduped, dropped = dedup.dedup(processed, max_similarity=max_similarity)
    logger.info("dedup dropped %d / %d", dropped, len(processed))

    # 4. Score + classify + format.
    from .quality import score as quality_score
    axolotl_lines: List[str] = []
    sharegpt_lines: List[str] = []
    unsloth_lines: List[str] = []
    score_values: List[float] = []
    error_counts: Counter = Counter()

    for i, trace in enumerate(deduped):
        composite, _breakdown = quality_score(trace)
        err = classify.classify(trace)
        tid = _trace_id(trace, i)

        score_values.append(composite)
        error_counts[err] += 1

        axolotl_lines.append(json.dumps(export.axolotl(trace, composite, tid, err), ensure_ascii=False))
        sharegpt_lines.append(json.dumps(export.sharegpt(trace), ensure_ascii=False))
        unsloth_lines.append(json.dumps(export.unsloth(trace, composite, tid, err), ensure_ascii=False))

    (output_dir / "data.jsonl").write_text("\n".join(axolotl_lines) + "\n", encoding="utf-8")
    (output_dir / "sharegpt.jsonl").write_text("\n".join(sharegpt_lines) + "\n", encoding="utf-8")
    (output_dir / "unsloth.jsonl").write_text("\n".join(unsloth_lines) + "\n", encoding="utf-8")

    stats.traces_out = len(deduped)

    (output_dir / "dataset_card.md").write_text(
        card.render(stats.as_dict(), score_values, error_counts), encoding="utf-8"
    )

    # Per-trace compression report.
    report = ["# Compression report", "", "| trace | spans | tokens_before | tokens_after | saved % |",
              "|-------|-------|---------------|--------------|---------|"]
    for row in stats.per_trace:
        before = row["tokens_before"] or 0
        after = row["tokens_after"] or 0
        saved = 0.0 if before == 0 else 100.0 * (1 - after / before)
        report.append(f"| {row['index']} | {row['spans']} | {before} | {after} | {saved:.1f} |")
    (output_dir / "compression_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return stats


def push_to_hub(out_dir: Path, repo_id: str, private: bool = True) -> None:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise SystemExit("install `huggingface_hub` to use --push-to-hub") from exc
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset")
    logger.info("pushed to https://huggingface.co/datasets/%s", repo_id)
