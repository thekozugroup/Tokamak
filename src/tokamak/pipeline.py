"""End-to-end orchestration. CLI is a thin wrapper over `run()`.

For each row (trace) the pipeline runs:

  1. Find reasoning spans + surrounding problem/answer context.
  2. Process each span: compress (terse rewrite), invert (skeleton ->
     fuller trace), or both (compress then invert). Modes can also be
     `rules` (regex-only) or `noop` (passthrough).
  3. QAQC: an independent validation agent grades the processed span on
     0..1. Optional `mirror_qaqc > 1` runs N judges in parallel and keeps
     the worst score (one bad reading dominates).
  4. Apply processed text back to the trace.
  5. Anonymize, dedup, score, classify, export.

The per-trace `signal` (average span signal) is written into the row's
`metadata.signal` and into a dedicated `signal` column in the
compression report.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import anonymize, card, classify, dedup, export, extract, prompts
from .caveman import estimate_tokens
from .processor import Processor, ProcessResult, SpanInput
from .validator import ValidationResult, Validator

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
# Stats
# ---------------------------------------------------------------------------

@dataclass
class TraceStats:
    spans: int = 0
    original_tokens: int = 0
    processed_tokens: int = 0
    signal_sum: float = 0.0
    signal_count: int = 0

    @property
    def signal(self) -> float:
        if self.signal_count == 0:
            return 0.0
        return self.signal_sum / self.signal_count


@dataclass
class RunStats:
    traces_in: int = 0
    traces_out: int = 0
    spans: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0   # name kept for backward-compat with card.render
    redactions: int = 0
    avg_signal: float = 0.0
    per_trace: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "traces_in": self.traces_in,
            "traces_out": self.traces_out,
            "spans": self.spans,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "redactions": self.redactions,
            "avg_signal": self.avg_signal,
        }


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _collect_spans(
    traces: List[Dict[str, Any]]
) -> Tuple[List[extract.Span], List[int]]:
    """Flatten spans across all traces. Returns (spans, trace_index_per_span)."""
    flat: List[extract.Span] = []
    owner: List[int] = []
    for ti, trace in enumerate(traces):
        for sp in extract.find_reasoning_spans(trace):
            flat.append(sp)
            owner.append(ti)
    return flat, owner


def _to_inputs(spans: List[extract.Span]) -> List[SpanInput]:
    return [
        SpanInput(problem=sp.problem, answer=sp.answer, reasoning=sp.text)
        for sp in spans
    ]


def _run_stage(
    spans: List[extract.Span],
    processor: Processor,
    seqs: int,
) -> List[ProcessResult]:
    inputs = _to_inputs(spans)
    return processor.process_batch(inputs, max_workers=max(1, seqs))


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _trace_id(trace: Dict[str, Any], index: int) -> str:
    raw = json.dumps(trace, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"t{index:06d}_{hashlib.sha1(raw).hexdigest()[:10]}"


def _apply_signal(trace: Dict[str, Any], signal: float) -> None:
    """Stamp `signal` into a stable place on the row for downstream tooling."""
    md = trace.setdefault("metadata", {})
    if not isinstance(md, dict):
        # Trace already used `metadata` for something else — keep both.
        md = {}
        trace["metadata"] = md
    md["signal"] = round(signal, 4)
    trace["signal"] = round(signal, 4)


def run(
    *,
    input_dir: Optional[Path],
    input_file: Optional[Path],
    output_dir: Path,
    compress_mode: str = "rules",
    caveman_level: str = "lite",       # back-compat: maps to processor `level`
    anonymize_level: str = "strict",
    max_similarity: float = 0.92,
    model: Optional[str] = None,
    llm_concurrency: int = 1,          # back-compat alias for `seqs`
    seqs: Optional[int] = None,
    process_mode: Optional[str] = None,  # "compress" | "invert" | "both" | "rules" | "noop"
    level: Optional[str] = None,
    qaqc: bool = False,
    qaqc_mirrors: int = 1,
) -> RunStats:
    """Run the full pipeline. See module docstring for stage order.

    Backward-compat: callers using `compress_mode` / `caveman_level` /
    `llm_concurrency` still work — they map onto the new processor mode,
    level, and concurrency parameters.
    """
    traces = load_traces(input_dir, input_file)
    if not traces:
        raise SystemExit("no traces loaded")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts").mkdir(exist_ok=True)
    (output_dir / "prompts" / "compress_lite.md").write_text(prompts.compress_prompt("lite"))
    (output_dir / "prompts" / "compress_full.md").write_text(prompts.compress_prompt("full"))
    (output_dir / "prompts" / "invert.md").write_text(prompts.invert_prompt())
    (output_dir / "prompts" / "validate.md").write_text(prompts.validate_prompt())
    # Back-compat: keep the old filenames around so existing consumers don't break.
    (output_dir / "prompts" / "caveman_lite.md").write_text(prompts.compress_prompt("lite"))
    (output_dir / "prompts" / "caveman_full.md").write_text(prompts.compress_prompt("full"))

    # Resolve effective config.
    eff_mode = (process_mode or compress_mode or "rules").lower()
    if eff_mode == "llm":
        eff_mode = "compress"          # back-compat alias
    eff_level = (level or caveman_level or "lite").lower()
    eff_seqs = seqs if seqs is not None else max(1, llm_concurrency)

    if eff_mode not in ("rules", "compress", "invert", "both", "noop"):
        raise ValueError(f"unknown process_mode: {eff_mode!r}")

    # Build processors and (optional) validator.
    stages: List[Processor]
    if eff_mode == "both":
        stages = [
            Processor("compress", level=eff_level, model=model),
            Processor("invert", model=model),
        ]
    else:
        stages = [Processor(eff_mode, level=eff_level, model=model)]

    validator = Validator(model=model, mirrors=qaqc_mirrors) if qaqc else None

    # Flatten spans.
    flat_spans, owner = _collect_spans(traces)

    stats = RunStats(traces_in=len(traces))
    per_trace_stats: List[TraceStats] = [TraceStats() for _ in traces]

    # Capture originals before any stage rewrites them — needed for QAQC.
    originals: List[SpanInput] = _to_inputs(flat_spans)

    # Sequential stages: compress → (optional) invert. Each stage rebatches.
    current_results: Optional[List[ProcessResult]] = None
    for processor in stages:
        # For the *next* stage, feed the previous stage's output as the
        # reasoning input — same problem/answer rails, new reasoning text.
        if current_results is not None:
            for sp, pr in zip(flat_spans, current_results):
                sp.text = pr.processed
        current_results = _run_stage(flat_spans, processor, eff_seqs)

    final_results = current_results or []

    # QAQC: validate each final-processed span vs its captured original.
    signals: List[Optional[ValidationResult]] = [None] * len(flat_spans)
    if validator is not None and final_results:
        from concurrent.futures import ThreadPoolExecutor

        def _grade(i: int) -> Tuple[int, ValidationResult]:
            orig_input = originals[i]
            mode_for_judge = "invert" if stages[-1].mode == "invert" else (
                "compress" if stages[-1].mode in ("compress", "rules") else "compress"
            )
            return i, validator.validate(
                orig_input, final_results[i].processed, mode_for_judge
            )

        if eff_seqs > 1:
            with ThreadPoolExecutor(max_workers=eff_seqs) as ex:
                for i, vr in ex.map(_grade, range(len(flat_spans))):
                    signals[i] = vr
        else:
            for i in range(len(flat_spans)):
                _, vr = _grade(i)
                signals[i] = vr

    # Apply final processed text back to the trace and roll up per-trace stats.
    for i, (sp, pr) in enumerate(zip(flat_spans, final_results)):
        sp.apply(pr.processed)
        ti = owner[i]
        ts = per_trace_stats[ti]
        ts.spans += 1
        ts.original_tokens += estimate_tokens(originals[i].reasoning)
        ts.processed_tokens += pr.processed_tokens
        if signals[i] is not None:
            ts.signal_sum += signals[i].signal  # type: ignore[union-attr]
            ts.signal_count += 1

    # Anonymize and stamp signal onto each row.
    processed_traces: List[Dict[str, Any]] = []
    signal_values: List[float] = []
    for ti, trace in enumerate(traces):
        ts = per_trace_stats[ti]
        stats.spans += ts.spans
        stats.original_tokens += ts.original_tokens
        stats.compressed_tokens += ts.processed_tokens

        trace, n_redactions = anonymize.redact_trace(trace, level=anonymize_level)
        stats.redactions += n_redactions

        # Default signal = 1.0 if QAQC disabled (we don't have a reading;
        # mark as unknown via -1.0 so consumers can distinguish).
        if validator is None:
            row_signal = -1.0
        else:
            row_signal = ts.signal if ts.signal_count else -1.0
        _apply_signal(trace, row_signal)
        if row_signal >= 0:
            signal_values.append(row_signal)

        stats.per_trace.append({
            "index": ti,
            "spans": ts.spans,
            "tokens_before": ts.original_tokens,
            "tokens_after": ts.processed_tokens,
            "redactions": n_redactions,
            "signal": row_signal,
        })
        processed_traces.append(trace)

    stats.avg_signal = (
        sum(signal_values) / len(signal_values) if signal_values else 0.0
    )

    # Dedup.
    deduped, dropped = dedup.dedup(processed_traces, max_similarity=max_similarity)
    logger.info("dedup dropped %d / %d", dropped, len(processed_traces))

    # Score + classify + format.
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

        axolotl_lines.append(json.dumps(
            export.axolotl(trace, composite, tid, err), ensure_ascii=False
        ))
        sharegpt_lines.append(json.dumps(export.sharegpt(trace), ensure_ascii=False))
        unsloth_lines.append(json.dumps(
            export.unsloth(trace, composite, tid, err), ensure_ascii=False
        ))

    (output_dir / "data.jsonl").write_text(
        "\n".join(axolotl_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "sharegpt.jsonl").write_text(
        "\n".join(sharegpt_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "unsloth.jsonl").write_text(
        "\n".join(unsloth_lines) + "\n", encoding="utf-8"
    )

    stats.traces_out = len(deduped)

    (output_dir / "dataset_card.md").write_text(
        card.render(stats.as_dict(), score_values, error_counts), encoding="utf-8"
    )

    # Per-trace processing report. `signal` column shown alongside compression.
    report = [
        "# Processing report",
        "",
        "| trace | spans | tokens_before | tokens_after | saved % | signal |",
        "|-------|-------|---------------|--------------|---------|--------|",
    ]
    for row in stats.per_trace:
        before = row["tokens_before"] or 0
        after = row["tokens_after"] or 0
        saved = 0.0 if before == 0 else 100.0 * (1 - after / before)
        sig = row["signal"]
        sig_cell = "—" if sig < 0 else f"{sig:.3f}"
        report.append(
            f"| {row['index']} | {row['spans']} | {before} | {after} | "
            f"{saved:.1f} | {sig_cell} |"
        )
    (output_dir / "compression_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

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
