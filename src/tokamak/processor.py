"""Per-span processing agents.

A Processor takes a reasoning span plus its surrounding context (problem +
answer) and rewrites the reasoning. Two modes are supported:

- compress  — terse rewrite, every step preserved, ~50% fewer tokens
- invert    — expand a compressed skeleton into a fuller trace, no new
              methods or assumptions, conclusion must match the given answer

Modes can be chained (compress → invert) via the pipeline; this module is
single-step.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

from . import prompts
from .caveman import (
    CompressResult,
    compress_rules,
    estimate_tokens,
)
from .llm_client import LLMClient

logger = logging.getLogger("tokamak.processor")


# ---------------------------------------------------------------------------
# Input/output container
# ---------------------------------------------------------------------------

@dataclass
class SpanInput:
    """A reasoning span plus the surrounding rails the processor uses for context."""
    problem: str
    answer: str
    reasoning: str


@dataclass
class ProcessResult:
    mode: str                        # "compress" | "invert" | "noop"
    original: str
    processed: str
    original_tokens: int
    processed_tokens: int

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.processed_tokens / self.original_tokens


def _format_compress_user(span: SpanInput) -> str:
    return (
        "PROBLEM:\n"
        f"{span.problem.strip() or '(no surrounding problem context provided)'}\n\n"
        "ANSWER:\n"
        f"{span.answer.strip() or '(no final answer provided)'}\n\n"
        "REASONING:\n"
        f"{span.reasoning}"
    )


def _format_invert_user(span: SpanInput) -> str:
    return (
        "Problem:\n"
        f"{span.problem.strip() or '(no surrounding problem context provided)'}\n\n"
        "Model's final answer:\n"
        f"{span.answer.strip() or '(no final answer provided)'}\n\n"
        "Reasoning Bubbles:\n"
        f"{span.reasoning}\n\n"
        "Reconstruct the full reasoning trace."
    )


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class Processor:
    """Run a single processing agent over a span.

    `mode`:
        rules    — pure regex compression of the reasoning text (no LLM,
                   ignores PROBLEM / ANSWER context)
        compress — terse LLM rewrite at intensity `level` (lite|full)
        invert   — LLM expansion of a compressed skeleton
        noop     — passthrough (used for baselines)
    """

    def __init__(
        self,
        mode: str = "compress",
        *,
        level: str = "lite",
        model: Optional[str] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        if mode not in ("rules", "compress", "invert", "noop"):
            raise ValueError(f"unknown processor mode: {mode!r}")
        self.mode = mode
        self.level = level
        self.model = model
        # rules and noop never touch the network — only build a client if needed.
        self._client = client if client is not None else (
            LLMClient(model=model) if mode in ("compress", "invert") else None
        )

    # ---- single-span ----

    def process(self, span: SpanInput) -> ProcessResult:
        text = span.reasoning
        if not text or not text.strip():
            return ProcessResult(self.mode, text, text, 0, 0)

        if self.mode == "noop":
            return ProcessResult(
                self.mode, text, text,
                estimate_tokens(text), estimate_tokens(text),
            )

        if self.mode == "rules":
            r = compress_rules(text, level=self.level)
            return ProcessResult(
                "rules", r.original, r.compressed,
                r.original_tokens, r.compressed_tokens,
            )

        if self.mode == "compress":
            system = prompts.compress_prompt(self.level)
            user = _format_compress_user(span)
        else:  # invert
            system = prompts.invert_prompt()
            user = _format_invert_user(span)

        try:
            assert self._client is not None
            out = self._client.chat(system=system, user=user)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s agent failed (%d chars in); falling back to original: %s",
                self.mode, len(text), exc,
            )
            out = text

        return ProcessResult(
            self.mode,
            original=text,
            processed=out,
            original_tokens=estimate_tokens(text),
            processed_tokens=estimate_tokens(out),
        )

    def __call__(self, span: SpanInput) -> ProcessResult:
        return self.process(span)

    # ---- batched ----

    def process_batch(
        self, spans: List[SpanInput], max_workers: int = 8
    ) -> List[ProcessResult]:
        """Process many spans concurrently. Order preserved.

        When the `tokamak_engine` Rust orchestrator is available AND the
        client is configured against an OpenAI-compatible endpoint, the
        whole batch is dispatched through tokio with semaphore-bounded
        concurrency. Otherwise a Python ThreadPoolExecutor parallelizes
        per-span calls.

        Set max_workers close to the endpoint's scheduler width
        (e.g. vLLM `--max-num-seqs`).
        """
        if not spans:
            return []
        if self.mode in ("rules", "noop"):
            return [self.process(s) for s in spans]

        assert self._client is not None
        if self.mode == "compress":
            system = prompts.compress_prompt(self.level)
            format_user = _format_compress_user
        else:
            system = prompts.invert_prompt()
            format_user = _format_invert_user

        items = [{"system": system, "user": format_user(s)} for s in spans]

        try:
            outs = self._client.batch_chat(items, max_workers=max(1, max_workers))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "batch %s failed (%d spans); falling back to per-span: %s",
                self.mode, len(spans), exc,
            )
            return [self.process(s) for s in spans]

        results: List[ProcessResult] = []
        for span, out in zip(spans, outs):
            text = span.reasoning
            # Empty out = call failed (Rust path) → fall back to original.
            processed = out if (out and out.strip()) else text
            results.append(ProcessResult(
                mode=self.mode,
                original=text,
                processed=processed,
                original_tokens=estimate_tokens(text),
                processed_tokens=estimate_tokens(processed),
            ))
        return results
