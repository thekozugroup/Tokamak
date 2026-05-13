"""QAQC validation agent.

A Validator grades a processed span against its original on a 0..1 scale.
Three axes are returned (logical fidelity, content fidelity, safety) plus a
combined `signal`. The signal is what lands in the dataset row's `signal`
column so downstream training can filter or weight by quality.

Mirrored validation runs N validators in parallel against the same input and
returns the *minimum* signal (worst-judge rule — one bad reading dominates,
matching the model card recommendation that high-risk data deserves strict
QA).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from . import prompts
from .llm_client import LLMClient
from .processor import SpanInput

logger = logging.getLogger("tokamak.validator")


@dataclass
class ValidationResult:
    signal: float                       # 0..1 — combined score (min of axes)
    logical_fidelity: float
    content_fidelity: float
    safety: float
    notes: str = ""
    raw: str = ""                       # raw model output, for audit
    per_judge: List[float] = field(default_factory=list)  # mirrored-judge scores


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_score(raw: str) -> ValidationResult:
    """Pull the first JSON object out of the model output and clamp scores."""
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        logger.warning("validator: no JSON object found in output; defaulting to 0.0")
        return ValidationResult(0.0, 0.0, 0.0, 0.0, notes="parse_error", raw=raw)
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("validator: JSON decode failed (%s); defaulting to 0.0", exc)
        return ValidationResult(0.0, 0.0, 0.0, 0.0, notes="parse_error", raw=raw)

    def _f(key: str) -> float:
        try:
            v = float(obj.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        if v != v:  # NaN
            return 0.0
        return max(0.0, min(1.0, v))

    logical = _f("logical_fidelity")
    content = _f("content_fidelity")
    safety = _f("safety")
    # If the model returned its own signal, accept it; else use worst-axis.
    if "signal" in obj:
        signal = _f("signal")
    else:
        signal = min(logical, content, safety)
    notes = str(obj.get("notes", "")).strip()
    return ValidationResult(
        signal=signal,
        logical_fidelity=logical,
        content_fidelity=content,
        safety=safety,
        notes=notes,
        raw=raw,
    )


def _format_validate_user(span: SpanInput, processed: str, mode: str) -> str:
    return (
        "PROBLEM:\n"
        f"{span.problem.strip() or '(none)'}\n\n"
        "ANSWER:\n"
        f"{span.answer.strip() or '(none)'}\n\n"
        f"MODE: {mode}\n\n"
        "ORIGINAL:\n"
        f"{span.reasoning}\n\n"
        "PROCESSED:\n"
        f"{processed}"
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class Validator:
    """Grade a processed span on 0..1. Optionally mirror N judges and take the
    worst score for robustness."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        client: Optional[LLMClient] = None,
        mirrors: int = 1,
    ) -> None:
        if mirrors < 1:
            raise ValueError("mirrors must be >= 1")
        self.mirrors = mirrors
        # Slightly tighter cap on validation max_tokens — JSON output is tiny.
        self._client = client if client is not None else LLMClient(
            model=model, max_tokens=512
        )
        self.system = prompts.validate_prompt()

    def validate(self, span: SpanInput, processed: str, mode: str) -> ValidationResult:
        if self.mirrors == 1:
            return self._single(span, processed, mode)
        with ThreadPoolExecutor(max_workers=self.mirrors) as ex:
            results = list(
                ex.map(lambda _i: self._single(span, processed, mode),
                       range(self.mirrors))
            )
        # Mirrored: worst-judge rule for the headline signal.
        per_judge = [r.signal for r in results]
        worst_idx = min(range(len(results)), key=lambda i: results[i].signal)
        agg = results[worst_idx]
        agg.per_judge = per_judge
        return agg

    def _single(self, span: SpanInput, processed: str, mode: str) -> ValidationResult:
        user = _format_validate_user(span, processed, mode)
        try:
            raw = self._client.chat(system=self.system, user=user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("validator call failed (%s); defaulting to 0.0", exc)
            return ValidationResult(0.0, 0.0, 0.0, 0.0, notes=f"call_error: {exc}")
        return _parse_score(raw)
