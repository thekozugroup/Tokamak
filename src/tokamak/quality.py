"""6-dimension quality scoring. Reported per TALOS doctrine; never used to filter."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

THINKING_TAGS = ["<thinking>", "<reasoning>", "<thought>", "<analyze>", "<scratchpad>"]
TOOL_CALL_TAGS = ["<tool_call>", "<function_calls>", "<invoke>"]
REFUSAL_PATTERNS = [
    r"i\s+can'?t?\s+(?:help|do|assist)",
    r"i'?m\s+sorry",
    r"i\s+(?:don't|do not)\s+know",
    r"unable\s+to",
    r"not\s+(?:able|allowed)\s+to",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _extract_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = trace.get("messages")
    if isinstance(msgs, list):
        return msgs
    convo = trace.get("conversations")
    if isinstance(convo, list):
        return convo
    return []


def _all_text(trace: Dict[str, Any]) -> str:
    return json.dumps(trace, ensure_ascii=False)


def _content_string(msg: Dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # Anthropic content-block list
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in c
        )
    return ""


WEIGHTS = {
    "reasoning_depth": 0.20,
    "structure": 0.20,
    "tool_calls": 0.15,
    "coherence": 0.15,
    "length": 0.15,
    "refusal": 0.15,
}


def score(trace: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    breakdown = {
        "reasoning_depth": _score_reasoning_depth(trace),
        "structure": _score_structure(trace),
        "tool_calls": _score_tool_calls(trace),
        "coherence": _score_coherence(trace),
        "length": _score_length(trace),
        "refusal": _score_refusal(trace),
    }
    composite = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)
    return composite, breakdown


def _score_reasoning_depth(trace: Dict[str, Any]) -> float:
    text = _all_text(trace).lower()
    msgs = _extract_messages(trace)
    thinking_hits = sum(1 for tag in THINKING_TAGS if tag in text)
    substantial = sum(1 for m in msgs if len(_content_string(m)) > 100)
    raw = 0.4 * min(thinking_hits / 2, 1.0) + 0.6 * min(substantial / 4, 1.0)
    return min(raw, 1.0)


def _score_structure(trace: Dict[str, Any]) -> float:
    msgs = _extract_messages(trace)
    if not msgs:
        return 0.0
    roles = [m.get("role", "") for m in msgs]
    has_user = any(r in ("user", "human") for r in roles)
    has_asst = any(r in ("assistant", "gpt") for r in roles)
    return float(has_user and has_asst) * (0.5 + 0.5 * min(len(msgs) / 4, 1.0))


def _score_tool_calls(trace: Dict[str, Any]) -> float:
    text = _all_text(trace).lower()
    return 1.0 if any(tag in text for tag in TOOL_CALL_TAGS) else 0.5


def _score_coherence(trace: Dict[str, Any]) -> float:
    msgs = _extract_messages(trace)
    if len(msgs) < 2:
        return 0.5
    # Penalize repeated identical assistant content (a flow-failure signal).
    asst = [
        _content_string(m) for m in msgs
        if m.get("role") in ("assistant", "gpt")
    ]
    if not asst:
        return 0.4
    unique = len(set(asst))
    return min(unique / max(len(asst), 1), 1.0)


def _score_length(trace: Dict[str, Any]) -> float:
    text_len = len(_all_text(trace))
    if text_len < 256:
        return 0.2
    if text_len > 100_000:
        return 0.5
    return 1.0


def _score_refusal(trace: Dict[str, Any]) -> float:
    msgs = _extract_messages(trace)
    asst_text = "\n".join(
        _content_string(m) for m in msgs
        if m.get("role") in ("assistant", "gpt")
    )
    if not asst_text:
        return 0.5
    refusals = len(_REFUSAL_RE.findall(asst_text))
    if refusals == 0:
        return 1.0
    return max(0.0, 1.0 - refusals * 0.25)
