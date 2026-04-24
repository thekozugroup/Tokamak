"""Lexical dedup. Optional semantic dedup if sentence-transformers is installed.

Boilerplate tool-call JSON is stripped before comparison so traces with
identical tool calls but different reasoning are still treated as distinct
(faithful to TALOS).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from .quality import _content_string, _extract_messages

_TOOL_BLOCK = re.compile(
    r"<(tool_call|function_calls|tool_result|output)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def _signature(trace: Dict[str, Any]) -> str:
    parts: List[str] = []
    for m in _extract_messages(trace):
        text = _content_string(m)
        text = _TOOL_BLOCK.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(f"{m.get('role','?')}:{text}")
    return "\n".join(parts)


def _shingles(text: str, k: int = 5) -> Set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup(traces: Iterable[Dict[str, Any]], max_similarity: float = 0.92) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    kept_shingles: List[Set[str]] = []
    dropped = 0

    for trace in traces:
        sig = _signature(trace)
        sh = _shingles(sig)
        is_dup = False
        for existing in kept_shingles:
            if _jaccard(sh, existing) >= max_similarity:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(trace)
        kept_shingles.append(sh)
    return kept, dropped
