"""Triple-format exporters: Axolotl, ShareGPT, Unsloth."""

from __future__ import annotations

from typing import Any, Dict, List

from .quality import _content_string, _extract_messages

DEFAULT_SYSTEM = (
    "You are a careful, terse reasoning model. Internal reasoning is "
    "rendered in caveman-lite style: no filler, no hedging, all technical "
    "substance preserved."
)


def _normalize(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Coerce content blocks down to plain strings."""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "")
        if role == "human":
            role = "user"
        if role == "gpt":
            role = "assistant"
        out.append({"role": role, "content": _content_string(m)})
    return out


def axolotl(trace: Dict[str, Any], score: float, trace_id: str, error_class: str) -> Dict[str, Any]:
    msgs = _normalize(_extract_messages(trace))
    if not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    return {
        "id": trace_id,
        "messages": msgs,
        "metadata": {
            "quality_score": round(score, 4),
            "error_class": error_class,
        },
    }


def sharegpt(trace: Dict[str, Any]) -> Dict[str, Any]:
    msgs = _normalize(_extract_messages(trace))
    if not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    rolemap = {"user": "human", "assistant": "gpt", "system": "system"}
    convos = [
        {"from": rolemap.get(m["role"], m["role"]), "value": m["content"]}
        for m in msgs
    ]
    return {"conversations": convos}


def unsloth(trace: Dict[str, Any], score: float, trace_id: str, error_class: str) -> Dict[str, Any]:
    msgs = _normalize(_extract_messages(trace))
    if not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    return {
        "id": trace_id,
        "messages": msgs,
        "score": round(score, 4),
        "error_class": error_class,
    }
