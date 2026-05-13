"""Triple-format exporters: Axolotl, ShareGPT, Unsloth."""

from __future__ import annotations

from typing import Any, Dict, List

from .quality import _content_string, _extract_messages

DEFAULT_SYSTEM = (
    "You are a careful, terse reasoning model. Internal reasoning is "
    "rendered in terse style: no filler, no hedging, all technical "
    "substance preserved."
)


def _signal(trace: Dict[str, Any]) -> Any:
    """Return the QAQC signal stamped onto the row by the pipeline, or None."""
    md = trace.get("metadata")
    if isinstance(md, dict) and "signal" in md:
        return md["signal"]
    if "signal" in trace:
        return trace["signal"]
    return None


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
    metadata = {
        "quality_score": round(score, 4),
        "error_class": error_class,
    }
    sig = _signal(trace)
    if sig is not None:
        metadata["signal"] = sig
    return {
        "id": trace_id,
        "messages": msgs,
        "metadata": metadata,
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
    row: Dict[str, Any] = {
        "id": trace_id,
        "messages": msgs,
        "score": round(score, 4),
        "error_class": error_class,
    }
    sig = _signal(trace)
    if sig is not None:
        row["signal"] = sig
    return row
