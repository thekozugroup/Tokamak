"""Locate reasoning spans inside heterogeneous trace formats.

Supports:
- Hermes-style JSONL — top-level `messages` list
- Claude/Anthropic message JSONL — `messages` list with `content` blocks
- OpenAI o1-style — message has a `reasoning` field
- Inline tag form — `<thinking>...</thinking>` etc. anywhere in any string

A reasoning span is described as an editable accessor: a (getter, setter)
pair so the patcher can rewrite in place without knowing the trace shape.

Each span optionally carries the surrounding PROBLEM and ANSWER context
extracted from the same message list, so downstream agents can be fed the
full rail (problem + answer + reasoning) instead of the reasoning alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

REASONING_TAGS = ["thinking", "think", "reasoning", "thought", "analyze", "scratchpad"]

_TAG_RE = re.compile(
    r"<(" + "|".join(REASONING_TAGS) + r")>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class Span:
    source: str                                # "tag:thinking" / "field:reasoning" / "block:thinking"
    text: str                                  # current contents of the reasoning span
    apply: Callable[[str], None]               # write replacement back into the trace
    problem: str = ""                          # surrounding user / system context
    answer: str = ""                           # the visible assistant answer that follows


def _walk(obj: Any, path: List[Any]) -> Iterator[Tuple[List[Any], Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + [k])
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, path + [i])


def _setter(root: Any, path: List[Any]) -> Callable[[Any], None]:
    def set_(value: Any) -> None:
        cursor = root
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
    return set_


def _replace_tag_spans(string_value: str, on_replace: Callable[[str], None]) -> List[Span]:
    """Build spans for every <thinking>...</thinking> match inside a string.

    The setter closes over a mutable buffer so multiple spans in the same
    string each rewrite their own match without clobbering siblings.
    """
    matches = list(_TAG_RE.finditer(string_value))
    if not matches:
        return []

    buf = [string_value]   # single-element list so closures share state

    spans: List[Span] = []
    for m in matches:
        tag = m.group(1)
        inner = m.group(2)
        outer_orig = m.group(0)

        def make(tag=tag, outer_orig=outer_orig):
            def apply(new_inner: str) -> None:
                replacement = f"<{tag}>{new_inner}</{tag}>"
                buf[0] = buf[0].replace(outer_orig, replacement, 1)
                on_replace(buf[0])
            return apply

        spans.append(Span(source=f"tag:{tag.lower()}", text=inner, apply=make()))
    return spans


# ---------------------------------------------------------------------------
# Context extraction — problem + answer pair surrounding a reasoning span
# ---------------------------------------------------------------------------

def _strip_reasoning_tags(text: str) -> str:
    """Remove all reasoning tag spans from a string to recover the visible answer."""
    return _TAG_RE.sub("", text).strip()


def _gather_problem(messages: List[Dict[str, Any]], assistant_idx: int) -> str:
    """Collect user/system messages preceding `assistant_idx` into a single problem string."""
    parts: List[str] = []
    for m in messages[:assistant_idx]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str):
            # Anthropic content-block form: join text blocks.
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                continue
        if role in ("system", "user", "human"):
            parts.append(content.strip())
    return "\n\n".join(p for p in parts if p)


def _extract_answer(message: Dict[str, Any]) -> str:
    """Pull the visible (non-reasoning) text out of an assistant message."""
    content = message.get("content")
    if isinstance(content, str):
        return _strip_reasoning_tags(content)
    if isinstance(content, list):
        # Anthropic-style content blocks: join the text blocks (drop thinking blocks).
        chunks: List[str] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                chunks.append(_strip_reasoning_tags(b["text"]))
        return "\n\n".join(c for c in chunks if c).strip()
    return ""


def _attach_context(
    span: Span, messages: List[Dict[str, Any]], assistant_idx: int
) -> None:
    span.problem = _gather_problem(messages, assistant_idx)
    span.answer = _extract_answer(messages[assistant_idx])


def find_reasoning_spans(trace: Dict[str, Any]) -> List[Span]:
    """Walk a trace and yield every editable reasoning span we can find.

    Spans returned by this function include `.problem` and `.answer`
    populated from the surrounding message list when one is available.
    """
    spans: List[Span] = []

    messages = trace.get("messages") if isinstance(trace, dict) else None

    # Pass 1: scan messages directly so we can attach problem/answer context.
    if isinstance(messages, list):
        for ai, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in ("assistant", "model"):
                continue
            content = msg.get("content")

            if isinstance(content, str) and _TAG_RE.search(content):
                set_at_path = _setter(trace, ["messages", ai, "content"])
                new_spans = _replace_tag_spans(content, set_at_path)
                for s in new_spans:
                    _attach_context(s, messages, ai)
                spans.extend(new_spans)
                continue

            if isinstance(content, list):
                for bi, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    # Anthropic content-block thinking
                    if block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                        set_block = _setter(trace, ["messages", ai, "content", bi, "thinking"])
                        s = Span(
                            source="block:thinking",
                            text=block["thinking"],
                            apply=set_block,
                        )
                        _attach_context(s, messages, ai)
                        spans.append(s)
                    # OpenAI-style "reasoning" field on a content block
                    for fld in ("reasoning", "thought", "analysis"):
                        val = block.get(fld)
                        if isinstance(val, str) and val.strip():
                            set_field = _setter(trace, ["messages", ai, "content", bi, fld])
                            s = Span(
                                source=f"field:{fld}",
                                text=val,
                                apply=set_field,
                            )
                            _attach_context(s, messages, ai)
                            spans.append(s)

            # OpenAI o1: top-level reasoning field on the message itself.
            for fld in ("reasoning", "thinking", "analysis", "thought"):
                val = msg.get(fld)
                if isinstance(val, str) and val.strip():
                    set_field = _setter(trace, ["messages", ai, fld])
                    s = Span(
                        source=f"field:{fld}",
                        text=val,
                        apply=set_field,
                    )
                    _attach_context(s, messages, ai)
                    spans.append(s)

    # Pass 2: catch any reasoning tags that live outside the messages array.
    if not spans:
        for path, node in _walk(trace, []):
            if isinstance(node, str) and "<" in node and _TAG_RE.search(node):
                set_at_path = _setter(trace, path)
                spans.extend(_replace_tag_spans(node, set_at_path))

    # Dedup.
    seen = set()
    unique: List[Span] = []
    for s in spans:
        key = (s.source, id(s.apply))
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique
