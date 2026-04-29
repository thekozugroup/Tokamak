"""Locate reasoning spans inside heterogeneous trace formats.

Supports:
- Hermes-style JSONL — top-level `messages` list
- Claude/Anthropic message JSONL — `messages` list with `content` blocks
- OpenAI o1-style — message has a `reasoning` field
- Inline tag form — `<thinking>...</thinking>` etc. anywhere in any string

A reasoning span is described as an editable accessor: a (getter, setter) pair
so the patcher can rewrite in place without knowing the trace shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Tuple

REASONING_TAGS = ["thinking", "think", "reasoning", "thought", "analyze", "scratchpad"]

_TAG_RE = re.compile(
    r"<(" + "|".join(REASONING_TAGS) + r")>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class Span:
    source: str             # "tag:thinking" / "field:reasoning" / "assistant_internal"
    text: str               # current contents
    apply: Callable[[str], None]   # write replacement back into the trace


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
    """Build spans for every <thinking>…</thinking> match inside a string.

    The setter closes over a mutable buffer so multiple spans in the same
    string each rewrite their own match without clobbering siblings.
    """
    matches = list(_TAG_RE.finditer(string_value))
    if not matches:
        return []

    buf = [string_value]   # single-element list so closures share state

    def make_setter(tag: str, span_start: int, span_end: int):
        def apply(new_inner: str) -> None:
            current = buf[0]
            replacement = f"<{tag}>{new_inner}</{tag}>"
            # Re-find by original outer text since indices shift after edits.
            # We stored the original outer text at construction time.
            # See `outer_orig` captured below.
            buf[0] = current.replace(make_setter.outer_orig, replacement, 1)  # type: ignore[attr-defined]
            on_replace(buf[0])
        return apply

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


def find_reasoning_spans(trace: Dict[str, Any]) -> List[Span]:
    """Walk a trace and yield every editable reasoning span we can find."""
    spans: List[Span] = []

    for path, node in _walk(trace, []):
        # Case 1: a string-valued leaf that contains <thinking>...</thinking>.
        if isinstance(node, str) and "<" in node:
            if not _TAG_RE.search(node):
                continue
            set_at_path = _setter(trace, path)
            spans.extend(_replace_tag_spans(node, set_at_path))
            continue

        # Case 2: dict with a dedicated reasoning field (OpenAI o1-style).
        if isinstance(node, dict):
            for field in ("reasoning", "thinking", "analysis", "thought"):
                val = node.get(field)
                if isinstance(val, str) and val.strip():
                    set_field = _setter(trace, path + [field])
                    spans.append(Span(
                        source=f"field:{field}",
                        text=val,
                        apply=set_field,
                    ))
                # Anthropic-style: a content block { type: "thinking", thinking: "..." }
                if node.get("type") == "thinking" and isinstance(node.get("thinking"), str):
                    set_block = _setter(trace, path + ["thinking"])
                    spans.append(Span(
                        source="block:thinking",
                        text=node["thinking"],
                        apply=set_block,
                    ))
                    break

    # Dedup: same path may match through both walks for OpenAI shape.
    seen = set()
    unique: List[Span] = []
    for s in spans:
        key = (s.source, id(s.apply))
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique
