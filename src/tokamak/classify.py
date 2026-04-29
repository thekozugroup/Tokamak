"""5-factor error taxonomy for agent traces."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from .quality import _extract_messages, _content_string

ERROR_CLASSES = (
    "tool_failure",
    "syntax_error",
    "reasoning_error",
    "safety_refusal",
    "timeout_stall",
    "none",
)

_TOOL_FAIL = re.compile(
    r"(?i)\b(?:tool\s+(?:failed|error|timeout)|exit\s+code\s+[1-9]|"
    r"command\s+not\s+found|error\s*:\s*\w+|traceback)"
)
_SYNTAX = re.compile(
    r"(?i)\b(syntaxerror|parseerror|unexpected\s+token|invalid\s+syntax|"
    r"unmatched\s+(?:bracket|paren|brace))"
)
_REASONING = re.compile(
    r"(?i)\b(i\s+made\s+an?\s+(?:mistake|error)|that\s+was\s+wrong|"
    r"let\s+me\s+reconsider|i\s+was\s+(?:wrong|incorrect)|"
    r"actually,?\s+(?:that's|that\s+is)\s+(?:wrong|incorrect))"
)
_SAFETY = re.compile(
    r"(?i)\b(i\s+can'?t\s+(?:help|assist|do)|i'?m\s+(?:not\s+able|unable)|"
    r"i\s+(?:must|cannot)\s+decline|against\s+my\s+guidelines)"
)
_TIMEOUT = re.compile(
    r"(?i)\b(timeout|timed\s+out|deadline\s+exceeded|hung|stalled|"
    r"no\s+response|connection\s+reset)"
)


def classify(trace: Dict[str, Any]) -> str:
    text = json.dumps(trace, ensure_ascii=False)
    if _TOOL_FAIL.search(text):
        return "tool_failure"
    if _SYNTAX.search(text):
        return "syntax_error"
    if _SAFETY.search(text):
        return "safety_refusal"
    if _TIMEOUT.search(text):
        return "timeout_stall"
    if _REASONING.search(text):
        return "reasoning_error"
    return "none"
