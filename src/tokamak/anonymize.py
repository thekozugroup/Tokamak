"""Regex + entropy PII redaction. Ported and slimmed from TALOS-trace-curator."""

from __future__ import annotations

import re
from typing import List, Tuple

# (pattern, replacement) — ordered, applied in sequence.
_DEFAULT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    (re.compile(r"sk-(?:ant|live|test)?[_-]?[a-zA-Z0-9_-]{20,}"), "<API_KEY>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<AWS_KEY>"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "<GH_TOKEN>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
    (re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+"), "<HOME>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"-----BEGIN [A-Z ]+ KEY-----.*?-----END [A-Z ]+ KEY-----", re.DOTALL), "<PRIVATE_KEY>"),
]


def _entropy(s: str) -> float:
    """Shannon entropy in bits per char. Cheap secret detector."""
    from math import log2
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((c / n) * log2(c / n) for c in counts.values())


_TOKEN_LIKE = re.compile(r"[A-Za-z0-9_\-]{24,}")


class Anonymizer:
    def __init__(self, level: str = "strict") -> None:
        self.level = level
        self.patterns = list(_DEFAULT_PATTERNS)

    def redact(self, text: str) -> Tuple[str, int]:
        if not text:
            return text, 0
        count = 0
        for pat, repl in self.patterns:
            matches = pat.findall(text)
            count += len(matches)
            text = pat.sub(repl, text)
        if self.level == "strict":
            text, n = self._entropy_redact(text)
            count += n
        return text, count

    def _entropy_redact(self, text: str) -> Tuple[str, int]:
        """Replace high-entropy tokens that look like secrets."""
        n = 0

        def maybe(m: re.Match) -> str:
            nonlocal n
            tok = m.group(0)
            # Skip obvious code identifiers and version-like strings.
            if "_" in tok and not any(c.isdigit() for c in tok):
                return tok
            if _entropy(tok) >= 4.0:
                n += 1
                return "<TOKEN>"
            return tok

        text = _TOKEN_LIKE.sub(maybe, text)
        return text, n


def redact_trace(trace, level: str = "strict") -> Tuple[object, int]:
    """Recursively redact every string in the trace."""
    anon = Anonymizer(level=level)
    total = [0]

    def walk(obj):
        if isinstance(obj, str):
            new, n = anon.redact(obj)
            total[0] += n
            return new
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    return walk(trace), total[0]
