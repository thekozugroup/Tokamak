"""Caveman compression — both rules-based (free, fast) and LLM-based (paid, accurate).

Rules engine never touches:
- Code fences ``` ... ``` or ~~~ ... ~~~
- Inline code  `...`
- URLs / file paths / identifiers / numbers
- Quoted strings "..."
- Anything inside <tool_call>...</tool_call> or similar tags
"""

from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Tuple

from . import prompts

# ---------------------------------------------------------------------------
# Protected spans — extract, compress around them, restore
# ---------------------------------------------------------------------------

_PROTECT_PATTERNS: List[re.Pattern] = [
    re.compile(r"```.*?```", re.DOTALL),          # fenced code
    re.compile(r"~~~.*?~~~", re.DOTALL),          # alt fenced code
    re.compile(r"`[^`\n]+`"),                      # inline code
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL),
    re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL),
    re.compile(r"<output>.*?</output>", re.DOTALL),
    re.compile(r"https?://\S+"),
    re.compile(r"/[\w./\-]+"),                     # unix paths
    re.compile(r"[A-Za-z]:\\[\w.\\\-]+"),          # windows paths
    re.compile(r'"[^"\n]{0,200}"'),               # short quoted strings
    re.compile(r"'[^'\n]{0,200}'"),
    re.compile(r"\b\d+(?:\.\d+)*\b"),             # numbers / versions
]


def _protect(text: str) -> Tuple[str, List[str]]:
    """Replace protected spans with placeholders. Return (masked_text, originals)."""
    originals: List[str] = []

    def stash(m: re.Match) -> str:
        originals.append(m.group(0))
        return f"\x00P{len(originals) - 1}\x00"

    for pat in _PROTECT_PATTERNS:
        text = pat.sub(stash, text)
    return text, originals


def _restore(text: str, originals: List[str]) -> str:
    def unstash(m: re.Match) -> str:
        idx = int(m.group(1))
        return originals[idx]
    return re.sub(r"\x00P(\d+)\x00", unstash, text)


# ---------------------------------------------------------------------------
# Rule-based caveman lite
# ---------------------------------------------------------------------------

# Words to delete entirely (case-insensitive, word-boundary).
_FILLER = [
    "just", "really", "basically", "actually", "simply", "literally",
    "obviously", "clearly", "essentially", "fundamentally", "honestly",
    "indeed", "very", "quite", "rather", "somewhat", "kind of", "sort of",
]

_PLEASANTRIES = [
    "of course", "sure thing", "no problem", "happy to help",
    "let me think", "let me see", "let's see", "okay so", "ok so",
    "alright so", "alright", "great", "perfect", "awesome",
    "first off", "first of all", "to begin with",
    "i'll start by", "i will start by", "let me start by",
    "i want to", "i'm going to", "i am going to", "i'll go ahead and",
    "now i", "now let me", "now let's", "next i'll", "next let me",
    # Sentence-opening pleasantries — only stripped when followed by punctuation
    # (handled via _OPENER_RE below, not in the bulk wordlist).
]

# Strip leading "Sure," / "Okay," / "Ok," / "Right," / "Well," at the start of
# a clause. We do not strip these mid-sentence because they may be content.
_OPENER_RE = re.compile(
    r"(?im)(?:(?<=^)|(?<=[.!?]\s)|(?<=\n))"
    r"(?:sure|okay|ok|right|well|yeah|yep|hmm)[\s,!.\-:]+"
)

_HEDGES = [
    "i think", "i believe", "i suppose", "i guess", "i'd say",
    "it seems", "it appears", "it would seem", "perhaps", "maybe",
    "possibly", "probably", "i'm not sure but", "if i recall correctly",
]

# Multi-word verbose -> short replacements.
_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bdue to the fact that\b", re.I), "because"),
    (re.compile(r"\bas a result of\b", re.I), "due to"),
    (re.compile(r"\bat this point in time\b", re.I), "now"),
    (re.compile(r"\bat this moment\b", re.I), "now"),
    (re.compile(r"\bin the event that\b", re.I), "if"),
    (re.compile(r"\bmake use of\b", re.I), "use"),
    (re.compile(r"\btake into account\b", re.I), "consider"),
    (re.compile(r"\bgive consideration to\b", re.I), "consider"),
    (re.compile(r"\bhas the ability to\b", re.I), "can"),
    (re.compile(r"\bis able to\b", re.I), "can"),
    (re.compile(r"\bin spite of the fact that\b", re.I), "though"),
    (re.compile(r"\bwith regard to\b", re.I), "re"),
    (re.compile(r"\bwith respect to\b", re.I), "re"),
    (re.compile(r"\bin terms of\b", re.I), "for"),
    (re.compile(r"\ba large number of\b", re.I), "many"),
    (re.compile(r"\ba small number of\b", re.I), "few"),
    (re.compile(r"\bthe vast majority of\b", re.I), "most"),
    (re.compile(r"\bimplement a solution for\b", re.I), "fix"),
    (re.compile(r"\bverify that\b", re.I), "check"),
    (re.compile(r"\bdetermine whether\b", re.I), "check if"),
    (re.compile(r"\bnotwithstanding\b", re.I), "though"),
]


def _wordlist_pattern(words: List[str]) -> re.Pattern:
    # word-boundary match; allows multi-word phrases.
    alts = sorted({re.escape(w) for w in words}, key=len, reverse=True)
    return re.compile(r"(?i)\b(?:" + "|".join(alts) + r")\b[ \t]*,?\s*")


_FILLER_RE = _wordlist_pattern(_FILLER)
_PLEAS_RE = _wordlist_pattern(_PLEASANTRIES)
_HEDGE_RE = _wordlist_pattern(_HEDGES)


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


@dataclass
class CompressResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


def estimate_tokens(text: str) -> int:
    """Cheap token estimate. ~4 chars/token for English; ~1 char/token for CJK."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def compress_rules(text: str, level: str = "lite") -> CompressResult:
    """Apply caveman lite/full rules without touching protected spans."""
    if not text or not text.strip():
        return CompressResult(text, text, 0, 0)

    masked, originals = _protect(text)
    out = masked

    # Sentence-opening pleasantries first ("Sure,", "Okay,", "Well,").
    out = _OPENER_RE.sub("", out)
    # Always: pleasantries + hedges + filler.
    out = _PLEAS_RE.sub("", out)
    out = _HEDGE_RE.sub("", out)
    out = _FILLER_RE.sub("", out)

    # Always: verbose -> short.
    for pat, repl in _REPLACEMENTS:
        out = pat.sub(repl, out)

    if level == "full":
        # Drop common articles when not at sentence start.
        out = re.sub(r"(?<=\S )\b(?:a|an|the)\b\s+", "", out, flags=re.I)
        # Telegraphic causality.
        out = re.sub(r"\b(so that|so)\b\s+", "-> ", out, flags=re.I)

    out = _collapse_whitespace(out)
    out = _restore(out, originals)

    return CompressResult(
        original=text,
        compressed=out,
        original_tokens=estimate_tokens(text),
        compressed_tokens=estimate_tokens(out),
    )


# ---------------------------------------------------------------------------
# LLM-based compression
# ---------------------------------------------------------------------------

class LLMCompressor:
    """Wraps Claude (SDK or CLI) to rewrite reasoning blocks."""

    def __init__(self, level: str = "lite", model: str | None = None) -> None:
        self.level = level
        self.system = prompts.system_prompt(level)
        self.model = model or os.environ.get("TOKAMAK_MODEL", "claude-sonnet-4-5")
        self._openai_base_url = os.environ.get("TOKAMAK_OPENAI_BASE_URL")
        self._openai_api_key = os.environ.get("TOKAMAK_OPENAI_API_KEY", "dummy")
        self._openai_client = None
        if self._openai_base_url:
            try:
                import openai  # type: ignore
                self._openai_client = openai.OpenAI(
                    base_url=self._openai_base_url,
                    api_key=self._openai_api_key,
                )
            except ImportError:
                raise RuntimeError(
                    "TOKAMAK_OPENAI_BASE_URL set but `openai` package not installed. "
                    "Install with: pip install openai"
                )

    def compress(self, text: str) -> CompressResult:
        if not text or not text.strip():
            return CompressResult(text, text, 0, 0)
        rewritten = self._call(text)
        return CompressResult(
            original=text,
            compressed=rewritten,
            original_tokens=estimate_tokens(text),
            compressed_tokens=estimate_tokens(rewritten),
        )

    # Allow the instance to be used as a Callable[[str], CompressResult],
    # preserving backwards compatibility with pipeline code that calls
    # `compressor(text)` directly.
    def __call__(self, text: str) -> CompressResult:
        return self.compress(text)

    def compress_batch(
        self, texts: List[str], max_workers: int = 8
    ) -> List[CompressResult]:
        """Compress multiple texts concurrently.

        Order is preserved: `result[i]` corresponds to `texts[i]`.

        Uses a ThreadPoolExecutor because each underlying call is a blocking
        HTTP request — the GIL is released during socket I/O, so threads
        saturate high-throughput endpoints (vLLM, TGI, SGLang) without
        requiring asyncio. Set `max_workers` close to the endpoint's
        `--max-num-seqs` for best throughput.
        """
        if not texts:
            return []
        if max_workers <= 1:
            return [self.compress(t) for t in texts]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(self.compress, texts))

    def _call(self, text: str) -> str:
        # OpenAI-compatible endpoint (e.g., vLLM) takes priority when configured
        if self._openai_client is not None:
            resp = self._openai_client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": text},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic  # type: ignore
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system,
                    messages=[{"role": "user", "content": text}],
                )
                return msg.content[0].text.strip()
            except ImportError:
                pass
        # Fallback: claude CLI
        try:
            result = subprocess.run(
                ["claude", "--print", "--system", self.system],
                input=text,
                text=True,
                capture_output=True,
                check=True,
                timeout=120,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "LLM compression requested but neither ANTHROPIC_API_KEY nor "
                "`claude` CLI is available."
            ) from exc


# ---------------------------------------------------------------------------
# Compressor selector
# ---------------------------------------------------------------------------

Compressor = Callable[[str], CompressResult]


def get_compressor(mode: str, level: str = "lite", model: str | None = None) -> Compressor:
    if mode == "rules":
        return lambda t: compress_rules(t, level=level)
    if mode == "llm":
        # Return the LLMCompressor instance itself. It is callable via __call__
        # (so existing `compressor(text)` call sites keep working) and exposes
        # `compress_batch` for the parallel pipeline path.
        return LLMCompressor(level=level, model=model)
    if mode == "noop":
        return lambda t: CompressResult(t, t, estimate_tokens(t), estimate_tokens(t))
    raise ValueError(f"unknown compress mode: {mode!r}")
