"""Shared LLM client used by every agent (compress, invert, validate).

One client, three backends, picked in this order:

1. OpenAI-compatible endpoint (vLLM / TGI / SGLang / local server) when
   `TOKAMAK_OPENAI_BASE_URL` is set.
2. Anthropic SDK when `ANTHROPIC_API_KEY` is set.
3. The `claude` CLI as a last resort.

Configuration is all environment-driven so the same client serves every
agent without per-agent flag plumbing. Per-request timeout and retry are
shared across agents.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

logger = logging.getLogger("tokamak.llm")


# Rust orchestrator (optional). When present and an OpenAI-compatible
# endpoint is configured, batch_chat() routes through it for memory-safe
# tokio-driven concurrency.
try:  # pragma: no cover — import is environment-dependent
    import tokamak_engine as _rust_engine  # type: ignore
except ImportError:  # pragma: no cover
    _rust_engine = None


def _retryable_exceptions():
    """Return the tuple of openai exceptions worth retrying, or () if SDK absent."""
    try:
        import openai  # type: ignore
        return (openai.APITimeoutError, openai.APIConnectionError)
    except ImportError:
        return ()


class LLMClient:
    """Thin shared client. One instance per agent, sharing an HTTP connection
    pool when an OpenAI-compatible backend is used."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> None:
        self.model = model or os.environ.get("TOKAMAK_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout if timeout is not None else float(
            os.environ.get("TOKAMAK_OPENAI_TIMEOUT", "180")
        )
        self.retries = retries if retries is not None else int(
            os.environ.get("TOKAMAK_OPENAI_RETRIES", "2")
        )
        self._openai_base_url = os.environ.get("TOKAMAK_OPENAI_BASE_URL")
        self._openai_api_key = os.environ.get("TOKAMAK_OPENAI_API_KEY", "dummy")
        self._openai_client = None
        if self._openai_base_url:
            try:
                import openai  # type: ignore
                self._openai_client = openai.OpenAI(
                    base_url=self._openai_base_url,
                    api_key=self._openai_api_key,
                    timeout=self.timeout,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "TOKAMAK_OPENAI_BASE_URL set but `openai` package not installed. "
                    "Install with: pip install openai"
                ) from exc

    # ---- public ----

    @property
    def has_rust_engine(self) -> bool:
        return _rust_engine is not None and self._openai_client is not None

    def batch_chat(
        self, items: List[Dict[str, str]], max_workers: int = 8
    ) -> List[str]:
        """Run a batch of (system, user) chats. Order preserved.

        Routes through the Rust orchestrator when both the `tokamak_engine`
        module is importable AND an OpenAI-compatible endpoint is configured.
        Otherwise falls back to a Python ThreadPoolExecutor over `self.chat`.

        Failures: when the Rust path returns None for an item (permanent
        failure after all retries) the caller still gets a string — we
        substitute the user message's body for the failed item so the
        pipeline can fall back to the original reasoning. The Python
        fallback raises through `chat` and lets the caller catch.
        """
        if not items:
            return []
        if self.has_rust_engine:
            assert _rust_engine is not None
            results = _rust_engine.batch_chat(
                base_url=self._openai_base_url,
                api_key=self._openai_api_key,
                model=self.model,
                items=items,
                max_workers=max(1, max_workers),
                timeout_s=float(self.timeout),
                retries=int(self.retries),
                max_tokens=int(self.max_tokens),
                temperature=float(self.temperature),
            )
            # None -> empty string so callers can detect and fall back.
            return [r if isinstance(r, str) else "" for r in results]

        if max_workers <= 1:
            return [self.chat(i["system"], i["user"]) for i in items]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(
                ex.map(lambda it: self.chat(it["system"], it["user"]), items)
            )

    def chat(self, system: str, user: str) -> str:
        """Single chat exchange with retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._chat_once(system, user)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = _retryable_exceptions()
                if not isinstance(exc, retryable) or attempt == self.retries:
                    raise
                backoff = min(2 ** attempt, 30)
                logger.warning(
                    "LLM call transient error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.retries + 1, exc, backoff,
                )
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    # ---- backends ----

    def _chat_once(self, system: str, user: str) -> str:
        if self._openai_client is not None:
            resp = self._openai_client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
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
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text.strip()
            except ImportError:
                pass

        try:
            result = subprocess.run(
                ["claude", "--print", "--system", system],
                input=user,
                text=True,
                capture_output=True,
                check=True,
                timeout=max(self.timeout, 30),
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "LLM call requested but neither TOKAMAK_OPENAI_BASE_URL, "
                "ANTHROPIC_API_KEY, nor the `claude` CLI is available."
            ) from exc
