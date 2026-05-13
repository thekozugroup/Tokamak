"""Processor tests — covers rules / compress / invert / noop modes without
touching the network. LLM modes stub out the underlying LLMClient.chat."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from tokamak.processor import Processor, SpanInput


SAMPLE = SpanInput(
    problem="Why is my Postgres query slow?",
    answer="Check EXPLAIN ANALYZE; likely missing index.",
    reasoning=(
        "Sure, let me think. Basically the slow query is most likely "
        "because there's no index on the filter column. I'll suggest "
        "EXPLAIN ANALYZE first."
    ),
)


def test_noop_passthrough():
    p = Processor("noop")
    r = p.process(SAMPLE)
    assert r.processed == SAMPLE.reasoning
    assert r.processed_tokens == r.original_tokens


def test_rules_reduces_tokens_and_ignores_context():
    p = Processor("rules", level="lite")
    r = p.process(SAMPLE)
    assert r.processed_tokens < r.original_tokens
    assert "basically" not in r.processed.lower()
    # No LLM client should have been built for rules mode.
    assert p._client is None


def test_compress_calls_llm_with_problem_and_answer_rails():
    """compress mode must feed PROBLEM + ANSWER as rails, not just reasoning."""
    captured = {}

    def fake_chat(self, system, user):
        captured["system"] = system
        captured["user"] = user
        # Stand-in compression: uppercase.
        return SAMPLE.reasoning.upper()

    with patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        p = Processor("compress", level="lite")
        r = p.process(SAMPLE)

    assert r.processed == SAMPLE.reasoning.upper()
    # Rails must appear in the user message that goes to the LLM.
    assert "PROBLEM:" in captured["user"]
    assert "ANSWER:" in captured["user"]
    assert "REASONING:" in captured["user"]
    assert SAMPLE.problem in captured["user"]
    assert SAMPLE.answer in captured["user"]
    # System prompt is the compress prompt.
    assert "rewrite an assistant's internal reasoning" in captured["system"].lower()


def test_invert_calls_llm_with_bubbles_format():
    """invert mode must use the Trace-Inverter-style user template."""
    captured = {}

    def fake_chat(self, system, user):
        captured["user"] = user
        captured["system"] = system
        return "expanded reasoning trace"

    with patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        p = Processor("invert")
        r = p.process(SAMPLE)

    assert r.processed == "expanded reasoning trace"
    assert "Reasoning Bubbles:" in captured["user"]
    assert "Model's final answer:" in captured["user"]
    assert "trace inversion model" in captured["system"].lower()


def test_compress_falls_back_to_original_on_llm_error():
    def boom_chat(self, system, user):
        raise RuntimeError("simulated endpoint down")

    with patch("tokamak.llm_client.LLMClient.chat", new=boom_chat):
        p = Processor("compress", level="lite")
        r = p.process(SAMPLE)

    # Permanent failure must NOT crash the pipeline; fall back to original.
    assert r.processed == SAMPLE.reasoning


def test_process_batch_preserves_order_and_runs_concurrently():
    """Batch path must keep order even with thread pool concurrency."""
    tracker: list = []
    lock = threading.Lock()

    def fake_chat(self, system, user):
        with lock:
            tracker.append(("start", threading.get_ident()))
        time.sleep(0.02)
        with lock:
            tracker.append(("end", threading.get_ident()))
        # echo the REASONING block prefix so we can verify ordering
        return user

    with patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        p = Processor("compress", level="lite")
        spans = [
            SpanInput(problem=f"p{i}", answer=f"a{i}", reasoning=f"r{i}")
            for i in range(6)
        ]
        results = p.process_batch(spans, max_workers=4)

    assert len(results) == 6
    # Each result must reference its own reasoning string by index.
    for i, r in enumerate(results):
        assert f"r{i}" in r.processed

    # And ≥2 distinct threads must have started a request before any ended.
    threads_seen: set = set()
    for kind, tid in tracker:
        if kind == "end":
            break
        threads_seen.add(tid)
    assert len(threads_seen) > 1, "expected concurrent starts across threads"


def test_process_batch_routes_through_llm_client_batch_chat():
    """When tokamak_engine is available the batch path delegates to
    LLMClient.batch_chat which dispatches Rust-side. We verify the call site
    by intercepting batch_chat."""
    captured: list = []

    def fake_batch(self, items, max_workers=1):
        captured.append((len(items), max_workers))
        return [f"OUT-{i}" for i, _ in enumerate(items)]

    with patch("tokamak.llm_client.LLMClient.batch_chat", new=fake_batch):
        p = Processor("compress", level="lite")
        spans = [SpanInput("p", "a", f"r{i}") for i in range(3)]
        results = p.process_batch(spans, max_workers=3)

    assert captured == [(3, 3)]
    assert [r.processed for r in results] == ["OUT-0", "OUT-1", "OUT-2"]


def test_process_batch_empty_fallback_to_original_on_blank_output():
    """If batch_chat returns an empty string for an item, the pipeline must
    fall back to the original reasoning so SFT data is never lost."""
    def fake_batch(self, items, max_workers=1):
        # Simulate one failed call (empty string) and one successful.
        return ["", "OK"]

    with patch("tokamak.llm_client.LLMClient.batch_chat", new=fake_batch):
        p = Processor("compress", level="lite")
        spans = [SpanInput("p", "a", "fail-me"), SpanInput("p", "a", "succeed")]
        results = p.process_batch(spans, max_workers=2)

    assert results[0].processed == "fail-me"   # fell back
    assert results[1].processed == "OK"
