"""Tests for LLMCompressor.compress_batch concurrency."""

from __future__ import annotations

import threading
import time

import pytest

from tokamak.caveman import CompressResult, LLMCompressor, get_compressor


def _make_compressor(per_call_delay: float = 0.0, tracker: list | None = None) -> LLMCompressor:
    """LLMCompressor with its `_call` stubbed so no HTTP is needed."""
    c = LLMCompressor()
    lock = threading.Lock()

    def fake_call(text: str) -> str:
        if tracker is not None:
            with lock:
                tracker.append(("start", threading.get_ident(), time.perf_counter()))
        if per_call_delay:
            time.sleep(per_call_delay)
        if tracker is not None:
            with lock:
                tracker.append(("end", threading.get_ident(), time.perf_counter()))
        # Deterministic "compression": uppercase stand-in.
        return text.upper()

    c._call = fake_call  # type: ignore[assignment]
    return c


def test_batch_empty_input():
    c = _make_compressor()
    assert c.compress_batch([]) == []


def test_batch_preserves_order():
    c = _make_compressor()
    texts = ["alpha", "bravo", "charlie", "delta"]
    results = c.compress_batch(texts, max_workers=4)
    assert [r.compressed for r in results] == [t.upper() for t in texts]
    assert [r.original for r in results] == texts


def test_batch_max_workers_one_is_sequential():
    c = _make_compressor()
    results = c.compress_batch(["a", "b", "c"], max_workers=1)
    assert [r.compressed for r in results] == ["A", "B", "C"]


def test_batch_uses_multiple_threads():
    tracker: list = []
    c = _make_compressor(per_call_delay=0.05, tracker=tracker)
    c.compress_batch(["a"] * 8, max_workers=4)
    # With 4 workers and 50ms per call, 8 calls in strict serial = 400ms.
    # Concurrent execution overlaps: observed tids > 1 and start events
    # from >=2 distinct tids appear before any end event.
    starts_before_first_end: set = set()
    for kind, tid, _ts in tracker:
        if kind == "end":
            break
        if kind == "start":
            starts_before_first_end.add(tid)
    assert len(starts_before_first_end) > 1, (
        "expected concurrent starts across threads, saw only one"
    )


def test_get_compressor_llm_mode_exposes_batch():
    """Regression: get_compressor('llm') must return an object with
    compress_batch so pipeline.run can detect and use the concurrent path.
    A prior version returned a bound method (engine.compress), silently
    disabling parallelism because hasattr(bound_method, 'compress_batch')
    is False.
    """
    c = get_compressor("llm")
    assert callable(c), "compressor must stay callable for single-text sites"
    assert hasattr(c, "compress_batch"), (
        "compressor must expose compress_batch so pipeline enables concurrency"
    )


def test_batch_preserves_empty_strings():
    c = _make_compressor()
    results = c.compress_batch(["real", "", "  "], max_workers=2)
    # Empty / whitespace-only texts short-circuit inside compress() without
    # calling _call, and return the original unchanged.
    assert results[0].compressed == "REAL"
    assert results[1].compressed == ""
    assert results[2].compressed == "  "
