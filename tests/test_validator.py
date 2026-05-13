"""Validator tests — covers JSON parsing, mirrored worst-judge rule, and
graceful degradation when the model output is malformed."""

from __future__ import annotations

from unittest.mock import patch

from tokamak.processor import SpanInput
from tokamak.validator import Validator, _parse_score


SAMPLE = SpanInput(
    problem="Hi",
    answer="Hello",
    reasoning="original reasoning trace",
)


def test_parse_score_clamps_to_unit_interval():
    raw = (
        '{"logical_fidelity": 1.4, "content_fidelity": -0.2, '
        '"safety": 0.7, "signal": 2.0, "notes": "wild"}'
    )
    r = _parse_score(raw)
    assert r.logical_fidelity == 1.0
    assert r.content_fidelity == 0.0
    assert r.safety == 0.7
    assert r.signal == 1.0
    assert r.notes == "wild"


def test_parse_score_defaults_to_worst_axis_when_signal_missing():
    raw = '{"logical_fidelity": 0.9, "content_fidelity": 0.8, "safety": 0.4}'
    r = _parse_score(raw)
    assert r.signal == 0.4


def test_parse_score_handles_garbage_output():
    r = _parse_score("not json")
    assert r.signal == 0.0
    assert r.notes == "parse_error"


def test_single_validator_returns_score():
    raw = (
        '{"logical_fidelity": 0.9, "content_fidelity": 0.95, '
        '"safety": 0.85, "signal": 0.85, "notes": "looks fine"}'
    )

    def fake_chat(self, system, user):
        return raw

    with patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        v = Validator(mirrors=1)
        r = v.validate(SAMPLE, "processed text", mode="compress")

    assert r.signal == 0.85
    assert r.notes == "looks fine"


def test_mirrored_validators_use_worst_judge_rule():
    """3 judges, signals 0.9 / 0.5 / 0.7 — final signal must be 0.5 (worst)."""
    responses = iter([
        '{"logical_fidelity": 1.0, "content_fidelity": 1.0, "safety": 0.9, "signal": 0.9, "notes": "good"}',
        '{"logical_fidelity": 0.6, "content_fidelity": 0.7, "safety": 0.5, "signal": 0.5, "notes": "weak"}',
        '{"logical_fidelity": 0.8, "content_fidelity": 0.9, "safety": 0.7, "signal": 0.7, "notes": "ok"}',
    ])
    lock = __import__("threading").Lock()

    def fake_chat(self, system, user):
        with lock:
            return next(responses)

    with patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        v = Validator(mirrors=3)
        r = v.validate(SAMPLE, "processed text", mode="compress")

    assert r.signal == 0.5
    assert sorted(r.per_judge) == [0.5, 0.7, 0.9]


def test_validator_handles_call_exception_gracefully():
    def boom_chat(self, system, user):
        raise RuntimeError("endpoint down")

    with patch("tokamak.llm_client.LLMClient.chat", new=boom_chat):
        v = Validator(mirrors=1)
        r = v.validate(SAMPLE, "processed", mode="compress")

    assert r.signal == 0.0
    assert "call_error" in r.notes
