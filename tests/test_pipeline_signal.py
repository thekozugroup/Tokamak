"""Pipeline tests that exercise the new signal column + processing modes.

These keep the network out: LLM-mode calls are intercepted at the
LLMClient.batch_chat / LLMClient.chat boundary so we don't need an
Anthropic key, OpenAI endpoint, or `claude` CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from tokamak import pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "examples" / "sample_trace.jsonl"


# ---------------------------------------------------------------------------
# Mode toggles
# ---------------------------------------------------------------------------

def test_rules_mode_writes_new_prompt_files(tmp_path):
    out = tmp_path / "rules"
    pipeline.run(
        input_dir=None, input_file=SAMPLE, output_dir=out,
        compress_mode="rules", caveman_level="lite",
        anonymize_level="standard", max_similarity=0.99,
    )
    # New prompt files for each agent.
    for fname in ("compress_lite.md", "compress_full.md", "invert.md", "validate.md"):
        assert (out / "prompts" / fname).exists(), f"missing {fname}"


def test_compress_mode_via_new_flag_works(tmp_path):
    """`process_mode='compress'` runs the LLM compressor. We mock the client."""
    def fake_batch(self, items, max_workers=1):
        return [f"COMPRESSED({len(it['user'])})" for it in items]

    out = tmp_path / "compress"
    with patch("tokamak.llm_client.LLMClient.batch_chat", new=fake_batch):
        stats = pipeline.run(
            input_dir=None, input_file=SAMPLE, output_dir=out,
            process_mode="compress", level="lite",
            anonymize_level="standard", max_similarity=0.99,
            seqs=2,
        )
    assert stats.traces_in == 3
    assert stats.spans >= 3


def test_both_mode_runs_two_stages(tmp_path):
    """`process_mode='both'` chains compress -> invert. Two stages == two
    batch calls per total span."""
    calls: list = []

    def fake_batch(self, items, max_workers=1):
        calls.append(len(items))
        return ["OUT" for _ in items]

    out = tmp_path / "both"
    with patch("tokamak.llm_client.LLMClient.batch_chat", new=fake_batch):
        pipeline.run(
            input_dir=None, input_file=SAMPLE, output_dir=out,
            process_mode="both", level="lite",
            anonymize_level="standard", max_similarity=0.99,
            seqs=4,
        )
    # 2 stages * (>=3) spans → 2 calls, each containing same count of spans.
    assert len(calls) == 2
    assert calls[0] == calls[1]


# ---------------------------------------------------------------------------
# QAQC signal column
# ---------------------------------------------------------------------------

def test_qaqc_writes_signal_column_into_rows(tmp_path):
    """With --qaqc, each row should carry a `signal` field in metadata."""
    def fake_batch(self, items, max_workers=1):
        return ["TERSE_OUT" for _ in items]

    def fake_chat(self, system, user):
        # Validator path — return a clean JSON object.
        return ('{"logical_fidelity": 0.9, "content_fidelity": 0.8, '
                '"safety": 0.75, "signal": 0.75, "notes": "ok"}')

    out = tmp_path / "qaqc"
    with patch("tokamak.llm_client.LLMClient.batch_chat", new=fake_batch), \
         patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        stats = pipeline.run(
            input_dir=None, input_file=SAMPLE, output_dir=out,
            process_mode="compress", level="lite",
            anonymize_level="standard", max_similarity=0.99,
            seqs=2, qaqc=True, qaqc_mirrors=1,
        )

    assert stats.avg_signal == 0.75

    # Every row should have a signal in metadata.
    rows = [json.loads(line) for line in (out / "data.jsonl").read_text().splitlines()]
    for r in rows:
        assert "metadata" in r
        assert "signal" in r["metadata"]
        assert 0.0 <= r["metadata"]["signal"] <= 1.0


def test_qaqc_mirrored_worst_judge_dominates(tmp_path):
    """Two mirrors with different opinions — the worst signal must surface."""
    judges = iter([
        # Pattern: every span gets 2 judges. We'll return alternating worse/better.
        '{"logical_fidelity": 1.0, "content_fidelity": 1.0, "safety": 1.0, "signal": 1.0, "notes": "ok"}',
        '{"logical_fidelity": 0.3, "content_fidelity": 0.4, "safety": 0.2, "signal": 0.2, "notes": "bad"}',
    ] * 64)  # plenty of buffer
    import threading
    lock = threading.Lock()

    def fake_chat(self, system, user):
        with lock:
            return next(judges)

    def fake_batch(self, items, max_workers=1):
        return ["OUT" for _ in items]

    out = tmp_path / "mirrors"
    with patch("tokamak.llm_client.LLMClient.batch_chat", new=fake_batch), \
         patch("tokamak.llm_client.LLMClient.chat", new=fake_chat):
        stats = pipeline.run(
            input_dir=None, input_file=SAMPLE, output_dir=out,
            process_mode="compress", level="lite",
            anonymize_level="standard", max_similarity=0.99,
            seqs=1, qaqc=True, qaqc_mirrors=2,
        )

    # The worst-judge rule means every row's signal lands at 0.2.
    assert abs(stats.avg_signal - 0.2) < 0.01


def test_qaqc_disabled_leaves_signal_unset(tmp_path):
    """Without --qaqc, rows should not carry a positive signal."""
    out = tmp_path / "nosignal"
    pipeline.run(
        input_dir=None, input_file=SAMPLE, output_dir=out,
        compress_mode="rules", caveman_level="lite",
        anonymize_level="standard", max_similarity=0.99,
    )
    # avg_signal must be 0 (no readings collected).
    rows = [json.loads(line) for line in (out / "data.jsonl").read_text().splitlines()]
    for r in rows:
        # Either signal absent or marked unknown via -1.
        sig = r.get("metadata", {}).get("signal")
        assert sig is None or sig < 0
