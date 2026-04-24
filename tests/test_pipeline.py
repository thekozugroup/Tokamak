"""End-to-end smoke test against the bundled sample."""

import json
from pathlib import Path

from tokamak import pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "examples" / "sample_trace.jsonl"


def test_e2e_rules_mode(tmp_path):
    out = tmp_path / "curated"
    stats = pipeline.run(
        input_dir=None,
        input_file=SAMPLE,
        output_dir=out,
        compress_mode="rules",
        caveman_level="lite",
        anonymize_level="standard",
        max_similarity=0.99,
    )
    assert stats.traces_in == 3
    assert stats.spans >= 3
    assert stats.compressed_tokens < stats.original_tokens

    # Outputs
    for fname in ["data.jsonl", "sharegpt.jsonl", "unsloth.jsonl",
                  "dataset_card.md", "compression_report.md"]:
        assert (out / fname).exists(), f"missing {fname}"

    # Each line of data.jsonl is valid JSON with messages.
    for line in (out / "data.jsonl").read_text().splitlines():
        rec = json.loads(line)
        assert "messages" in rec
        assert "metadata" in rec
        assert "quality_score" in rec["metadata"]

    # Prompts shipped.
    assert (out / "prompts" / "caveman_lite.md").exists()
    assert (out / "prompts" / "caveman_full.md").exists()


def test_noop_mode_preserves_token_count(tmp_path):
    out = tmp_path / "noop"
    stats = pipeline.run(
        input_dir=None,
        input_file=SAMPLE,
        output_dir=out,
        compress_mode="noop",
        caveman_level="lite",
        anonymize_level="standard",
        max_similarity=0.99,
    )
    assert stats.original_tokens == stats.compressed_tokens
