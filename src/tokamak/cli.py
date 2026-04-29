"""Tokamak CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import pipeline, prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokamak",
        description=(
            "Caveman-compress reasoning channels in agent traces and curate "
            "the result into Axolotl/ShareGPT/Unsloth training data."
        ),
    )
    p.add_argument("--input-dir", type=Path, help="Directory with raw .jsonl traces")
    p.add_argument("--input-file", type=Path, help="Single .jsonl trace file")
    p.add_argument("--output-dir", type=Path, default=Path("./curated"),
                   help="Output directory (default: ./curated)")

    p.add_argument("--compress-mode", choices=["rules", "llm", "noop"], default="rules",
                   help="rules = free regex; llm = call Claude; noop = skip compression")
    p.add_argument("--caveman-level", choices=["lite", "full"], default="lite",
                   help="lite (default) keeps grammar; full drops articles too")
    p.add_argument("--model", default=None,
                   help="Anthropic model for --compress-mode llm (default: env TOKAMAK_MODEL or claude-sonnet-4-5)")
    p.add_argument("--llm-concurrency", type=int, default=1,
                   help="Concurrent LLM calls for --compress-mode llm. Size to match "
                        "the endpoint's scheduler width (e.g., vLLM --max-num-seqs). "
                        "Default 1 (sequential).")

    p.add_argument("--anonymize-level", choices=["standard", "strict"], default="strict")
    p.add_argument("--max-similarity", type=float, default=0.92,
                   help="Dedup threshold (Jaccard on 5-shingles); 1.0 disables")

    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--repo-id", default="", help="HF dataset repo id (with --push-to-hub)")
    p.add_argument("--public", action="store_true", help="Make HF repo public")

    p.add_argument("--print-prompt", choices=["lite", "full"],
                   help="Print the caveman system prompt and exit")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.print_prompt:
        print(prompts.system_prompt(args.print_prompt))
        return

    if not args.input_dir and not args.input_file:
        print("error: --input-dir or --input-file is required", file=sys.stderr)
        sys.exit(2)

    stats = pipeline.run(
        input_dir=args.input_dir,
        input_file=args.input_file,
        output_dir=args.output_dir,
        compress_mode=args.compress_mode,
        caveman_level=args.caveman_level,
        anonymize_level=args.anonymize_level,
        max_similarity=args.max_similarity,
        model=args.model,
        llm_concurrency=args.llm_concurrency,
    )

    saved = 0.0
    if stats.original_tokens:
        saved = 100.0 * (1 - stats.compressed_tokens / stats.original_tokens)
    print()
    print(f"  traces in        : {stats.traces_in}")
    print(f"  traces out       : {stats.traces_out}")
    print(f"  reasoning spans  : {stats.spans}")
    print(f"  tokens before    : {stats.original_tokens:,}")
    print(f"  tokens after     : {stats.compressed_tokens:,}")
    print(f"  tokens saved     : {saved:.1f}%")
    print(f"  PII redactions   : {stats.redactions}")
    print(f"  output dir       : {args.output_dir.resolve()}")

    if args.push_to_hub:
        if not args.repo_id:
            print("error: --repo-id required with --push-to-hub", file=sys.stderr)
            sys.exit(2)
        pipeline.push_to_hub(args.output_dir, args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
