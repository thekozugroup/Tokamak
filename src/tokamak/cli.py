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
            "Process the reasoning channel of agent traces — compress (terse "
            "rewrite), invert (skeleton -> fuller trace), or both — then curate "
            "the result into Axolotl / ShareGPT / Unsloth training data with a "
            "signal score per row."
        ),
    )
    p.add_argument("--input-dir", type=Path, help="Directory with raw .jsonl traces")
    p.add_argument("--input-file", type=Path, help="Single .jsonl trace file")
    p.add_argument(
        "--output-dir", type=Path, default=Path("./curated"),
        help="Output directory (default: ./curated)",
    )

    # --- new, primary flags ---
    p.add_argument(
        "--mode", "--process-mode", dest="mode",
        choices=["rules", "compress", "invert", "both", "noop"],
        default=None,
        help=(
            "Processing mode. rules = free regex; compress = terse LLM rewrite; "
            "invert = expand a compressed skeleton; both = compress then invert; "
            "noop = passthrough. Default: rules."
        ),
    )
    p.add_argument(
        "--level", choices=["lite", "full"], default=None,
        help="Compression intensity. lite (default) keeps grammar; full drops articles.",
    )
    p.add_argument(
        "--seqs", type=int, default=None,
        help=(
            "Concurrent agents per stage (compress / invert / validate). "
            "Size to the endpoint's scheduler width (e.g. vLLM --max-num-seqs). "
            "Default 1 (sequential)."
        ),
    )
    p.add_argument(
        "--qaqc", action="store_true",
        help="Enable mirrored validation agent; writes a `signal` (0..1) score per row.",
    )
    p.add_argument(
        "--qaqc-mirrors", type=int, default=1,
        help=(
            "Number of mirrored validators per span (worst-judge rule). "
            "Each judge runs on its own short-lived agent scoped to one span. "
            "Default 1."
        ),
    )

    # --- legacy / back-compat flags ---
    p.add_argument(
        "--compress-mode", choices=["rules", "llm", "noop"], default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--caveman-level", choices=["lite", "full"], default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--llm-concurrency", type=int, default=None,
        help=argparse.SUPPRESS,
    )

    p.add_argument(
        "--model", default=None,
        help="LLM model id (default: env TOKAMAK_MODEL or claude-sonnet-4-5)",
    )

    p.add_argument("--anonymize-level", choices=["standard", "strict"], default="strict")
    p.add_argument(
        "--max-similarity", type=float, default=0.92,
        help="Dedup threshold (Jaccard on 5-shingles); 1.0 disables",
    )

    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--repo-id", default="", help="HF dataset repo id (with --push-to-hub)")
    p.add_argument("--public", action="store_true", help="Make HF repo public")

    p.add_argument(
        "--print-prompt", choices=["lite", "full", "invert", "validate"],
        help="Print a system prompt and exit",
    )
    return p


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.compress_mode == "llm":
        return "compress"
    if args.compress_mode:
        return args.compress_mode
    return "rules"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.print_prompt:
        if args.print_prompt == "invert":
            print(prompts.invert_prompt())
        elif args.print_prompt == "validate":
            print(prompts.validate_prompt())
        else:
            print(prompts.compress_prompt(args.print_prompt))
        return

    if not args.input_dir and not args.input_file:
        print("error: --input-dir or --input-file is required", file=sys.stderr)
        sys.exit(2)

    mode = _resolve_mode(args)
    level = args.level or args.caveman_level or "lite"
    seqs = args.seqs if args.seqs is not None else (args.llm_concurrency or 1)

    stats = pipeline.run(
        input_dir=args.input_dir,
        input_file=args.input_file,
        output_dir=args.output_dir,
        process_mode=mode,
        level=level,
        anonymize_level=args.anonymize_level,
        max_similarity=args.max_similarity,
        model=args.model,
        seqs=seqs,
        qaqc=args.qaqc,
        qaqc_mirrors=max(1, args.qaqc_mirrors),
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
    if args.qaqc:
        print(f"  avg signal       : {stats.avg_signal:.3f}")
    print(f"  output dir       : {args.output_dir.resolve()}")

    if args.push_to_hub:
        if not args.repo_id:
            print("error: --repo-id required with --push-to-hub", file=sys.stderr)
            sys.exit(2)
        pipeline.push_to_hub(args.output_dir, args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
