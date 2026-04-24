Tokamak rewrites the internal reasoning channels of agent traces in caveman-lite style — same logical steps, roughly half the tokens — then runs a full TALOS-style curation pipeline to produce training-ready SFT data. It merges two existing projects so the token savings land exactly where reasoning datasets hurt most: the `<thinking>` block.

Reasoning traces dominate the cost of building "thinking" datasets. A single multi-step trajectory can run 4–10× the tokens of the final answer. Caveman proved you can compress LLM *output* ~75% with no accuracy loss. TALOS proved you can curate raw agent sessions into clean, scored, deduplicated training data. Tokamak welds them so the same training-token budget covers far more reasoning steps.

## Screenshots

_Screenshot pending. Terminal capture of a real run will land at `docs/screenshot.png`._

## How it works

A reasoning span is any `<thinking>`, `<reasoning>`, `<thought>`, `<analyze>`, or `<scratchpad>` block, plus OpenAI o1-style `reasoning` fields and Anthropic `{type: "thinking"}` content blocks. Tokamak walks every trace, rewrites each span in place, and leaves user turns, tool calls, tool results, code fences, file paths, and final assistant answers byte-identical to the source.

Compression runs in one of three modes. Rules mode applies a regex pass that strips filler, hedging, and pleasantries while protecting code, paths, quoted strings, and errors — fast and free, typically 10–25% reduction. LLM mode calls Claude with a caveman-lite system prompt that explicitly instructs it to preserve every reasoning step — slower, typically 50–75% reduction. Noop mode is a passthrough for baselines.

After compression each trace flows through the TALOS pipeline: regex + entropy-based PII anonymization, six-dimension quality scoring (reported, never used to filter), five-factor error classification, lexical deduplication that ignores boilerplate tool-call JSON, and simultaneous export to Axolotl, ShareGPT, and Unsloth formats. A dataset card is generated with compression statistics, and an optional `--push-to-hub` flag ships the result to the Hugging Face Hub.

## Stack

- Python 3.10+, zero required runtime dependencies
- Optional: Anthropic SDK (LLM compression), scikit-learn (dedup), sentence-transformers (semantic dedup), huggingface_hub (publish)
- Pytest for the regression suite
- Derived from [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) and [DJLougen/TALOS-trace-curator](https://github.com/DJLougen/TALOS-trace-curator), both MIT

## Status

Active
