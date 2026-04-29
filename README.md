Tokamak rewrites the internal reasoning channels of agent traces in caveman-lite style — same logical steps, roughly half the tokens — then runs a full curation pipeline to produce training-ready SFT data. Token savings land exactly where reasoning datasets hurt most: the `<thinking>` block.

Reasoning traces dominate the cost of building "thinking" datasets. A single multi-step trajectory can run 4–10× the tokens of the final answer. Tokamak compresses only that channel, leaves user turns, tool calls, code, and final answers byte-identical, and emits clean JSONL ready for full fine-tuning frameworks. Same training-token budget covers far more reasoning steps.

## Screenshots

![Tokamak — toroidal plasma confinement, the namesake metaphor for compressed reasoning](./docs/screenshot.png)

## How it works

A reasoning span is any `<thinking>`, `<reasoning>`, `<thought>`, `<analyze>`, or `<scratchpad>` block, plus OpenAI o1-style `reasoning` fields and Anthropic `{type: "thinking"}` content blocks. Tokamak walks every trace, rewrites each span in place, and protects code fences, file paths, URLs, quoted strings, error messages, and tool-call JSON byte-for-byte.

Compression runs in one of three modes. Rules mode applies a regex pass that strips filler, hedging, and pleasantries while masking protected spans — fast and free, typically 10–25% reduction. LLM mode calls a model (Anthropic API, `claude` CLI, or any OpenAI-compatible endpoint such as a local vLLM server) with a caveman-lite system prompt that explicitly preserves every reasoning step — slower, typically 50–75% reduction. Concurrent batching saturates self-hosted scheduler width across the whole dataset. Noop mode is a passthrough for baselines.

After compression each trace flows through the curation pipeline: regex + entropy-based PII anonymization, six-dimension quality scoring (reported, never used to filter), five-factor error classification, lexical deduplication that ignores boilerplate tool-call JSON, and simultaneous export to Axolotl, ShareGPT, and Unsloth formats. A dataset card is generated with compression statistics, and an optional `--push-to-hub` flag ships the result to the Hugging Face Hub.

## Stack

- Python 3.10+, zero required runtime dependencies
- Optional: Anthropic SDK (LLM compression), scikit-learn (dedup), sentence-transformers (semantic dedup), huggingface_hub (publish)
- vLLM-compatible: point `--compress-mode llm` at any OpenAI-spec endpoint
- Pytest regression suite

## Status

Active
