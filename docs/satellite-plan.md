# Kōzu Satellite — Build Plan v0.1

**Owner:** thekozugroup · **Status:** draft · **Target tape-out:** Satellite v0.1 by week 10 (ending 2026-07-05)

## TL;DR

- **Primary base:** `Qwen/Qwen3-1.7B` (Apache-2.0, 32k ctx, dual-mode reasoning tokenizer). Backup: `HuggingFaceTB/SmolLM2-1.7B` (Apache-2.0, cleaner pretrain mix).
- **Frankenmerge target:** recommended **NO** frankenmerge at the Satellite tier. DUS is a Planetary-class move. For Satellite we ship two SKUs — `Satellite-1.7B` (Qwen3 straight SFT) and `Satellite-3B-DUS` (Qwen3-1.7B → ~3B via mergekit passthrough + heal) only if the 1.7B eval baseline undershoots targets. Details in §4.
- **DGX Spark GPU-hours budgeted:** ~420 hours total across data gen, SFT stages, heal, and eval (math in §3/§5). Fits in ~3 wall-clock weeks of 20h/day utilization, leaves headroom.
- **Single biggest risk:** Tokamak's LLM compression mode stripping intermediate CoT steps the 1.7B student needs to imitate — a larger teacher can skip steps a small student cannot. Mitigation: rules-only compression on reasoning SFT splits, LLM compression only on verbose/dialog splits. See §9.

Hardware assumption throughout: **one DGX Spark, GB10 Grace Blackwell, 128 GB unified LPDDR5X, ~273 GB/s memory bandwidth, ~100 BF16 TFLOPS measured (MAMF), ~208 FP8 TFLOPS, ~1 PFLOP FP4 sparse** ([NVIDIA dev blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/), [LMSYS review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)). Memory bandwidth — not FLOPS — is the binding constraint for sub-10B training; plan accordingly.

---

## 1. Base model selection

Criteria weighting: license (hard gate), tokenizer efficiency on English + code + multilingual, reasoning aptitude out-of-the-box (GSM8K, MMLU, IFEval), ecosystem maturity for GGUF/MLX/ONNX, and availability of a true base (non-instruct) checkpoint we can SFT cleanly.

| Model | Params | License | MMLU (5-shot) | Tokenizer vocab | GGUF/MLX | Base ckpt? | Verdict |
|---|---|---|---|---|---|---|---|
| **Qwen/Qwen3-1.7B** | 1.7B | Apache-2.0 | ~59 (reported) | 151k BBPE | mature | yes | **Primary** |
| Qwen/Qwen3-0.6B | 0.6B | Apache-2.0 | ~48 | 151k BBPE | mature | yes | too small, keep as distill-student stretch goal |
| HuggingFaceTB/SmolLM2-1.7B | 1.7B | Apache-2.0 | ~50 | 49k | mature | yes | **Backup** — cleaner pretrain (FineWeb-Edu/DCLM/Stack, 11T tokens) |
| meta-llama/Llama-3.2-3B | 3.2B | Llama Community License (commercial <700M MAU) | 58 base | 128k | very mature | yes | rejected — license carveout + "Built with Llama" attribution |
| meta-llama/Llama-3.2-1B | 1.2B | Llama Community License | ~32 (MMLU-Pro 0.226) | 128k | very mature | yes | rejected — same license friction, weaker than Qwen3-1.7B |
| google/gemma-3-1b | 1B | Gemma Terms of Use | not directly published | 256k | mature | yes | rejected — Gemma ToU is more restrictive than Apache; good QAT INT4 checkpoints but not worth license tail-risk |
| google/gemma-3-270m | 270M | Gemma ToU | n/a (IFEval 51.2) | 256k | mature | yes | too small for reasoning; interesting for a future Nano class |
| microsoft/Phi-4-mini | 3.8B | MIT | 73 | 200k | mature | yes | strong candidate but 3.8B blows the Satellite size envelope; reserve for Planetary base |
| TinyLlama-1.1B | 1.1B | Apache-2.0 | ~26 | 32k Llama-1 | mature | yes | rejected — weak pretrain |

**Decision: Qwen3-1.7B primary, SmolLM2-1.7B backup.** Qwen3 wins on raw reasoning (dual-mode thinking was native during pretraining, so the student already knows the `<think>` convention Tokamak targets) and Apache-2.0 is unambiguous. SmolLM2 is the backup specifically because its pretraining corpus is fully documented (FineWeb-Edu / DCLM / Stack), which derisks licensing of the model we ship even if someone later litigates Qwen3's pretrain mix.

Sources: [Qwen3-0.6B card](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3 tech report](https://arxiv.org/pdf/2505.09388), [SmolLM2-1.7B card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B), [Llama 3.2 3B evals](https://huggingface.co/datasets/meta-llama/Llama-3.2-3B-evals), [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/).

---

## 2. Data pipeline (Hadron → Tokamak → SFT)

Target mix for an **agentic edge reasoner** — not a general chatbot. Satellite's job is tool-using, on-device reasoning at ≤3B params.

### 2.1 Seed prompts fed to Hadron

Domain mix (percentage = share of raw generated rows, not tokens):

| Domain | Share | Seed source |
|---|---|---|
| Tool-use / function calling | 25% | xLAM-function-calling-60k subset + synthetic tool schemas scraped from MCP registry |
| Code reasoning (bug find, patch, explain) | 20% | OSS GitHub issues + Stack Exchange + LiveCodeBench-style prompts |
| Math / symbolic | 15% | GSM8K + MATH train + synthetic word problems |
| Multi-hop QA | 15% | HotpotQA + MuSiQue paraphrased seeds |
| Agent planning (multi-step, self-correction) | 15% | AgentInstruct-style seeds + synthetic "plan → critique → revise" templates |
| Instruction following (format-constrained) | 10% | IFEval-style constraints, synthetic |

### 2.2 Hadron configuration

- **Teacher tier A (reasoning ground truth):** Anthropic Claude via API — 30% of generations, used for highest-difficulty seeds (agent planning, multi-hop QA). This is where the `<think>` blocks are richest.
- **Teacher tier B (bulk volume):** local **Qwen3-32B-Instruct** running on DGX Spark via vLLM or llama.cpp (Q8 fits in ~34 GB, leaves headroom on 128 GB unified). Produces 70% of generations.
- **distilabel pipelines used:** `TextGeneration` with CoT prompt, `UltraFeedback`-style critique pass for reward-model-ready pairs (ignored at SFT stage, kept for future DPO), `SelfInstruct` for seed expansion.
- **License hygiene:** Claude outputs can be used per Anthropic's commercial terms for training a non-competing model; we tag every Claude-sourced row with `teacher=claude` in metadata so we can rebuild the dataset without it if terms change. Qwen3 outputs are Apache-2.0 downstream.

Refs: [distilabel synthetic reasoning examples](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled), [Qwen3 on DGX Spark](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639).

### 2.3 Tokamak consumption

Hadron exports ShareGPT-style JSONL. Tokamak ingests with a **format shim** (`tokamak.readers.sharegpt`) and rewrites the `<thinking>` spans per the README spec.

**Per-slice compression policy:**

| Slice | Mode | Expected reduction |
|---|---|---|
| Tool-use / function calling | `rules` only | 10–20% |
| Code reasoning | `rules` only | 10–25% (protects code fences by design) |
| Math / symbolic | `rules` only | 10–20% (protects equations) |
| Multi-hop QA | `llm` (Claude caveman-lite) | 50–70% |
| Agent planning | `llm` | 55–75% |
| Instruction following | `llm` | 40–60% |

Rationale: rules mode never drops a logical step, so we use it on slices where every step matters to the student. LLM mode is only applied where the raw traces are verbose-by-default (narrative QA, plan/critique dialogues).

TALOS pipeline post-compression runs as-is: PII anonymization, 6-dim quality scoring (logged, not filter-gated at this stage), 5-factor error classification, lexical dedup across all Hadron batches using the boilerplate-ignore option described in the Tokamak README. Export to **Axolotl JSONL** (primary training format) and **Unsloth** (backup path if we switch trainers mid-plan).

### 2.4 Dataset sizing

| Stage | Raw rows (pre-Tokamak) | Post-curation rows | Tokens (post-compression, BF16 tokenizer) |
|---|---|---|---|
| Stage-1 SFT | 1.2M | ~700k (dedup + quality floor 3/6) | ~450M tokens |
| Heal corpus | n/a (pretraining-style) | n/a | 5–10B tokens (see §5) |
| Stage-2 SFT | 600k (refresh + hard-negative mining on Stage-1 failures) | ~350k | ~220M tokens |

Hadron generation cost estimate: at ~80 tok/s for Qwen3-32B Q8 on DGX Spark and ~2k output tokens avg, the Qwen tier alone is ~800k rows × 2k / 80 = ~5.5M seconds = ~1,500 GPU-hours. **This blows the budget** — so we (a) compress seed prompts aggressively via Hadron's autoreason shortcuts, (b) only generate the stretch volume if Stage-1 evals demand it, and (c) push half the bulk volume to batched Claude API calls (network-bound, not DGX-bound). Realistic Hadron DGX load: **~200 GPU-hours**, two passes of ~300k rows each, rest from Claude. Revisit §10 timeline.

### 2.5 Dedup

Two-pass:
1. Within each Hadron batch: MinHash LSH on prompt+response (threshold 0.8) — runs inside Hadron.
2. Across batches, pre-SFT: Tokamak's lexical dedup on reasoning-block-stripped text, plus a content-hash of the final answer. This catches the common failure mode where the teacher produces two traces with different thinking but the same conclusion.

---

## 3. Stage-1 SFT

**Trainer:** Axolotl (primary) with DeepSpeed ZeRO-2. Unsloth as backup if Axolotl hits a memory wall — Unsloth on a 1.7B LoRA needs ~5 GB VRAM per [Unsloth docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune), trivial on DGX Spark.

**Full-FT vs QLoRA call:**

| Model | Full-FT VRAM (seq 4096, bs 1, AdamW, BF16 + FP32 master) | QLoRA VRAM | Decision |
|---|---|---|---|
| Qwen3-1.7B | ~28 GB params+grads+optimizer + ~12 GB activations = ~40 GB | ~6 GB | **Full-FT**. 40 GB fits comfortably in 128 GB unified. Full-FT > QLoRA quality at this size; no reason to QLoRA when VRAM is a non-issue. |
| Qwen3-3B (if we promote to 3B backup) | ~70 GB | ~12 GB | Full-FT still fits (70 GB of 128 GB). Full-FT. |
| 7B+ (Planetary, out of scope) | >140 GB | ~24 GB | QLoRA would be required on Spark |

VRAM math for full-FT 1.7B at seq 4096, per-GPU bs 1, grad accum 16:
- Params (BF16): 1.7B × 2 B = 3.4 GB
- Grads (BF16): 3.4 GB
- Optimizer AdamW (FP32 m + v + master): 1.7B × 12 B = 20.4 GB
- Activations (rough, grad checkpointing on, 32 layers × 2k hidden × 4096 seq × 2 B): ~8 GB
- Total: ~35 GB. Spark has 128 GB unified. Comfortable.

**Hyperparams (Stage-1, Qwen3-1.7B):**
```
base_model: Qwen/Qwen3-1.7B
sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true
micro_batch_size: 4
gradient_accumulation_steps: 16   # effective batch 64
num_epochs: 3
learning_rate: 5.0e-6             # conservative for base; raise to 1e-5 if loss plateaus early
lr_scheduler: cosine
warmup_ratio: 0.03
optimizer: adamw_torch_fused
weight_decay: 0.01
bf16: true
tf32: true
gradient_checkpointing: true
flash_attention: true
train_on_inputs: false            # mask user turns, train on assistant + <think> only
```

**Wall-clock estimate:** 450M tokens × 3 epochs = 1.35B train tokens. DGX Spark at ~100 BF16 TFLOPS MAMF, assume ~60% MFU at this scale → ~60 TFLOPS effective. For a 1.7B dense model, training FLOPs ≈ 6 × N × D = 6 × 1.7e9 × 1.35e9 = 1.38e19 FLOPs. 1.38e19 / 6e13 = ~230,000 seconds = ~64 hours. **Budget: 80 GPU-hours** (including eval during training + restart overhead).

Note: memory bandwidth may bottleneck this below the FLOPS ceiling. If observed MFU is <40%, switch to FP8 training via Transformer Engine on GB10 (doubles throughput on Blackwell per [NVIDIA blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)).

---

## 4. Frankenmerge — explicit recommendation: SKIP for Satellite v0.1

I evaluated DUS (SOLAR-10.7B recipe, [arXiv 2312.15166](https://arxiv.org/abs/2312.15166)) for scaling Qwen3-1.7B → ~3B and Qwen3-3B → ~7B.

**The SOLAR recipe:** take a 32-layer base, duplicate layers to 48 (keep layers 0..23 from copy A, 8..31 from copy B, concatenate), yielding a model with n=32 → s=48 via m=8 overlap-trim. Implemented in mergekit with `merge_method: passthrough`.

**Why I recommend skipping at Satellite tier:**

1. **Purpose mismatch.** Satellite's charter is *edge*. A 2.5B frankenmerge that needs ~5–10B tokens of heal to stop babbling costs us the same GPU-hours as a *better* 1.7B SFT run, ships a larger artifact, and still loses to Phi-4-mini on reasoning benches.
2. **Heal cost is non-trivial.** SOLAR's paper doesn't publish exact heal-token budget but DUS recipes in the wild report 2–10B tokens to close the perplexity gap. At 1.35B training tokens per SFT epoch, that's 2–8× our Stage-1 cost just to recover baseline.
3. **Unhealable-damage risk.** Layer duplication with overlap can create mode-collapsed attention heads if the cut points intersect critical induction circuits. Qwen3's layer norm and RoPE configuration is not the same as Mistral's, and no one has published a well-healed Qwen3 DUS recipe as of this writing. That's research, not engineering.

**What we do instead for Satellite:**

- Ship `Satellite-1.7B-v0.1` from §3 Stage-1 SFT alone if it hits the eval bar in §7.
- If it misses by >3 pts on aggregate, **switch base to Qwen3-3B** (same Apache-2.0 license, ~5.5 GB BF16, still edge-viable at Q4) and redo Stage-1. Full-FT on 3B still fits Spark (§3 table).
- Reserve DUS for Planetary class (6B → 10.7B), where the size budget justifies the heal cost.

**If we defy this recommendation** — recipe would be:

```yaml
# mergekit DUS: Qwen3-1.7B (28 layers) → ~2.5B (42 layers)
merge_method: passthrough
dtype: bfloat16
slices:
  - sources:
      - model: kozu/satellite-1.7b-sft-v0.1
        layer_range: [0, 21]
  - sources:
      - model: kozu/satellite-1.7b-sft-v0.1
        layer_range: [7, 28]
```

Run with `mergekit-yaml config.yml ./satellite-2.5b-dus --allow-crimes`. Then heal per §5.

Refs: [SOLAR 10.7B paper](https://arxiv.org/html/2312.15166v2), [mergekit DUS how-to](https://ssawant.github.io/posts/mergekit/Mearge_LLMs_with_mergekit.html), [iDUS reference implementation](https://github.com/gauss5930/iDUS).

---

## 5. Healing (only if we run §4)

Post-merge continued pretraining to close the perplexity gap from duplicated layers.

**Dataset:** `HuggingFaceFW/fineweb-edu` — 10% sample (~130B tokens exists; we need ~5B). Mix with 20% code from `bigcode/the-stack-v2-dedup` (sampled) so we don't catastrophically forget code reasoning. Rationale per [FineWeb paper](https://arxiv.org/html/2406.17557v1): FineWeb-Edu matches generalist pretrains at 10× fewer tokens on reasoning benches.

**Token budget:** 5B heal tokens as default. If PPL on held-out FineWeb-Edu is still >1.15× base-model PPL after 3B, push to 8B. Hard cap 10B.

**Sequence length:** 4096 (matches Qwen3 native; don't extend here, heal doesn't need long ctx).

**LR schedule:**
- Peak LR: 2e-5 (lower than Stage-1 SFT because we're rehabilitating existing weights, not teaching new behavior)
- Warmup: 100 steps
- Cosine decay to 2e-6 over full heal budget
- Optimizer: AdamW, β=(0.9, 0.95), wd=0.1

**"Healed enough" signals:**
1. Held-out FineWeb-Edu PPL within 10% of pre-merge base model PPL.
2. HellaSwag and ARC-E within 2 absolute points of pre-merge SFT checkpoint. These are cheap and correlate with general competence; if they recover, heal worked.
3. A 200-sample qualitative read-through — heal is done when outputs are fluent, not when loss is minimum. Over-healing reintroduces base-model personality and wipes SFT.

**GPU-hours:** 5B tokens × 6 × 2.5e9 params / 60 TFLOPS = 1.25e18 / 6e13 = ~21,000 s = ~6 hours per epoch. Budget 40 GPU-hours (one pass plus a safety margin for restarts and checkpoint eval).

Healing references the classic DUS finding that the scaled model's performance initially drops below the base and continued pretraining restores it ([SOLAR paper §3.2](https://arxiv.org/html/2312.15166v2)).

---

## 6. Stage-2 SFT

Re-SFT the healed merged model (or the Stage-1 SFT model if we skipped merge) on a **refreshed** Tokamak-curated reasoning set. 350k rows / ~220M tokens. Same Axolotl config as Stage-1 but:
- 2 epochs (not 3) — we're sharpening, not teaching from scratch.
- LR peak 2e-6 (lower; we're not undoing heal).
- Add a 5% DPO-ready preference subset if bench-hacking is desired (out of scope for v0.1).

Why Stage-2 works: heal restores generality but wipes the task-specific priors we put in at Stage-1. Stage-2 is shorter and cheaper than Stage-1 because the model already knows the format, it just needs the heal-caused drift corrected. Skip Stage-2 if we skipped merge — the Stage-1 model is the final.

**GPU-hours:** ~30 hours.

---

## 7. Eval harness

Run in three tiers: cheap (after every SFT epoch), medium (after each stage), expensive (before release).

**Cheap tier (every epoch, ~10 min):**
- HellaSwag (acc, 1k sample subset)
- ARC-Easy (acc, full)
- PIQA (acc, full)

**Medium tier (after each stage, ~1 hr):**
- MMLU (5-shot, full)
- MMLU-Pro (5-shot, 2k subset — full is expensive)
- GSM8K (0-shot CoT, full)
- IFEval (full)
- BBH (full, 3-shot)

**Expensive tier (before release, ~4 hr):**
- GPQA-diamond (0-shot and CoT)
- HumanEval (pass@1, 10 samples, temp 0.2)
- LiveCodeBench lite subset
- Edge latency: tokens/sec on M2 Pro via MLX 4-bit, on Jetson Orin Nano via llama.cpp Q4_K_M, on iPhone 16 Pro via MLC-LLM (if time permits)

**Comparison table template (populate after each stage):**

| Eval | Qwen3-1.7B base | Stage-1 SFT | Post-merge (skip if §4 skipped) | Post-heal | Stage-2 SFT | Satellite-1.7B-v0.1 target |
|---|---|---|---|---|---|---|
| MMLU (5-shot) | 59.0 (reported) | | | | | ≥60 |
| MMLU-Pro | ~25 (estimated, unverified) | | | | | ≥28 |
| GSM8K CoT | ~45 (reported) | | | | | ≥55 |
| IFEval | ~55 | | | | | ≥65 |
| HumanEval pass@1 | ~30 | | | | | ≥35 |
| BBH | ~35 | | | | | ≥40 |
| GPQA-diamond | ~25 | | | | | ≥27 |
| MLX Q4 tok/s (M2 Pro) | ~50 | — | — | — | — | ≥40 |
| Jetson Orin Q4 tok/s | ~20 | — | — | — | — | ≥15 |

Baselines marked "reported" come from Qwen3 tech report / HF card; ones marked "estimated, unverified" are my guess and **must be re-measured before we publish numbers**.

Harness: `lm-evaluation-harness` for MMLU/MMLU-Pro/BBH/ARC/HellaSwag, `lighteval` as cross-check, custom script for IFEval (reference impl from Google), `EleutherAI/gsm8k` harness task. Edge latency via `llama-bench` and `mlx_lm.generate --temp 0 --max-tokens 128`.

---

## 8. Export + quantization

Three targets, one script per target.

**GGUF (llama.cpp):**
```bash
# 1. HF → F16 GGUF
python llama.cpp/convert_hf_to_gguf.py kozu/satellite-1.7b-v0.1 \
  --outfile satellite-1.7b-f16.gguf
# 2. Quantize to Q4_K_M, Q5_K_M, Q8_0
for q in Q4_K_M Q5_K_M Q8_0; do
  ./llama-quantize satellite-1.7b-f16.gguf satellite-1.7b-$q.gguf $q
done
```
Expected on-disk sizes: Q4_K_M ~1.0 GB, Q5_K_M ~1.2 GB, Q8_0 ~1.8 GB.

**MLX (Apple Silicon):**
```bash
mlx_lm.convert --hf-path kozu/satellite-1.7b-v0.1 \
  --mlx-path satellite-1.7b-mlx-4bit -q --q-bits 4 --q-group-size 64
mlx_lm.convert --hf-path kozu/satellite-1.7b-v0.1 \
  --mlx-path satellite-1.7b-mlx-8bit -q --q-bits 8
```
Expected sizes: 4-bit ~1.0 GB, 8-bit ~1.8 GB. MLX reportedly ~15–30% faster than GGUF at same quant on Apple Silicon per [willitrunai benchmark](https://willitrunai.com/blog/qwen-3-5-mlx-apple-silicon-guide).

**ONNX (cross-platform / mobile):**
```bash
optimum-cli export onnx --model kozu/satellite-1.7b-v0.1 \
  --task text-generation-with-past \
  --opset 18 satellite-1.7b-onnx/
python -m onnxruntime.quantization.matmul_4bits_quantizer \
  --input satellite-1.7b-onnx/model.onnx \
  --output satellite-1.7b-onnx-int4/model.onnx \
  --block_size 32
```
Expected size: INT4 ~1.1 GB. ONNX is for Windows/Android deployment.

All three artifacts pushed to `kozu/satellite-1.7b-v0.1-{gguf,mlx,onnx}` repos on HF Hub.

Refs: [llama.cpp convert guide](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html), [mlx-lm README](https://pypi.org/project/mlx-lm/).

---

## 9. Risk register

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Tokamak LLM-mode compression elides intermediate CoT steps that a 1.7B student can't fill in, producing worse GSM8K than base | med | high | Rules-only compression on reasoning slices (§2.3). Ablate by training one epoch with rules-only vs llm-mode on a 50k subset and diffing GSM8K @1000 steps. If llm-mode costs >2 pts GSM8K, fall back to rules-only globally. |
| 2 | Frankenmerge produces unhealable damage | n/a | critical if attempted | Skipped for v0.1 (§4). Revisit at Planetary. |
| 3 | DGX Spark's 273 GB/s memory bandwidth throttles BF16 training below modeled throughput, doubling wall-clock | high | med | Switch to FP8 via Transformer Engine on GB10 (Blackwell supports native FP8 matmul, 208 TFLOPS measured). Fall back to micro-bs=2 with grad-accum=32 if memory-bound. Budget includes 20% slack. |
| 4 | License contamination from Claude-generated data blocks commercial release | low | critical | Tag every Claude row `teacher=claude`. Re-run Stage-1 from Qwen-only subset as a clean-room contingency build. Keep both checkpoints until legal review. |
| 5 | Synthetic-data collapse: 2 teacher models → homogeneous failure modes that the student memorizes | med | med | Teacher diversity (Claude + Qwen3-32B). Inject 15% human-written data (OpenAssistant + oasst2 + HumanEval train) post-Hadron. Quality-floor filter on Tokamak's 6-dim score — drop bottom 10% even though the README says scoring is "reported, not filter". We override for this run and document the deviation. |

---

## 10. Timeline (single engineer, full-time, starting 2026-04-27)

Week counts from **week of 2026-04-27 = Week 1**. Ship target: **Week 10 (week of 2026-06-29)**.

| Week | Work | Checkpoint |
|---|---|---|
| **1** (Apr 27) | Environment: DGX Spark setup, CUDA 12.8, Axolotl, Unsloth, mergekit, vLLM, Tokamak + Hadron installed. Smoke-train Qwen3-1.7B on 10k rows to verify full pipeline. | End of week: successful 1-epoch smoke on 10k Alpaca rows, eval harness runs. |
| **2** (May 4) | Hadron seed curation. Spin up Qwen3-32B vLLM server on Spark. Generate first 200k-row batch (tool-use + code + math slices). | End of week: 200k raw rows, MinHash dedup clean. |
| **3** (May 11) | Hadron batch 2 (QA + planning + IFEval slices), 400k rows. Claude API generation in parallel for high-difficulty subset. Tokamak rewrite + TALOS curation on all 600k. | End of week: 600k curated rows exported to Axolotl format. Dataset card drafted. |
| **4** (May 18) | Hadron batch 3 stretch (200k, optional). Stage-1 SFT kickoff on Qwen3-1.7B. Live-track loss + cheap evals every 2k steps. | End of week: Stage-1 epoch 1 complete, cheap-eval green. |
| **5** (May 25) | Stage-1 SFT epochs 2–3. Medium-tier eval. **Go/no-go decision on §4 frankenmerge.** Default: NO. If aggregate eval lift <3 pts vs base, promote to Qwen3-3B and restart Stage-1 (add 1 week). | Fri: full medium-tier eval table populated. GO/NO-GO filed. |
| **6** (Jun 1) | *If §4 skipped (expected):* Stage-2 SFT (light polish run, 1 epoch, DPO-ready pref data prep). Expensive-tier eval. *If §4 taken:* mergekit passthrough + heal kicks off. | End of week: final training checkpoint. |
| **7** (Jun 8) | Quantization sweep: GGUF Q4/Q5/Q8, MLX 4/8bit, ONNX INT4. Edge latency benchmarks on M-series Mac + Jetson Orin + iPhone (if device available). | End of week: 8 quantized artifacts, latency table complete. |
| **8** (Jun 15) | Red-team + qualitative eval. 500-prompt read-through. Safety eval (XSTest, ToxiGen small). License audit of all training sources. Draft model card. | End of week: model card draft, safety report. |
| **9** (Jun 22) | Buffer / slack week. Fix whatever broke. Rerun any failed eval. Final artifact push to HF Hub (private). | End of week: release candidate on private Hub. |
| **10** (Jun 29) | Public release. Blog post. GGUF/MLX/ONNX on Hub public. `Satellite-1.7B-v0.1` tag cut. | **v0.1 ships.** |

**Total DGX Spark GPU-hours:**

| Workload | Hours |
|---|---|
| Environment + smoke tests | 10 |
| Hadron generation (Qwen3-32B local) | 200 |
| Stage-1 SFT (3 epochs) | 80 |
| (Optional) merge + heal | 40 |
| (Optional) Stage-2 SFT | 30 |
| Eval runs (across stages) | 30 |
| Quantization / export | 5 |
| Slack / restarts / reruns | 25 |
| **Total** | **~420** |

At 20 h/day effective utilization that's 21 days of compute, well inside the 10-week plan.

---

## Open questions (owner: thekozugroup, resolve by Week 2)

1. Do we have commercial-terms coverage to use Claude API outputs as training data for a model we ship publicly? If not, drop to Qwen3-only teacher and accept ~5–10% quality delta.
2. Is there a Jetson Orin Nano device on hand for edge latency eval? If not, edge-latency numbers are Mac-only at v0.1.
3. Do we want to publish the TALOS quality scores alongside the dataset? (Transparency vs giving reviewers ammunition.)
4. What's the name of the Nano class if we ever split below 1B? (Not blocking; aesthetics.)

---

*End of plan. Review cadence: weekly during Weeks 1–4, daily during Weeks 5–6.*
