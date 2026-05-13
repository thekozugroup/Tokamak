# tokamak-engine

Memory-safe concurrent orchestrator for Tokamak's row-scoped reasoning agents.

Each row in a Tokamak run becomes its own short-lived agent — compress,
invert, or validate — and the engine fires them at a hard concurrency cap
against an OpenAI-compatible endpoint (vLLM, TGI, SGLang, Together, etc.).

## Why Rust

- **Memory safety** — concurrent processing of thousands of rows without
  data races. Each row's state is owned by its async task. No shared mutable
  state across rows.
- **Bounded concurrency** — a `tokio::sync::Semaphore` enforces the
  user-configured `seqs` count exactly. No drift, no overshoot.
- **Throughput** — `reqwest` over `rustls` with HTTP keep-alive and a single
  shared client pool. Saturates a busy local endpoint without a Python GIL
  bottleneck.

## Build

```bash
cd rust/tokamak-engine
pip install maturin
maturin develop --release      # installs `tokamak_engine` into current venv
```

Tokamak's Python pipeline automatically uses the Rust path when the
`tokamak_engine` module is importable; otherwise it falls back to a
`ThreadPoolExecutor`.

## Public API

```python
import tokamak_engine

results = tokamak_engine.batch_chat(
    base_url   = "http://localhost:8000/v1",
    api_key    = "dummy",
    model      = "Qwen/Qwen2.5-7B-Instruct",
    items      = [{"system": SYS, "user": "..."}, ...],
    max_workers= 64,
    timeout_s  = 180.0,
    retries    = 2,
    max_tokens = 4096,
    temperature= 0.0,
)
# results[i] is the assistant string or None on permanent failure.
```

## Scope

- One endpoint, one model per call.
- No streaming. Bulk batch only.
- No Anthropic / claude-CLI backends — those stay in Python.

Anything outside that scope belongs in `src/tokamak/llm_client.py`.
