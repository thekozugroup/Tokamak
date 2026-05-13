# tokamak.skill

A Claude Code / Claude Agent SDK skill bundle for [Tokamak](https://github.com/thekozugroup/Tokamak).

## Install

Drop the unpacked directory under your skills root, then run the installer once:

```bash
# Claude Code
cp -r tokamak.skill ~/.claude/skills/tokamak
bash ~/.claude/skills/tokamak/install.sh
```

`install.sh` detects the host platform, picks the matching prebuilt binary from `bin/`, and symlinks it to `~/.local/bin/tokamak`. No Rust toolchain, no Python, no Anthropic SDK — single binary, ~4 MB.

## What's inside

```
tokamak.skill/
  SKILL.md              # agent guide for elicitation + orchestration
  install.sh            # picks the right prebuilt binary
  bin/
    tokamak-darwin-arm64
    tokamak-darwin-x86_64
    tokamak-linux-x86_64
    tokamak-linux-aarch64
  prompts/              # bundled compress / invert / validate prompts
  examples/sample_trace.jsonl
```

## Build the bundle yourself

```bash
git clone https://github.com/thekozugroup/Tokamak
cd Tokamak/rust
cargo build -p tokamak --release
cp target/release/tokamak ../skill/bin/tokamak-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)
```

Or trigger the GitHub Actions release workflow:

```bash
gh workflow run release.yml
gh run download -p tokamak.skill
```
