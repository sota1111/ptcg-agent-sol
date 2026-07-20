# PTCG Agent Sol

Reproducible, local-first workspace for the Kaggle Pokémon TCG AI Battle Challenge. It provides a small JSONL agent protocol, CPU and RTX 3080 Ti presets, an eight-hour wall-clock guard, and an explicit Kaggle download workflow that keeps credentials and competition data out of Git.

> Before downloading or submitting anything, read and accept the competition rules on Kaggle. Competition data must remain in `data/`, which is ignored by Git.

## Quick start

Requirements: Python 3.11+ and Git. A CUDA-enabled PyTorch build is optional for the GPU preset.

```bash
git clone https://github.com/sota1111/ptcg-agent-sol.git
cd ptcg-agent-sol
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ptcg-agent doctor --config configs/cpu.toml
ptcg-agent smoke --config configs/cpu.toml
pytest
```

The smoke command runs without competition data or network access and prints a JSON summary.

## Agent I/O contract

`ptcg-agent run` reads one JSON object per line from stdin and writes exactly one JSON object per line to stdout. Logs go to stderr. Each input must contain a non-empty `legal_actions` list; the baseline deterministically selects the first action.

```bash
printf '%s\n' '{"request_id":"demo-1","observation":{},"legal_actions":["pass"]}' \
  | ptcg-agent run --config configs/cpu.toml
# {"request_id":"demo-1","action":"pass"}
```

This repository deliberately keeps the adapter narrow: update the input adapter only after confirming the current competition's official runtime and schema.

The adapter also accepts the common masked form, with equal-length `actions` and `action_mask` arrays
either at the request root or inside `observation`. It rejects empty masks and never asks a policy to
choose outside the normalized legal set.

## Reproducible policy training

The built-in tabular learner provides a dependency-free, deterministic baseline for validating the
complete train/checkpoint/evaluate lifecycle before wiring in competition data. It self-plays a small
turn-based battle proxy, saves versioned checkpoints atomically, and evaluates against the first-legal
action baseline.

```bash
ptcg-agent train --config configs/cpu.toml --episodes 10000 --checkpoint artifacts/policy.json
ptcg-agent train --config configs/cpu.toml --episodes 20000 --checkpoint artifacts/policy.json --resume
ptcg-agent evaluate --config configs/cpu.toml --checkpoint artifacts/policy.json --episodes 1000
printf '%s\n' '{"observation":{"remaining":3,"player":1},"legal_actions":[1,2,3]}' \
  | ptcg-agent run --config configs/cpu.toml --checkpoint artifacts/policy.json
```

Training uses the configured seed and honours the same maximum eight-hour budget as inference. On a
budget stop it writes a final checkpoint and exits 124; rerun with `--resume`. CPU and CUDA presets use
the same deterministic policy/checkpoint format. A CUDA preset verifies accelerator availability up
front, making device configuration failures explicit rather than silently falling back to CPU.

## CPU, GPU, and compute budget

Presets live in `configs/`. Override any preset safely with CLI flags:

```bash
ptcg-agent doctor --config configs/gpu-3080ti.toml
ptcg-agent run --config configs/gpu-3080ti.toml --max-hours 8 < requests.jsonl
```

- `cpu.toml` requires no accelerator.
- `gpu-3080ti.toml` requests CUDA, mixed precision, and a conservative batch size for 12 GB VRAM.
- `max_hours` must be in `(0, 8]`. The runner checks elapsed wall time before each request and exits with code 124 when exhausted. This is a guardrail, not a scheduler; allow time for checkpointing and submission preparation.
- `doctor` fails when a CUDA preset is selected but CUDA is unavailable. Install a PyTorch build matching your driver from the official PyTorch instructions, then install this project.

## Kaggle authentication and data

Never add a token to this repository. Choose one of Kaggle's supported local authentication methods:

1. Set `KAGGLE_API_TOKEN` in your shell or secret manager, or place the legacy token at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
2. Install the optional client: `python -m pip install -e '.[kaggle]'`.
3. Accept the competition rules in the Kaggle web UI.
4. Obtain the exact competition slug from its URL and download into the ignored cache:

```bash
ptcg-agent data download --competition <competition-slug>
```

The command invokes `kaggle competitions download -c <slug> -p data/raw` without printing credentials. Archives remain untouched under `data/raw`; inspect the competition license/rules before extracting or redistributing them. Generated models belong in ignored `artifacts/` unless the rules explicitly permit publication.

## Development checks

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

No competition data, model weights, cards, credentials, or proprietary Pokémon assets are included. Source code is available under the [MIT License](LICENSE); that license does not grant rights to third-party data, trademarks, or game assets.
