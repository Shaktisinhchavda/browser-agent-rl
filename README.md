# Browser Control Fine-Tuning with GRPO + RL

Fine-tune a small LLM (Qwen2.5-0.5B) for browser control using GRPO (Group Relative Policy Optimization)
on a local 4GB VRAM GPU. Uses BrowserGym via OpenEnv for the RL environment and WandB for monitoring.

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Docker: training-gpu                    │
│  ┌──────────────────┐  ┌──────────────────────┐ │
│  │   GRPOTrainer    │  │   vLLM Server        │ │
│  │   (LoRA + 4-bit) │  │   (Colocated GPU)    │ │
│  │                  │◄─┤   Qwen2.5-0.5B       │ │
│  └──────────────────┘  └──────────────────────┘ │
│         GPU (4GB VRAM)                           │
│         WandB Logging ──► wandb.ai              │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (reward queries)
                   ▼
┌─────────────────────────────────────────────────┐
│          Docker: browsergym-env                  │
│  ┌──────────────────────────────────────────┐   │
│  │   BrowserGym (MiniWoB - click-test)       │   │
│  │   OpenEnv Server (FastAPI)               │   │
│  │   Playwright + Chromium                  │   │
│  └──────────────────────────────────────────┘   │
│         CPU Only                                 │
└─────────────────────────────────────────────────┘
```

## Components

| Component | Description |
|-----------|-------------|
| **Model** | `Qwen/Qwen2.5-0.5B` — 0.5B params, fits in 4GB with QLoRA |
| **RL Algorithm** | GRPO via HuggingFace TRL |
| **Environment** | BrowserGym (MiniWoB++ `click-test` task) via OpenEnv |
| **Fine-tuning** | LoRA + 4-bit quantization (QLoRA) |
| **Inference** | vLLM colocated on same GPU as trainer |
| **Monitoring** | Weights & Biases (WandB) |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) — fast Python package manager |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker + Docker Compose
- NVIDIA GPU with 4GB+ VRAM
- NVIDIA Container Toolkit (`nvidia-docker`)
- WandB account (free tier works)

## Quick Start

### 1. Install uv (if not installed)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup environment variables

```bash
cp .env.example .env
# Edit .env and add your WANDB_API_KEY
```

### 3. Build and launch (Docker)

```bash
# Build both containers
docker compose build

# Start the BrowserGym environment first
docker compose up browsergym-env -d

# Wait ~30s for it to be ready, then start training
docker compose up training-gpu
```

### 4. Local development (without Docker)

```bash
# Install all dependencies + create venv
uv sync

# Run training locally
uv run python -m browser_control.fine_tune qwen2_0.5b_lora.yaml

# Evaluate a checkpoint
uv run python -m browser_control.evaluate qwen2_0.5b_lora.yaml ./checkpoints/my-run
```

### 5. Monitor on WandB

Open [wandb.ai](https://wandb.ai) and look for the project `browser-control-grpo`.

## Configuration

Edit `configs/qwen2_0.5b_lora.yaml` to adjust:
- `learning_rate`, `warmup_steps` — training speed
- `num_generations`, `max_steps` — rollout settings
- `lora_r`, `lora_alpha` — LoRA rank/scaling
- `vllm_gpu_memory_utilization` — GPU memory split between vLLM and trainer

## Task: click-test

The `click-test` task from MiniWoB++ presents a simple button on a page.
The model must learn to output `click('<bid>')` to click the correct button.
This is the simplest browser control task — perfect for validating the pipeline.

## Project Structure

```
browser-control/
├── configs/
│   ├── qwen2_0.5b_lora.yaml        # LoRA config (recommended for 4GB)
│   └── qwen2_0.5b_full.yaml        # Full fine-tune config
├── src/
│   └── browser_control/
│       ├── __init__.py
│       ├── config.py                # Pydantic config loader
│       ├── fine_tune.py             # GRPO training loop
│       ├── evaluate.py              # Evaluation script
│       └── paths.py                 # Path utilities
├── docker/
│   ├── Dockerfile.training          # GPU training container (uses uv)
│   ├── Dockerfile.browsergym        # BrowserGym env container (uses uv)
│   └── browsergym_server.py         # FastAPI server for BrowserGym
├── docker-compose.yml
├── pyproject.toml                   # uv-managed project config
├── .python-version                  # Python version pin for uv
├── .env.example
├── Makefile
└── README.md
```
