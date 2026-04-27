# ============================================
# Makefile for Browser Control GRPO Training
# ============================================

# Default config file
CONFIG ?= qwen2_0.5b_lora.yaml

# ---- Docker commands ----

.PHONY: build up down train train-full env logs clean

# Build all Docker images
build:
	docker compose build

# Start BrowserGym environment only
env:
	docker compose up browsergym-env -d

# Start training (launches both containers)
train:
	CONFIG_FILE=$(CONFIG) docker compose up

# Start training in background
train-bg:
	CONFIG_FILE=$(CONFIG) docker compose up -d

# Full fine-tune (not recommended for 4GB VRAM)
train-full:
	CONFIG_FILE=qwen2_0.5b_full.yaml docker compose up

# View training logs
logs:
	docker compose logs -f training-gpu

# View BrowserGym logs
logs-env:
	docker compose logs -f browsergym-env

# Stop all containers
down:
	docker compose down

# Remove everything (containers, volumes, images)
clean:
	docker compose down -v --rmi all

# ---- Local development (with uv) ----

.PHONY: install sync local-train local-eval lock

# Install uv (if not already installed)
install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
sync:
	uv sync

# Generate/update uv.lock
lock:
	uv lock

# Train locally (requires BrowserGym running separately)
local-train:
	uv run python -m browser_control.fine_tune $(CONFIG)

# Evaluate a trained model
local-eval:
	uv run python -m browser_control.evaluate $(CONFIG) $(MODEL_PATH)

# ---- Utilities ----

.PHONY: check-gpu check-env

# Check if GPU is available
check-gpu:
	nvidia-smi

# Check if BrowserGym environment is running
check-env:
	curl -s http://localhost:7860/health | python -m json.tool
