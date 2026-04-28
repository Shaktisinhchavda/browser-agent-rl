# ============================================
# Makefile for Browser Control GRPO Training
# ============================================

CONFIG ?= qwen2_0.5b_lora.yaml

.PHONY: build up down sft train eval logs clean

# Build all Docker images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Stop all containers
down:
	docker compose down

# Step 1: SFT warmup (teaches model click('bid') format)
sft:
	docker compose run --rm training-gpu python -m browser_control.sft_warmup $(CONFIG)

# Step 2: GRPO training (RL on top of SFT model)
train:
	docker compose run --rm training-gpu python -m browser_control.fine_tune $(CONFIG) --sft-checkpoint /model_checkpoints/Qwen2.5-0.5B-Instruct-sft-warmup

# Step 3: Evaluate (pass MODEL_PATH= and SFT_PATH=)
eval:
	docker compose run --rm training-gpu python -m browser_control.evaluate $(CONFIG) $(MODEL_PATH) --sft-checkpoint $(SFT_PATH)

# View training logs
logs:
	docker compose logs -f training-gpu

# View BrowserGym logs
logs-env:
	docker compose logs -f browsergym-env

# Remove everything
clean:
	docker compose down -v --rmi all
