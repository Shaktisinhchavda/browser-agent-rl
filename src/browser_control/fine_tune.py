"""
GRPO Fine-tuning for Browser Control

This is the core training script. It:
1. Connects to the BrowserGym environment (running in a separate Docker container)
2. Uses GRPOTrainer from HuggingFace TRL for policy optimization
3. Optionally uses vLLM (colocated on GPU) for fast rollout generation
4. Logs everything to WandB

Adapted from Liquid4All/cookbook browser-control example,
but runs locally on your own GPU instead of Modal.
"""

import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
import wandb

from .config import FineTuningConfig
from .env_client import BrowserGymClient
from .paths import get_path_model_checkpoints


# ---------------------------------------------------------------------------
# Prompt construction & action parsing
# ---------------------------------------------------------------------------

def make_user_prompt(goal: str, step_num: int, axtree: str, error: str = "") -> str:
    """Build user prompt from the current browser state."""
    prompt_parts = [f"Step {step_num + 1}"]

    if goal:
        prompt_parts.append(f"Goal: {goal}")

    if error:
        prompt_parts.append(f"Previous action error: {error}")

    # Include accessibility tree (truncated to fit context)
    if axtree:
        max_len = 1500  # Conservative for 0.5B model with 1024 context
        axtree_truncated = axtree[:max_len] + "..." if len(axtree) > max_len else axtree
        prompt_parts.append(f"Page structure:\n{axtree_truncated}")

    prompt_parts.append("What action do you take?")

    return "\n\n".join(prompt_parts)


def parse_action(response_text: str) -> str:
    """Extract BrowserGym action from model response text."""
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line
    # Fallback to noop
    return "noop()"


# ---------------------------------------------------------------------------
# Reward function (used by GRPOTrainer)
# ---------------------------------------------------------------------------

def make_reward_func(client: BrowserGymClient, system_prompt: str, max_steps: int):
    """
    Shaped reward function — v3 (fixes click('') shortcut).

    The model already learned click() format. Now it needs to learn to
    extract the correct bid from the page structure.

    Reward ladder:
      0.0   — gibberish or noop
      0.05  — contains action keyword
      0.1   — click with empty bid: click('') — penalized!
      0.3   — click with an actual bid: click('13')
      0.5   — click with a bid that exists in the page
      1.0   — task actually completed in the environment
    """
    import re

    def reward_func(completions: list[list[dict]], prompts=None, **kwargs) -> list[float]:
        rewards = []

        for idx, completion_messages in enumerate(completions):
            if isinstance(completion_messages, list):
                generated_text = completion_messages[-1].get("content", "")
            elif isinstance(completion_messages, str):
                generated_text = completion_messages
            else:
                generated_text = str(completion_messages)

            raw_text = generated_text.strip()
            action_str = parse_action(raw_text)

            # --- Reward ladder ---
            reward = 0.0

            # Level 1: Contains action keyword
            if any(kw in raw_text.lower() for kw in ["click", "fill", "scroll"]):
                reward = 0.05

            # Level 2: click() with empty bid — learned format but lazy
            if action_str.startswith("click(") and action_str in ("click('')", 'click("")', "click()"):
                reward = 0.1  # Penalized vs having a real bid

            # Level 3: click() with an actual bid value
            bid_match = re.search(r"click\(['\"](\d+)['\"]\)", action_str)
            if bid_match:
                reward = 0.3
                bid_value = bid_match.group(1)

                # Level 4: bid exists in the prompt context (model read the page)
                # Get the prompt for this completion to check bid validity
                prompt_text = ""
                if prompts and idx < len(prompts):
                    p = prompts[idx]
                    if isinstance(p, list):
                        prompt_text = " ".join(m.get("content", "") for m in p)
                    else:
                        prompt_text = str(p)

                if f"[{bid_value}]" in prompt_text:
                    reward = 0.5

            # Level 5: Execute in environment for actual task reward
            if action_str != "noop()" and action_str not in ("click('')", 'click("")', "click()"):
                try:
                    result = client.reset()
                    if not result.done:
                        result = client.step(action_str)
                    if result.reward > 0:
                        reward = 1.0
                except Exception:
                    pass  # Keep format-based reward

            rewards.append(reward)
            print(f"  Raw: {raw_text[:60]!r} → Action: {action_str} → R: {reward}")

        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# PEFT / LoRA config builder
# ---------------------------------------------------------------------------

def create_peft_config(config: FineTuningConfig) -> LoraConfig | None:
    """Create LoRA config from training config. Returns None if PEFT is disabled."""
    if not config.use_peft:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
        use_rslora=config.use_rslora,
    )


# ---------------------------------------------------------------------------
# Quantization config builder (for QLoRA on 4GB VRAM)
# ---------------------------------------------------------------------------

def create_quantization_config(config: FineTuningConfig) -> BitsAndBytesConfig | None:
    """Create 4-bit quantization config for QLoRA. Returns None if disabled."""
    if not config.use_4bit:
        return None

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.float16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    )


# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

def build_dataset(config: FineTuningConfig, client: BrowserGymClient) -> Dataset:
    """
    Build a training dataset of prompts for GRPO.

    Each prompt contains the system prompt + a browser observation
    from a fresh environment reset.
    """
    prompts = []

    for i in range(config.dataset_size):
        try:
            result = client.reset()
            obs = result.observation

            goal = obs.goal or config.default_goal
            axtree = obs.axtree_txt or ""

            user_prompt = make_user_prompt(goal, step_num=0, axtree=axtree)

            # Build chat-format prompt
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            prompts.append(messages)

        except Exception as e:
            print(f"  ⚠ Failed to collect prompt {i}: {e}")
            # Fallback prompt
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": f"Goal: {config.default_goal}\n\nWhat action do you take?"},
            ]
            prompts.append(messages)

    print(f"Collected {len(prompts)} training prompts from BrowserGym")

    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def fine_tune(config: FineTuningConfig) -> None:
    """
    Fine-tune a language model for browser control using GRPO.

    This replaces the Modal-based approach from the reference repo
    with direct local GPU execution via Docker.
    """
    # --- WandB setup ---
    if config.wandb_enabled:
        print(f"Initializing WandB: project={config.wandb_project_name}, "
              f"run={config.wandb_experiment_name}")
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_experiment_name,
            config=config.__dict__,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # --- Connect to BrowserGym environment ---
    print(f"Connecting to BrowserGym at {config.browsergym_url}")
    client = BrowserGymClient(base_url=config.browsergym_url)

    # Verify connection
    try:
        health = client.health()
        print(f"BrowserGym server is healthy: {health}")
    except Exception as e:
        print(f"❌ Cannot connect to BrowserGym at {config.browsergym_url}")
        print(f"   Error: {e}")
        print(f"   Make sure the browsergym-env container is running:")
        print(f"   docker compose up browsergym-env -d")
        sys.exit(1)

    # --- Build training dataset (prompts from BrowserGym) ---
    print("Collecting training prompts from BrowserGym...")
    dataset = build_dataset(config, client)

    # --- Output directory for checkpoints ---
    output_dir = get_path_model_checkpoints(config.wandb_experiment_name)
    print(f"Checkpoints will be saved to: {output_dir}")

    # --- Build quantization config (QLoRA) ---
    quant_config = create_quantization_config(config)
    if quant_config:
        print("4-bit quantization enabled (QLoRA mode)")

    # --- Build GRPOConfig ---
    print("Creating GRPOConfig...")
    grpo_kwargs = dict(
        max_steps=config.dataset_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        max_completion_length=config.max_completion_length,
        output_dir=output_dir,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.wandb_enabled else "none",
        seed=config.seed,
        remove_unused_columns=False,
    )

    # vLLM settings (only when enabled — requires Linux + GPU)
    if config.use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = config.vllm_mode
        grpo_kwargs["vllm_gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
        print(f"vLLM enabled: mode={config.vllm_mode}, "
              f"gpu_mem={config.vllm_gpu_memory_utilization}")

    # Add gradient checkpointing if enabled
    if config.gradient_checkpointing:
        grpo_kwargs["gradient_checkpointing"] = True
        print("Gradient checkpointing enabled (saves VRAM, slower training)")

    grpo_config = GRPOConfig(**grpo_kwargs)

    # --- Build PEFT/LoRA config ---
    peft_config = create_peft_config(config)
    if peft_config:
        print(f"LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"Target modules: {config.lora_target_modules}")

    # --- Build model kwargs (for quantization) ---
    model_kwargs = {}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    # --- Create reward function ---
    reward_fn = make_reward_func(client, config.system_prompt, config.max_steps)

    # --- Load Model Manually (to handle quantization without model_init_kwargs) ---
    print(f"Loading model {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )

    # --- Create GRPOTrainer ---
    print(f"Setting up GRPOTrainer with model: {config.model_name}")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
    )

    # --- Train ---
    print("\n" + "=" * 60)
    print("Starting GRPO training...")
    print("=" * 60 + "\n")

    trainer_stats = trainer.train(
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    # --- Save model ---
    print(f"\nSaving model to {output_dir}")
    if config.use_peft:
        print("Saving LoRA adapter weights only")
    trainer.save_model(output_dir)

    # --- Optionally push to HF Hub ---
    if config.push_to_hf:
        print("Pushing model to HuggingFace Hub...")
        if config.use_peft:
            print("Note: Pushing LoRA adapters only")
        trainer.push_to_hub()

    print("\n✅ Training complete!")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    """CLI entrypoint: python -m browser_control.fine_tune <config_file>"""
    if len(sys.argv) < 2:
        print("Usage: python -m browser_control.fine_tune <config_file.yaml>")
        print("Example: python -m browser_control.fine_tune qwen2_0.5b_lora.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    config = FineTuningConfig.from_yaml(file_name=config_file)

    try:
        fine_tune(config=config)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
