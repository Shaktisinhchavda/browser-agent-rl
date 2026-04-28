"""
GRPO Fine-tuning for Browser Control

This is the core training script. It:
1. Connects to the BrowserGym environment (running in a separate Docker container)
2. Uses GRPOTrainer from HuggingFace TRL for policy optimization
3. Logs everything to WandB

Adapted from Liquid4All/cookbook browser-control example.
Note: rollout_func requires vLLM (unavailable on 4GB VRAM),
so we use standard GRPO with environment-based reward instead.
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
# Prompt construction & action parsing (same as reference repo)
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
        max_len = 2000
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
# Reward function: runs action in environment + shaped bonus
# ---------------------------------------------------------------------------

def make_reward_func(client: BrowserGymClient, system_prompt: str, max_steps: int):
    """
    Reward function for GRPO that executes actions in BrowserGym.

    Returns 1.0 for task success, with shaped bonuses for valid actions.
    """
    import re

    def reward_func(completions, **kwargs) -> list[float]:
        rewards = []

        for completion in completions:
            # Extract generated text
            if isinstance(completion, list):
                text = completion[-1].get("content", "")
            elif isinstance(completion, str):
                text = completion
            else:
                text = str(completion)

            raw = text.strip()
            action_str = parse_action(raw)

            # Shaped reward ladder
            reward = 0.0

            # Has action keyword
            if any(kw in raw.lower() for kw in ["click", "fill", "scroll"]):
                reward = 0.05

            # Has parentheses (action-like format)
            if "(" in raw and ")" in raw:
                reward = 0.1

            # Valid parseable action
            if action_str != "noop()":
                reward = 0.2

            # Click with a real bid number
            bid_match = re.search(r"click\(['\"](\d+)['\"]\)", action_str)
            if bid_match:
                reward = 0.5

            # Execute in environment for full reward
            if bid_match:
                try:
                    result = client.reset()
                    if not result.done:
                        result = client.step(action_str)
                    if result.reward > 0:
                        reward = 1.0
                except Exception:
                    pass

            rewards.append(reward)
            print(f"  [{raw[:50]!r}] → {action_str} → R:{reward}")

        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# Build training dataset with one-shot example
# ---------------------------------------------------------------------------

def build_dataset(config: FineTuningConfig, client: BrowserGymClient) -> Dataset:
    """
    Build training dataset: collect page observations from BrowserGym.
    No one-shot example — system prompt already explains the format.
    """
    prompts = []

    for i in range(config.dataset_size):
        try:
            result = client.reset()
            obs = result.observation

            goal = obs.goal or config.default_goal
            axtree = obs.axtree_txt or ""

            user_prompt = make_user_prompt(goal, step_num=0, axtree=axtree)

            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompts.append(messages)

        except Exception as e:
            print(f"  ⚠ Failed to collect prompt {i}: {e}")
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": f"Step 1\n\nGoal: {config.default_goal}\n\nWhat action do you take?"},
            ]
            prompts.append(messages)

    print(f"Collected {len(prompts)} training prompts from BrowserGym")

    return Dataset.from_dict({"prompt": prompts})


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
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def fine_tune(config: FineTuningConfig) -> None:
    """
    Fine-tune a language model for browser control using GRPO.
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

    try:
        health = client.health()
        print(f"BrowserGym server is healthy: {health}")
    except Exception as e:
        print(f"❌ Cannot connect to BrowserGym at {config.browsergym_url}")
        print(f"   Error: {e}")
        sys.exit(1)

    # --- Build training dataset (prompts with one-shot example) ---
    print("Collecting training prompts from BrowserGym...")
    dataset = build_dataset(config, client)

    # --- Output directory for checkpoints ---
    output_dir = get_path_model_checkpoints(config.wandb_experiment_name)
    print(f"Checkpoints will be saved to: {output_dir}")

    # --- Build quantization config ---
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
        temperature=1.5,  # Higher temp = more diverse generations for GRPO exploration
    )

    if config.use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = config.vllm_mode
        grpo_kwargs["vllm_gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
        print(f"vLLM enabled: mode={config.vllm_mode}, "
              f"gpu_mem={config.vllm_gpu_memory_utilization}")

    if config.gradient_checkpointing:
        grpo_kwargs["gradient_checkpointing"] = True
        print("Gradient checkpointing enabled")

    grpo_config = GRPOConfig(**grpo_kwargs)

    # --- Build PEFT/LoRA config ---
    peft_config = create_peft_config(config)
    if peft_config:
        print(f"LoRA enabled: r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"Target modules: {config.lora_target_modules}")

    # --- Model kwargs ---
    model_kwargs = {}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    # --- Reward function ---
    reward_fn = make_reward_func(client, config.system_prompt, config.max_steps)

    # --- Load model (optionally from SFT warmup checkpoint) ---
    sft_checkpoint = None
    if len(sys.argv) > 3 and sys.argv[2] == "--sft-checkpoint":
        sft_checkpoint = sys.argv[3]

    if sft_checkpoint:
        from peft import PeftModel
        print(f"Loading base model {config.model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **model_kwargs
        )
        print(f"Loading SFT warmup LoRA from {sft_checkpoint}...")
        model = PeftModel.from_pretrained(base_model, sft_checkpoint)
        model = model.merge_and_unload()  # Merge LoRA into base for fresh GRPO LoRA
        print("SFT LoRA merged into base model. GRPO will add fresh LoRA on top.")
    else:
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

    if config.push_to_hf:
        print("Pushing model to HuggingFace Hub...")
        trainer.push_to_hub()

    print("\n✅ Training complete!")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    """CLI entrypoint: python -m browser_control.fine_tune <config_file>"""
    if len(sys.argv) < 2:
        print("Usage: python -m browser_control.fine_tune <config_file.yaml>")
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
