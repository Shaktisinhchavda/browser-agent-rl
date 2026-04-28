"""
Evaluation script for trained browser control models.

Loads a trained model (with or without LoRA adapters) and runs it
against BrowserGym tasks to measure success rate.
"""

import sys
import os
import numpy as np
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import FineTuningConfig
from .env_client import BrowserGymClient
from .paths import get_path_to_media


def parse_action(response_text: str) -> str:
    """Parse BrowserGym action from model response."""
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line
    return "noop()"


def make_user_prompt(goal: str, step_num: int, axtree: str, error: str = "") -> str:
    """Create user prompt from observation."""
    prompt_parts = [f"Step {step_num + 1}"]

    if goal:
        prompt_parts.append(f"Goal: {goal}")

    if error:
        prompt_parts.append(f"Previous action error: {error}")

    if axtree:
        max_len = 1500
        axtree_truncated = axtree[:max_len] + "..." if len(axtree) > max_len else axtree
        prompt_parts.append(f"Page structure:\n{axtree_truncated}")

    prompt_parts.append("What action do you take?")

    return "\n\n".join(prompt_parts)


def save_screenshot(screenshot, episode: int, step: int) -> str:
    """Save screenshot to media directory."""
    media_dir = get_path_to_media()
    screenshot_array = np.array(screenshot, dtype=np.uint8)
    screenshot_image = Image.fromarray(screenshot_array)
    screenshot_path = os.path.join(media_dir, f"episode_{episode}_step_{step}.png")
    screenshot_image.save(screenshot_path)
    return screenshot_path


def evaluate(
    client: BrowserGymClient,
    model,
    tokenizer,
    system_prompt: str,
    default_goal: str,
    max_steps: int,
    episodes: int = 10,
):
    """
    Run the model against the BrowserGym environment for evaluation.

    Args:
        client: BrowserGymClient HTTP client
        model: Loaded model (with or without LoRA)
        tokenizer: Corresponding tokenizer
        system_prompt: System prompt for the model
        default_goal: Fallback goal text
        max_steps: Max steps per episode
        episodes: Number of evaluation episodes
    """
    successes = 0
    total = 0

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

        result = client.reset()
        observation = result.observation

        for step_num in range(max_steps):
            if result.done:
                break

            # Build prompt
            goal = observation.goal or default_goal
            axtree = observation.axtree_txt or ""
            error = observation.error if observation.last_action_error else ""

            user_prompt = make_user_prompt(goal, step_num, axtree, error)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Generate action
            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=64)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            action_str = parse_action(generated_text)
            print(f"  Step {step_num + 1}: raw={generated_text[:80]!r} → {action_str}")

            # Execute action
            result = client.step(action_str)
            observation = result.observation

        # Check if task was completed successfully
        reward = result.reward
        if reward > 0:
            successes += 1
            print(f"  ✅ Episode {episode + 1}: SUCCESS (reward={reward})")
        else:
            print(f"  ❌ Episode {episode + 1}: FAILED (reward={reward})")

        total += 1

    success_rate = successes / total if total > 0 else 0.0
    print(f"\n{'=' * 40}")
    print(f"Results: {successes}/{total} successful ({success_rate:.1%})")
    print(f"{'=' * 40}")

    return success_rate


def main():
    """CLI entrypoint for evaluation."""
    if len(sys.argv) < 2:
        print("Usage: python -m browser_control.evaluate <config_file.yaml> [model_path]")
        print("Example: python -m browser_control.evaluate qwen2_0.5b_lora.yaml ./checkpoints/my-run")
        sys.exit(1)

    config_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Check for --sft-checkpoint flag
    sft_checkpoint = None
    if "--sft-checkpoint" in sys.argv:
        idx = sys.argv.index("--sft-checkpoint")
        if idx + 1 < len(sys.argv):
            sft_checkpoint = sys.argv[idx + 1]

    config = FineTuningConfig.from_yaml(file_name=config_file)

    # Load base model
    print(f"Loading base model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Stage 1: Merge SFT LoRA if provided
    if sft_checkpoint:
        print(f"Loading and merging SFT LoRA from {sft_checkpoint}")
        model = PeftModel.from_pretrained(model, sft_checkpoint)
        model = model.merge_and_unload()
        print("SFT LoRA merged into base model")

    # Stage 2: Apply GRPO LoRA if provided
    if model_path and config.use_peft:
        print(f"Loading GRPO LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)

    # Connect to BrowserGym
    client = BrowserGymClient(base_url=config.browsergym_url)

    try:
        health = client.health()
        print(f"BrowserGym server is healthy: {health}")
    except Exception as e:
        print(f"❌ Cannot connect to BrowserGym: {e}")
        sys.exit(1)

    # Run evaluation
    evaluate(
        client=client,
        model=model,
        tokenizer=tokenizer,
        system_prompt=config.system_prompt,
        default_goal=config.default_goal,
        max_steps=config.max_steps,
        episodes=10,
    )


if __name__ == "__main__":
    main()
