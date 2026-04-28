"""
SFT Warmup for Browser Control

Teaches the model the BrowserGym action format using supervised fine-tuning
on examples collected from the environment. After this, the model can
produce click('13') reliably, enabling GRPO to work.

Usage:
    python -m browser_control.sft_warmup qwen2_0.5b_lora.yaml
"""

import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from .config import FineTuningConfig
from .env_client import BrowserGymClient
from .paths import get_path_model_checkpoints


def make_user_prompt(goal: str, step_num: int, axtree: str) -> str:
    """Build user prompt from browser state."""
    parts = [f"Step {step_num + 1}"]
    if goal:
        parts.append(f"Goal: {goal}")
    if axtree:
        max_len = 2000
        axtree_trunc = axtree[:max_len] + "..." if len(axtree) > max_len else axtree
        parts.append(f"Page structure:\n{axtree_trunc}")
    parts.append("What action do you take?")
    return "\n\n".join(parts)


def find_button_bid(axtree: str) -> str | None:
    """Extract the bid of the clickable button from the axtree."""
    import re
    # Look for [bid] button 'text' pattern
    match = re.search(r"\[(\d+)\]\s*button", axtree)
    if match:
        return match.group(1)
    return None


def collect_sft_data(
    client: BrowserGymClient,
    config: FineTuningConfig,
    tokenizer,
    num_examples: int = 50,
) -> Dataset:
    """
    Collect SFT training data by:
    1. Resetting the environment to get page observations
    2. Extracting the correct button bid from the axtree
    3. Creating (prompt, correct_action) pairs
    """
    texts = []

    print(f"Collecting {num_examples} SFT examples from BrowserGym...")

    for i in range(num_examples):
        try:
            result = client.reset()
            obs = result.observation
            goal = obs.goal or config.default_goal
            axtree = obs.axtree_txt or ""

            # Find the correct button bid
            bid = find_button_bid(axtree)
            if not bid:
                print(f"  ⚠ No button found in example {i}, skipping")
                continue

            correct_action = f"click('{bid}')"

            user_prompt = make_user_prompt(goal, step_num=0, axtree=axtree)

            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": correct_action},
            ]

            # Convert to training text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            if i < 3:
                print(f"  Example {i}: axtree has [{bid}] button → {correct_action}")

        except Exception as e:
            print(f"  ⚠ Failed to collect example {i}: {e}")

    print(f"Collected {len(texts)} valid SFT examples")

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )

    # For causal LM, labels = input_ids (model learns to predict next token)
    encodings["labels"] = encodings["input_ids"].clone()

    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["labels"],
    })


def sft_warmup(config: FineTuningConfig) -> str:
    """
    Run SFT warmup: teach model the action format, then save LoRA weights.
    Returns the path to the saved model.
    """
    # --- Connect to BrowserGym ---
    print(f"Connecting to BrowserGym at {config.browsergym_url}")
    client = BrowserGymClient(base_url=config.browsergym_url)

    try:
        health = client.health()
        print(f"BrowserGym server is healthy: {health}")
    except Exception as e:
        print(f"❌ Cannot connect to BrowserGym: {e}")
        sys.exit(1)

    # --- Load model and tokenizer ---
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
    )

    # --- Apply LoRA ---
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()  # Required for gradient checkpointing + LoRA
    model.print_trainable_parameters()

    # --- Collect SFT data ---
    dataset = collect_sft_data(
        client=client,
        config=config,
        tokenizer=tokenizer,
        num_examples=50,
    )

    # --- Output directory ---
    output_dir = get_path_model_checkpoints(
        f"{config.model_name.split('/')[-1]}-sft-warmup"
    )
    print(f"SFT checkpoint will be saved to: {output_dir}")

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    print("\n" + "=" * 60)
    print("Starting SFT warmup training...")
    print("=" * 60 + "\n")

    trainer.train()

    # --- Save LoRA weights ---
    print(f"\nSaving SFT warmup LoRA weights to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ SFT warmup complete! Model saved to: {output_dir}")
    print(f"\nNext step: Run GRPO training with this as the base:")
    print(f"  python -m browser_control.fine_tune qwen2_0.5b_lora.yaml --resume {output_dir}")

    return output_dir


def main():
    """CLI entrypoint"""
    if len(sys.argv) < 2:
        print("Usage: python -m browser_control.sft_warmup <config_file.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = FineTuningConfig.from_yaml(file_name=config_file)

    sft_warmup(config=config)


if __name__ == "__main__":
    main()
