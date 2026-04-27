from typing import Optional, Self
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs


class FineTuningConfig(BaseSettings):
    """Configuration for GRPO fine-tuning of browser control models."""

    seed: int
    resume_from_checkpoint: Optional[str] = None

    # Language Model parameters
    model_name: str          # HuggingFace model name (e.g. Qwen/Qwen2.5-0.5B)
    max_seq_length: int      # Context window size in tokens
    system_prompt: str       # System prompt for the LM

    # BrowserGym environment
    browsergym_url: str      # URL where BrowserGym server is accessible
    dataset_size: int        # Number of training episodes
    default_goal: str        # Default goal if env doesn't provide one

    # Training hyperparameters
    learning_rate: float     # Max learning rate for optimizer
    warmup_steps: int        # Linear warmup steps

    # vLLM inference settings
    max_steps: int                      # Max steps per rollout episode
    per_device_train_batch_size: int    # Samples per device per step
    num_generations: int                # Completions per prompt for GRPO
    generation_batch_size: int          # Batch size during generation
    max_completion_length: int          # Max tokens in generated completions
    use_vllm: bool                      # Whether to use vLLM for generation
    vllm_mode: str                      # "colocate" = same GPU as trainer
    vllm_gpu_memory_utilization: float  # Fraction of GPU memory for vLLM

    # 4-bit quantization (QLoRA)
    use_4bit: bool = False                      # Enable 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"     # Compute dtype for 4-bit
    bnb_4bit_quant_type: str = "nf4"            # Quantization type

    # Gradient checkpointing
    gradient_checkpointing: bool = False

    # Experiment tracking
    wandb_enabled: bool
    wandb_project_name: str
    wandb_experiment_name: Optional[str] = None
    logging_steps: int
    push_to_hf: Optional[bool] = False

    # LoRA configuration
    use_peft: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        """Load configuration from a YAML file in the configs directory."""
        file_path = str(Path(get_path_to_configs()) / file_name)
        print(f"Loading config from {file_path}")
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @model_validator(mode="after")
    def set_experiment_name(self):
        """Auto-generate experiment name from model name + timestamp."""
        if self.wandb_experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.wandb_experiment_name = f"{model_short}-browsergym-{timestamp}"
        return self
