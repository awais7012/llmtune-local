"""Dataclass for all fine-tuning hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainConfig:
    # Auth
    hf_token: Optional[str] = None  # kept in memory only, never written to disk

    # Model
    model_id: str = "meta-llama/Llama-3.2-1B"
    quantization: Literal["none", "4bit", "8bit"] = "4bit"

    # Dataset
    dataset_path: str = ""
    dataset_format: Literal["json", "jsonl", "csv", "text"] = "jsonl"
    text_column: str = "text"
    max_seq_length: int = 512

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = False

    # Output
    output_dir: str = str(Path.home() / "llmtune-output")
    save_steps: int = 50
    logging_steps: int = 5

    # Resume / publish
    resume_from_checkpoint: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
