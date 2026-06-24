"""Dataclass for CNN / image-classification fine-tuning hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ImageTrainConfig:
    # Auth
    hf_token: Optional[str] = None

    # Model
    model_id: str = "microsoft/resnet-50"

    # Dataset
    dataset_path: str = ""
    val_split: float = 0.1  # fraction held out for validation when no val split exists

    # Training mode
    # feature_extraction — backbone frozen, only classifier head trained  (fastest)
    # full              — all layers trained with a lower LR              (most accurate)
    # lora              — LoRA adapters on attention layers (ViT / Swin)  (good balance)
    training_mode: Literal["feature_extraction", "full", "lora"] = "feature_extraction"

    # Image preprocessing
    image_size: int = 224

    # Training
    num_epochs: int = 10
    per_device_train_batch_size: int = 16
    learning_rate: float = 1e-3
    lr_scheduler_type: str = "cosine"

    # LoRA (used when training_mode == "lora")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Output
    output_dir: str = str(Path.home() / "llmtune-cnn-output")
    save_steps: int = 50
    logging_steps: int = 10
