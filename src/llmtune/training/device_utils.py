"""Cross-platform device detection and safe model saving."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' — works on Mac, Windows, and Linux."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    labels = {
        "cuda": "NVIDIA GPU (CUDA)",
        "mps": "Apple Silicon GPU",
        "cpu": "CPU",
    }
    return labels.get(device.lower(), device.upper())


def default_output_dir(kind: str = "llm") -> str:
    """Platform-correct default output folder in the user's home directory."""
    name = "llmtune-cnn-output" if kind == "cnn" else "llmtune-output"
    return str(Path.home() / name)


def platform_name() -> str:
    return platform.system()  # Darwin, Windows, Linux


def safe_save_trainer(trainer: Any, output_dir: str, device: str) -> None:
    """
    Save HuggingFace Trainer weights safely on all hardware.

    Apple Silicon (MPS) can segfault if empty_cache/synchronize run after training.
    Moving to CPU before save avoids that on every platform.
    """
    if device == "mps":
        trainer.model = trainer.model.to("cpu")
    trainer.save_model(output_dir)
