"""App configuration endpoint."""

from __future__ import annotations

import platform
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from llmtune.training.device_utils import default_output_dir, detect_device, device_label

router = APIRouter(prefix="/api", tags=["config"])


class AppConfig(BaseModel):
    # No login required — kept for frontend compatibility.
    skip_auth: bool = True
    platform: str
    device: str
    device_label: str
    default_llm_output: str
    default_cnn_output: str
    home_dir: str
    vram_gb: float | None = None


@router.get("/config", response_model=AppConfig)
def get_config():
    dev = detect_device()
    vram: float | None = None
    try:
        import torch
        if torch.cuda.is_available():
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        # MPS shares system RAM — report None so frontend uses default
    except Exception:
        pass
    return AppConfig(
        skip_auth=True,
        platform=platform.system(),
        device=dev,
        device_label=device_label(dev),
        default_llm_output=default_output_dir("llm"),
        default_cnn_output=default_output_dir("cnn"),
        home_dir=str(Path.home()),
        vram_gb=vram,
    )
