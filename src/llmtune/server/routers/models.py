"""Model catalog endpoints — static list + live HF search + cached models."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/models", tags=["models"])

POPULAR_MODELS: list[tuple[str, str, str]] = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat", "~2.2 GB"),
    ("Qwen/Qwen2.5-1.5B", "Qwen 2.5 1.5B", "~3.0 GB"),
    ("microsoft/phi-3-mini-4k-instruct", "Phi-3 Mini 3.8B", "~7.6 GB"),
    ("Qwen/Qwen2.5-7B", "Qwen 2.5 7B", "~14 GB"),
    ("meta-llama/Llama-3.2-1B", "Llama 3.2 1B", "~2.0 GB"),
    ("meta-llama/Llama-3.2-3B", "Llama 3.2 3B", "~6.4 GB"),
    ("meta-llama/Llama-3.1-8B", "Llama 3.1 8B", "~16 GB"),
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B", "~14 GB"),
    ("google/gemma-2-2b", "Gemma 2 2B", "~4.9 GB"),
    ("google/gemma-2-9b", "Gemma 2 9B", "~18 GB"),
]

POPULAR_CNN_MODELS: list[tuple[str, str, str]] = [
    ("microsoft/resnet-50", "ResNet-50", "~100 MB"),
    ("google/efficientnet-b0", "EfficientNet-B0", "~21 MB"),
    ("google/efficientnet-b4", "EfficientNet-B4", "~75 MB"),
    ("facebook/convnext-tiny-224", "ConvNeXt Tiny", "~113 MB"),
    ("google/vit-base-patch16-224", "ViT Base/16", "~330 MB"),
    ("facebook/deit-small-patch16-224", "DeiT Small", "~87 MB"),
    ("microsoft/swin-tiny-patch4-window7-224", "Swin Tiny", "~107 MB"),
]


class ModelInfo(BaseModel):
    id: str
    label: str
    size: str


class ModelsResponse(BaseModel):
    llm: list[ModelInfo]
    cnn: list[ModelInfo]


@router.get("", response_model=ModelsResponse)
def list_models():
    return ModelsResponse(
        llm=[ModelInfo(id=m[0], label=m[1], size=m[2]) for m in POPULAR_MODELS],
        cnn=[ModelInfo(id=m[0], label=m[1], size=m[2]) for m in POPULAR_CNN_MODELS],
    )


@router.get("/search")
def search_models(q: str, limit: int = 8, task: str = "text-generation"):
    """Live search HuggingFace Hub for models matching query."""
    try:
        from huggingface_hub import list_models as hf_list_models

        results = list(
            hf_list_models(
                search=q,
                task=task,
                limit=limit,
                sort="downloads",
                direction=-1,
            )
        )
        return [
            {
                "id": m.modelId,
                "label": m.modelId.split("/")[-1] if "/" in m.modelId else m.modelId,
                "downloads": getattr(m, "downloads", 0) or 0,
                "size": None,
            }
            for m in results
            if m.modelId
        ]
    except Exception:
        # Fail silently — user can still type the model ID manually
        return []


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _read_base_model(folder: Path, kind: str) -> str:
    """Best-effort read of the base model id a fine-tuned folder was trained from."""
    import json

    candidates = (
        ["adapter_config.json", "config.json"]
        if kind == "llm"
        else ["config.json"]
    )
    for name in candidates:
        f = folder / name
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("base_model_name_or_path", "_name_or_path"):
            val = data.get(key)
            if val:
                return str(val)
    return ""


@router.get("/trained")
def trained_models():
    """
    Scan the local output folders for fine-tuned models the user has produced.

    A folder counts as an LLM adapter if it has adapter_config.json /
    adapter_model.safetensors, or a CNN model if it has label_map.json (or a
    config + preprocessor pair). All on-device — nothing is uploaded.
    """
    roots = [
        (Path.home() / "llmtune-output", "llm"),
        (Path.home() / "llmtune-cnn-output", "cnn"),
    ]
    out: list[dict] = []
    for base, _default_kind in roots:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            # Skip intermediate checkpoints and merge artifacts — they aren't
            # standalone models the user produced.
            if d.name.startswith("checkpoint-") or d.name.endswith("-merged"):
                continue
            is_llm = (d / "adapter_config.json").is_file() or (
                d / "adapter_model.safetensors"
            ).is_file()
            is_cnn = (d / "label_map.json").is_file() or (
                (d / "config.json").is_file() and (d / "preprocessor_config.json").is_file()
            )
            if not (is_llm or is_cnn):
                continue
            kind = "llm" if is_llm else "cnn"
            try:
                modified = d.stat().st_mtime
            except OSError:
                modified = 0.0
            out.append(
                {
                    "name": d.name,
                    "path": str(d),
                    "mode": kind,
                    "base": _read_base_model(d, kind),
                    "size_bytes": _dir_size(d),
                    "modified": modified,
                }
            )
    out.sort(key=lambda x: x["modified"], reverse=True)
    return {"models": out}


@router.get("/cached")
def cached_models():
    """Return list of model IDs already present in the HuggingFace cache."""
    # HF_HUB_CACHE points straight at the hub dir; HF_HOME is the parent
    # ("$HF_HOME/hub"); otherwise fall back to the default cache location.
    if os.environ.get("HF_HUB_CACHE"):
        cache_dir = Path(os.environ["HF_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        cache_dir = Path(os.environ["HF_HOME"]) / "hub"
    else:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return {"models": []}

    models: list[str] = []
    try:
        for d in cache_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("models--"):
                continue
            # Convert "models--org--name" → "org/name"
            parts = d.name[len("models--"):].split("--")
            if len(parts) < 2:
                continue
            model_id = "/".join(parts)
            # Only include models that have actual snapshot content
            snapshots = d / "snapshots"
            if snapshots.exists():
                try:
                    if any(snapshots.iterdir()):
                        models.append(model_id)
                except PermissionError:
                    pass
    except PermissionError:
        pass

    return {"models": sorted(models)}
