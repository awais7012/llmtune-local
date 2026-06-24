"""Utilities for Ollama model detection and HuggingFace ID mapping."""

from __future__ import annotations

import subprocess

# ── Ollama → HuggingFace model ID mapping ───────────────────────────────────
OLLAMA_HF_MAP: dict[str, str] = {
    "tinyllama:latest":          "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama":                 "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2:1b":               "meta-llama/Llama-3.2-1B",
    "llama3.2:3b":               "meta-llama/Llama-3.2-3B",
    "llama3.1:8b":               "meta-llama/Llama-3.1-8B",
    "llama3.1:70b":              "meta-llama/Llama-3.1-70B",
    "llama3:8b":                 "meta-llama/Meta-Llama-3-8B",
    "llama3:70b":                "meta-llama/Meta-Llama-3-70B",
    "llama2:7b":                 "meta-llama/Llama-2-7b-hf",
    "llama2:13b":                "meta-llama/Llama-2-13b-hf",
    "mistral:7b":                "mistralai/Mistral-7B-v0.1",
    "mistral:latest":            "mistralai/Mistral-7B-v0.1",
    "mistral":                   "mistralai/Mistral-7B-v0.1",
    "phi3:mini":                 "microsoft/phi-3-mini-4k-instruct",
    "phi3:3.8b":                 "microsoft/phi-3-mini-4k-instruct",
    "phi3:medium":               "microsoft/Phi-3-medium-4k-instruct",
    "qwen2.5:0.5b":              "Qwen/Qwen2.5-0.5B",
    "qwen2.5:1.5b":              "Qwen/Qwen2.5-1.5B",
    "qwen2.5:7b":                "Qwen/Qwen2.5-7B",
    "qwen2.5:14b":               "Qwen/Qwen2.5-14B",
    "qwen2:7b":                  "Qwen/Qwen2-7B",
    "gemma2:2b":                 "google/gemma-2-2b",
    "gemma2:9b":                 "google/gemma-2-9b",
    "gemma:2b":                  "google/gemma-2b",
    "gemma:7b":                  "google/gemma-7b",
    "deepseek-coder:1.3b":       "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-coder:6.7b":       "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-coder:33b":        "deepseek-ai/deepseek-coder-33b-base",
    "codellama:7b":              "codellama/CodeLlama-7b-hf",
    "codellama:13b":             "codellama/CodeLlama-13b-hf",
    "codellama:34b":             "codellama/CodeLlama-34b-hf",
    "vicuna:7b":                 "lmsys/vicuna-7b-v1.5",
    "vicuna:13b":                "lmsys/vicuna-13b-v1.5",
    "solar:10.7b":               "upstage/SOLAR-10.7B-v1.0",
    "neural-chat:7b":            "Intel/neural-chat-7b-v3-3",
    "starling-lm:7b":            "berkeley-nest/Starling-LM-7B-alpha",
    "openchat:7b":               "openchat/openchat-3.5-0106",
    "zephyr:7b":                 "HuggingFaceH4/zephyr-7b-beta",
    "dolphin-mistral:7b":        "cognitivecomputations/dolphin-2.6-mistral-7b",
    "nous-hermes2:10.7b":        "NousResearch/Nous-Hermes-2-SOLAR-10.7B",
    "wizard-vicuna-uncensored:7b": "cognitivecomputations/WizardLM-7B-Uncensored",
}


def is_ollama_installed() -> bool:
    """Return True if the ollama CLI is available."""
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def list_ollama_models() -> list[dict]:
    """Return installed Ollama models as dicts with keys: name, size_str, hf_id."""
    try:
        r = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return []
        models: list[dict] = []
        for line in r.stdout.strip().splitlines()[1:]:  # skip header row
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            size_str = f"{parts[2]} {parts[3]}" if len(parts) > 3 else "?"
            models.append({
                "name": name,
                "size_str": size_str,
                "hf_id": map_ollama_to_hf(name),
            })
        return models
    except Exception:
        return []


def map_ollama_to_hf(model_name: str) -> str | None:
    """Map an Ollama model name to its HuggingFace model ID (best guess)."""
    lower = model_name.lower()
    if lower in OLLAMA_HF_MAP:
        return OLLAMA_HF_MAP[lower]
    base = lower.split(":")[0]
    for key, hf_id in OLLAMA_HF_MAP.items():
        if key.split(":")[0] == base:
            return hf_id
    return None
