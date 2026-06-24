"""Merge a LoRA adapter into its base model, and export to GGUF.

Used by the model-library actions (test/inference, push to Hub, GGUF export).
All heavy imports are local so importing this module stays cheap.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

Logger = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def read_base_model_id(model_dir: str | Path) -> Optional[str]:
    """Return the base model id from an adapter folder, or None if not an adapter."""
    cfg = Path(model_dir) / "adapter_config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return data.get("base_model_name_or_path")
    except Exception:
        return None


def is_adapter(model_dir: str | Path) -> bool:
    return (Path(model_dir) / "adapter_config.json").is_file()


def merge_adapter(
    model_dir: str,
    out_dir: str,
    hf_token: str | None = None,
    on_log: Logger = _noop,
) -> str:
    """
    Merge a LoRA adapter into its base model and save a standalone HF model to
    out_dir. If model_dir is already a full model (no adapter), it's copied/loaded
    and re-saved. Returns out_dir.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok_kwargs: dict = {"trust_remote_code": True}
    load_kwargs: dict = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if hf_token:
        tok_kwargs["token"] = hf_token
        load_kwargs["token"] = hf_token

    if is_adapter(model_dir):
        from peft import PeftModel

        base_id = read_base_model_id(model_dir)
        if not base_id:
            raise RuntimeError("Adapter is missing its base model id (adapter_config.json).")
        on_log(f"Loading base model: {base_id}")
        base = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
        on_log("Applying LoRA adapter…")
        model = PeftModel.from_pretrained(base, model_dir)
        on_log("Merging adapter weights into the base model…")
        model = model.merge_and_unload()
        tok = AutoTokenizer.from_pretrained(model_dir, **tok_kwargs)
    else:
        on_log("Loading full model…")
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        tok = AutoTokenizer.from_pretrained(model_dir, **tok_kwargs)

    on_log(f"Saving merged model → {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return out_dir


# Where we auto-install llama.cpp if the user doesn't already have it.
_MANAGED_LLAMA_DIR = Path.home() / ".llmtune" / "llama.cpp"
_LLAMA_REPO = "https://github.com/ggerganov/llama.cpp.git"


def _converter_in(base: Path) -> Optional[Path]:
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        p = base / name
        if p.is_file():
            return p
    return None


def _find_llama_cpp_converter() -> Optional[Path]:
    """Locate llama.cpp's convert_hf_to_gguf.py, if the user has llama.cpp around."""
    candidates: list[Path] = []
    env = os.getenv("LLAMA_CPP_DIR") or os.getenv("LLAMACPP_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        _MANAGED_LLAMA_DIR,
        Path.home() / "llama.cpp",
        Path.home() / "code" / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/usr/local/llama.cpp"),
    ]
    for base in candidates:
        found = _converter_in(base)
        if found:
            return found
    return None


def _ensure_gguf_dep(on_log: Logger = _noop) -> None:
    """The converter imports the `gguf` python package — install it if missing."""
    try:
        import gguf  # noqa: F401
        return
    except Exception:
        pass
    on_log("Installing the 'gguf' Python package…")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "gguf"], timeout=600)


def ensure_converter(on_log: Logger = _noop) -> Optional[Path]:
    """
    Return a path to llama.cpp's converter, installing llama.cpp on first use.

    Tries existing locations first; otherwise shallow-clones llama.cpp into
    ~/.llmtune/llama.cpp. Returns None only if git is unavailable / clone fails.
    """
    found = _find_llama_cpp_converter()
    if found:
        _ensure_gguf_dep(on_log)
        return found

    if not shutil.which("git"):
        return None

    try:
        if _MANAGED_LLAMA_DIR.exists() and not _converter_in(_MANAGED_LLAMA_DIR):
            shutil.rmtree(_MANAGED_LLAMA_DIR, ignore_errors=True)
        if not _MANAGED_LLAMA_DIR.exists():
            _MANAGED_LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
            on_log("Setting up llama.cpp (one-time, shallow clone)…")
            subprocess.run(
                ["git", "clone", "--depth", "1", _LLAMA_REPO, str(_MANAGED_LLAMA_DIR)],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
    except Exception as e:
        on_log(f"llama.cpp setup failed: {e}")
        return None

    _ensure_gguf_dep(on_log)
    return _converter_in(_MANAGED_LLAMA_DIR)


def export_gguf(
    model_dir: str,
    out_path: str,
    quant: str = "q8_0",
    hf_token: str | None = None,
    on_log: Logger = _noop,
) -> dict:
    """
    Merge (if needed) and convert a model to GGUF.

    Returns {"status": "ok"|"merged_only", "gguf": <path or "">,
             "merged": <merged dir>, "message": <str>}.

    GGUF conversion needs llama.cpp's converter. If it isn't found we still
    produce the merged HF model (the hard part) and explain how to finish.
    """
    merged_dir = str(Path(out_path).with_suffix("")) + "-merged"
    merge_adapter(model_dir, merged_dir, hf_token=hf_token, on_log=on_log)

    converter = ensure_converter(on_log)
    if converter is None:
        return {
            "status": "merged_only",
            "gguf": "",
            "merged": merged_dir,
            "message": (
                "Merged model is ready, but llama.cpp couldn't be set up automatically "
                "(git may be missing). Install git, or clone llama.cpp manually and run: "
                f"python convert_hf_to_gguf.py {merged_dir} --outfile {out_path} --outtype {quant}"
            ),
        }

    on_log(f"Converting to GGUF with {converter.name} ({quant})…")
    # Make the repo's gguf-py importable in case the pip package isn't present.
    env = os.environ.copy()
    gguf_py = converter.parent / "gguf-py"
    if gguf_py.is_dir():
        env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(converter), merged_dir, "--outfile", out_path, "--outtype", quant]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    except Exception as e:
        return {"status": "merged_only", "gguf": "", "merged": merged_dir, "message": f"Converter failed to run: {e}"}

    if proc.returncode != 0 or not Path(out_path).is_file():
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return {"status": "merged_only", "gguf": "", "merged": merged_dir, "message": f"GGUF conversion failed: {tail}"}

    # Clean up the intermediate merged model to save disk.
    shutil.rmtree(merged_dir, ignore_errors=True)
    on_log(f"GGUF written → {out_path}")
    return {"status": "ok", "gguf": out_path, "merged": "", "message": f"GGUF model written to {out_path}"}
