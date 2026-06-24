"""Model-library actions: test (inference), push to HuggingFace Hub, GGUF export.

These operate on fine-tuned models in the local output folders. They load models
into memory and can take a while; FastAPI runs these sync handlers in a threadpool
so they don't block other requests.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/library", tags=["library"])

# Cache the last-loaded inference model so repeated prompts are fast.
_infer_cache: dict = {"path": None, "model": None, "tokenizer": None, "device": None}


def _check_dir(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"Model folder not found: {p}")
    return p


class InferRequest(BaseModel):
    path: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    hf_token: str | None = None


@router.post("/infer")
def infer(body: InferRequest):
    """Run a quick generation against a fine-tuned model to sanity-check it."""
    model_dir = _check_dir(body.path)
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from llmtune.training.device_utils import detect_device
        from llmtune.training.merge_utils import is_adapter, read_base_model_id

        if (model_dir / "label_map.json").is_file() and not is_adapter(model_dir):
            raise HTTPException(status_code=400, detail="This is an image model — text testing isn't supported.")

        device = detect_device()
        if _infer_cache["path"] != str(model_dir):
            tok_kwargs: dict = {"trust_remote_code": True}
            load_kwargs: dict = {"trust_remote_code": True, "torch_dtype": torch.float32 if device == "mps" else torch.float16}
            if body.hf_token:
                tok_kwargs["token"] = body.hf_token
                load_kwargs["token"] = body.hf_token

            if is_adapter(model_dir):
                from peft import PeftModel

                base_id = read_base_model_id(model_dir)
                if not base_id:
                    raise HTTPException(status_code=400, detail="Adapter is missing its base model id.")
                base = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
                model = PeftModel.from_pretrained(base, str(model_dir))
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **tok_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **tok_kwargs)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            if device in ("mps", "cuda"):
                model = model.to(device)
            _infer_cache.update(path=str(model_dir), model=model, tokenizer=tokenizer, device=device)
        else:
            model = _infer_cache["model"]
            tokenizer = _infer_cache["tokenizer"]

        # Format like training did (Dolly-style instruction prompt).
        prompt = f"### Instruction:\n{body.prompt}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max(16, min(body.max_new_tokens, 1024)),
                do_sample=body.temperature > 0,
                temperature=max(0.01, body.temperature),
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"output": text.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e


class PushRequest(BaseModel):
    path: str
    repo_id: str
    token: str
    private: bool = True


@router.post("/push")
def push(body: PushRequest):
    """Upload a fine-tuned model folder to the HuggingFace Hub."""
    model_dir = _check_dir(body.path)
    if "/" not in body.repo_id.strip():
        raise HTTPException(status_code=400, detail="Repo id must look like username/model-name.")
    if not body.token.strip():
        raise HTTPException(status_code=400, detail="A HuggingFace token with write access is required.")
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=body.token.strip())
        api.create_repo(body.repo_id.strip(), exist_ok=True, private=body.private)
        api.upload_folder(folder_path=str(model_dir), repo_id=body.repo_id.strip())
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "authoriz" in msg.lower():
            raise HTTPException(status_code=401, detail="HuggingFace rejected the token — it needs write access.") from e
        raise HTTPException(status_code=502, detail=f"Push failed: {msg[:300]}") from e
    return {"status": "ok", "url": f"https://huggingface.co/{body.repo_id.strip()}"}


class ExportRequest(BaseModel):
    path: str
    quant: str = "q8_0"
    hf_token: str | None = None


@router.post("/export-gguf")
def export_gguf_endpoint(body: ExportRequest):
    """Merge the adapter into the base model and convert to GGUF (for Ollama/llama.cpp)."""
    model_dir = _check_dir(body.path)
    out_path = str(model_dir) + f".{body.quant}.gguf"
    try:
        from llmtune.training.merge_utils import export_gguf

        result = export_gguf(str(model_dir), out_path, quant=body.quant, hf_token=body.hf_token)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e
