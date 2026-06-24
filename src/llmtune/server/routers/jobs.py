"""Training job API and WebSocket streaming."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from llmtune.server.jobs import JobStatus, job_manager
from llmtune.training.config import TrainConfig
from llmtune.training.image_config import ImageTrainConfig

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobRequest(BaseModel):
    mode: Literal["llm", "cnn"] = "llm"
    model_id: str
    hf_token: str | None = None
    dataset_path: str
    dataset_format: Literal["json", "jsonl", "csv", "text"] = "jsonl"
    text_column: str = "text"
    max_seq_length: int = 2048
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    warmup_steps: int = 50
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    quantization: Literal["none", "4bit", "8bit"] = "none"
    training_mode: Literal["feature_extraction", "full", "lora"] = "feature_extraction"
    lr_scheduler_type: str = "cosine"
    validation_split: float = 0.0
    pack_sequences: bool = False
    shuffle: bool = True
    output_dir: str = ""
    save_checkpoints: bool = True
    eval_on_validation: bool = True
    resume_from_checkpoint: bool = False
    push_to_hub: bool = False
    hub_model_id: str | None = None


class JobResponse(BaseModel):
    id: str
    mode: str
    status: str
    step: int = 0
    total_steps: int = 0
    loss: float = -1.0
    phase: str = "loading"
    device: str = ""
    output_path: str = ""
    error: str = ""
    logs: list[str] = Field(default_factory=list)


def _state_to_response(state) -> JobResponse:
    return JobResponse(
        id=state.id,
        mode=state.mode,
        status=state.status.value,
        step=state.step,
        total_steps=state.total_steps,
        loss=state.loss,
        phase=state.phase,
        device=state.device,
        output_path=state.output_path,
        error=state.error,
        logs=state.logs[-100:],
    )


@router.post("", response_model=JobResponse)
def create_job(body: JobRequest):
    ds_path = Path(body.dataset_path).expanduser()
    if body.mode == "cnn":
        if not ds_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset folder not found: {ds_path}. Use a folder with class subfolders (cats/, dogs/).",
            )
    else:
        p = body.dataset_path.strip()
        is_hf = (
            p
            and p[0] not in "/~."
            and len(p.split("/")) == 2
            and not (len(p) > 1 and p[1] == ":")
        )
        if not is_hf and not ds_path.exists():
            raise HTTPException(status_code=400, detail=f"Dataset not found: {ds_path}")

    if body.mode == "cnn":
        output = body.output_dir or str(Path.home() / "llmtune-cnn-output")
        config = ImageTrainConfig(
            model_id=body.model_id,
            hf_token=body.hf_token,
            dataset_path=body.dataset_path,
            training_mode=body.training_mode,
            num_epochs=body.num_epochs,
            per_device_train_batch_size=body.per_device_train_batch_size,
            learning_rate=body.learning_rate,
            lora_r=body.lora_r,
            lora_alpha=body.lora_alpha,
            lora_dropout=body.lora_dropout,
            output_dir=output,
        )
        job_id = job_manager.create_cnn_job(config)
    else:
        output = body.output_dir or str(Path.home() / "llmtune-output")
        config = TrainConfig(
            model_id=body.model_id,
            hf_token=body.hf_token,
            dataset_path=body.dataset_path,
            dataset_format=body.dataset_format,
            text_column=body.text_column,
            max_seq_length=body.max_seq_length,
            num_epochs=body.num_epochs,
            per_device_train_batch_size=body.per_device_train_batch_size,
            gradient_accumulation_steps=body.gradient_accumulation_steps,
            learning_rate=body.learning_rate,
            warmup_ratio=body.warmup_ratio,
            lora_r=body.lora_r,
            lora_alpha=body.lora_alpha,
            lora_dropout=body.lora_dropout,
            target_modules=body.target_modules,
            quantization=body.quantization,
            lr_scheduler_type=body.lr_scheduler_type,
            output_dir=output,
            resume_from_checkpoint=body.resume_from_checkpoint,
            push_to_hub=body.push_to_hub,
            hub_model_id=body.hub_model_id,
        )
        job_id = job_manager.create_llm_job(config)

    state = job_manager.get(job_id)
    if not state:
        raise HTTPException(status_code=500, detail="Failed to create job")
    return _state_to_response(state)



@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    state = job_manager.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return _state_to_response(state)


@router.post("/{job_id}/stop")
def stop_job(job_id: str):
    if not job_manager.stop(job_id):
        raise HTTPException(status_code=404, detail="Job not found or not running")
    state = job_manager.get(job_id)
    return {"status": state.status.value if state else "stopped"}


@router.websocket("/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    state = job_manager.get(job_id)
    if not state:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    queue = job_manager.subscribe(job_id)

    await websocket.send_json({
        "type": "snapshot",
        "status": state.status.value,
        "step": state.step,
        "total": state.total_steps,
        "loss": state.loss,
        "phase": state.phase,
        "device": state.device,
        "logs": state.logs[-50:],
        "output_path": state.output_path,
        "error": state.error,
    })

    try:
        while True:
            if state.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED):
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(event)
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, queue)
