"""Training job manager with WebSocket event broadcasting."""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llmtune.training.config import TrainConfig
from llmtune.training.image_config import ImageTrainConfig


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class JobState:
    id: str
    mode: str  # "llm" | "cnn"
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    step: int = 0
    total_steps: int = 0
    loss: float = -1.0
    output_path: str = ""
    error: str = ""
    phase: str = "loading"
    device: str = ""


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._tuners: dict[str, Any] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def create_llm_job(self, config: TrainConfig) -> str:
        job_id = str(uuid.uuid4())
        state = JobState(id=job_id, mode="llm")
        self._jobs[job_id] = state
        self._start_llm(job_id, config)
        return job_id

    def create_cnn_job(self, config: ImageTrainConfig) -> str:
        job_id = str(uuid.uuid4())
        state = JobState(id=job_id, mode="cnn")
        self._jobs[job_id] = state
        self._start_cnn(job_id, config)
        return job_id

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def stop(self, job_id: str) -> bool:
        tuner = self._tuners.get(job_id)
        state = self._jobs.get(job_id)
        if not tuner or not state:
            return False
        tuner.stop()
        if state.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            state.status = JobStatus.STOPPED
            state.phase = "stopped"
            state.logs.append("Stop requested — halting…")
            self._broadcast(job_id, {"type": "log", "message": "Stop requested — halting…"})
            self._broadcast(job_id, {"type": "status", "status": "stopped"})
        return True

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)

    def _broadcast(self, job_id: str, event: dict) -> None:
        for q in self._subscribers.get(job_id, []):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def _on_log(self, job_id: str, msg: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            state.logs.append(msg)
            if len(state.logs) > 500:
                state.logs = state.logs[-500:]
            if msg.startswith("Device:"):
                state.device = msg.replace("Device:", "").strip()
            lower = msg.lower()
            if "not in local cache" in lower or "downloading" in lower:
                state.phase = "downloading"
            elif "download complete" in lower or "loading model" in lower or "loading tokenizer" in lower:
                state.phase = "loading"
            elif "loading dataset" in lower or "scanning image" in lower or "preprocessing" in lower:
                state.phase = "dataset"
            elif "starting training" in lower:
                state.phase = "training"
                state.status = JobStatus.RUNNING
        self._broadcast(job_id, {"type": "log", "message": msg})

    def _on_progress(self, job_id: str, step: int, total: int, loss: float) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            state.step = step
            state.total_steps = total
            # loss = -1 is a step-only tick — keep the last real loss.
            if loss >= 0:
                state.loss = loss
                state.phase = "training"
                state.status = JobStatus.RUNNING
        self._broadcast(job_id, {
            "type": "progress",
            "step": step,
            "total": total,
            "loss": loss,
        })

    def _on_done(self, job_id: str, path: str) -> None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state:
                state.status = JobStatus.COMPLETED
                state.output_path = path
                state.phase = "done"
        self._broadcast(job_id, {"type": "done", "path": path})

    def _on_error(self, job_id: str, error: Exception) -> None:
        msg = str(error)
        with self._lock:
            state = self._jobs.get(job_id)
            if state:
                state.status = JobStatus.FAILED
                state.error = msg
                state.phase = "error"
        self._broadcast(job_id, {"type": "error", "message": msg})

    def _start_llm(self, job_id: str, config: TrainConfig) -> None:
        from llmtune.training import FineTuner

        tuner = FineTuner(
            config=config,
            on_log=lambda m: self._on_log(job_id, m),
            on_progress=lambda s, t, l: self._on_progress(job_id, s, t, l),
            on_done=lambda p: self._on_done(job_id, p),
            on_error=lambda e: self._on_error(job_id, e),
        )
        self._tuners[job_id] = tuner
        with self._lock:
            state = self._jobs[job_id]
            state.status = JobStatus.RUNNING
        tuner.start()

    def _start_cnn(self, job_id: str, config: ImageTrainConfig) -> None:
        from llmtune.training.image_trainer import ImageFineTuner

        tuner = ImageFineTuner(
            config=config,
            on_log=lambda m: self._on_log(job_id, m),
            on_progress=lambda s, t, l: self._on_progress(job_id, s, t, l),
            on_done=lambda p: self._on_done(job_id, p),
            on_error=lambda e: self._on_error(job_id, e),
        )
        self._tuners[job_id] = tuner
        with self._lock:
            state = self._jobs[job_id]
            state.status = JobStatus.RUNNING
        tuner.start()


job_manager = JobManager()
