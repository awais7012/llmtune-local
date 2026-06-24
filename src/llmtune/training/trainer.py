"""LoRA fine-tuning via HuggingFace PEFT + TRL SFTTrainer.

M1/M2 Mac note: bitsandbytes 4-bit/8-bit quantization does NOT work on MPS.
We auto-detect MPS and skip quantization automatically on Apple Silicon.
Regular float16 LoRA works fine with unified memory.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from .config import TrainConfig


class _ProgressTqdm:
    """Tqdm-compatible shim that aggregates download bytes and reports to the UI."""

    _lock: threading.Lock = threading.Lock()
    _total_bytes: int = 0
    _done_bytes: int = 0
    _last_log_at: float = 0.0
    _start_time: float = 0.0
    _on_progress: Callable | None = None
    _on_log: Callable | None = None

    @classmethod
    def _reset(cls, on_progress: Callable, on_log: Callable) -> None:
        import time
        with cls._lock:
            cls._total_bytes = 0
            cls._done_bytes = 0
            cls._last_log_at = 0.0
            cls._start_time = time.time()
            cls._on_progress = on_progress
            cls._on_log = on_log

    @classmethod
    def _clear(cls) -> None:
        with cls._lock:
            cls._on_progress = None
            cls._on_log = None

    def __init__(self, *args, **kwargs) -> None:
        self.total: int = int(kwargs.get("total") or 0)
        self.unit: str = str(kwargs.get("unit") or "")
        self.n: int = 0
        if self.unit == "B" and self.total > 0:
            with type(self)._lock:
                type(self)._total_bytes += self.total

    def update(self, n: int = 1) -> bool:
        self.n += n
        if self.unit != "B":
            return True
        import time
        cls = type(self)
        now = time.time()
        with cls._lock:
            cls._done_bytes += n
            done = cls._done_bytes
            total = cls._total_bytes
            start = cls._start_time
            on_progress = cls._on_progress
            on_log = cls._on_log
            should_log = on_log is not None and (now - cls._last_log_at) >= 1.5
            if should_log:
                cls._last_log_at = now
        if total <= 0 or on_progress is None:
            return True
        pct = min(99, int(100 * done / total))
        on_progress(pct, 100, -1.0)
        if should_log and on_log:
            dl_mb = done / 1_048_576
            tot_mb = total / 1_048_576
            elapsed = now - start
            speed = (done / 1_048_576) / elapsed if elapsed > 0.5 else 0
            speed_str = f"  •  {speed:.1f} MB/s" if speed > 0 else ""
            on_log(f"Downloading: {dl_mb:.0f} MB / {tot_mb:.0f} MB  ({pct}%){speed_str}")
        return True

    def __enter__(self) -> "_ProgressTqdm":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        pass

    def set_postfix(self, *_, **__) -> None:
        pass

    def set_description(self, *_, **__) -> None:
        pass

    def write(self, s: str = "", **__) -> None:
        pass

    def reset(self, total: int | None = None) -> bool:
        return True


def _detect_device() -> str:
    from .device_utils import detect_device
    return detect_device()


def _load_raw_dataset(cfg: TrainConfig):
    from datasets import load_dataset

    path = cfg.dataset_path.strip()
    token = cfg.hf_token or None
    if not Path(path).exists():
        # Treat as HuggingFace Hub dataset id
        return load_dataset(path, split="train", token=token)
    ext = Path(path).suffix.lower()
    fmt = {".jsonl": "json", ".json": "json", ".csv": "csv", ".txt": "text"}.get(ext, "json")
    return load_dataset(fmt, data_files=path, split="train")


def _format_example(example: dict, text_col: str) -> dict:
    # Dolly-style: instruction + optional context + response
    if "instruction" in example and "response" in example:
        ctx = (example.get("context") or "").strip()
        body = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Context:\n{ctx}\n\n### Response:\n{example['response']}"
            if ctx
            else f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        )
        return {"text": body}
    return {"text": str(example.get(text_col, ""))}


class FineTuner:
    """Runs LoRA fine-tuning in a background thread."""

    def __init__(
        self,
        config: TrainConfig,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int, float], None] | None = None,
        on_done: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self.config = config
        self.on_log = on_log or (lambda msg: None)
        self.on_progress = on_progress or (lambda step, total, loss: None)
        self.on_done = on_done or (lambda path: None)
        self.on_error = on_error or (lambda e: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        if self._thread:
            self._thread.join()

    def _run(self) -> None:
        import traceback as _tb
        try:
            self._train()
        except Exception as e:
            self.on_log("━" * 40)
            self.on_log(_tb.format_exc())
            self.on_error(e)

    def _train(self) -> None:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer

        import os
        # Disable tokenizer + dataset multiprocessing — avoids fds_to_keep errors
        # when running inside Textual's event loop (Textual holds open terminal FDs
        # that are invalid in forked child processes on Python 3.12+).
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        cfg = self.config
        device = _detect_device()
        on_mps = device == "mps"

        self.on_log(f"Device: {device.upper()}")

        # ── Download from HuggingFace if model not in local cache ──────
        is_local = Path(cfg.model_id).exists()
        if not is_local:
            try:
                from huggingface_hub import try_to_load_from_cache

                # Check for the actual WEIGHTS, not just config.json — a previous
                # interrupted run can leave config.json cached while the weights
                # are still missing. If we only checked config.json we'd skip our
                # progress-reporting download and let from_pretrained pull the
                # weights with a bar that never reaches the UI (looks "stuck").
                weight_files = [
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                    "pytorch_model.bin.index.json",
                ]
                has_weights = any(
                    try_to_load_from_cache(cfg.model_id, w) for w in weight_files
                )
                needs_download = not has_weights
            except Exception:
                needs_download = True

            if needs_download:
                self.on_log("Model not in local cache — starting download from HuggingFace…")
                try:
                    from huggingface_hub import snapshot_download
                    try:
                        from huggingface_hub.utils import HfHubHTTPError as _HfHTTPErr
                    except ImportError:
                        _HfHTTPErr = None  # type: ignore
                    _ProgressTqdm._reset(self.on_progress, self.on_log)
                    dl_kwargs: dict = {}
                    if cfg.hf_token:
                        dl_kwargs["token"] = cfg.hf_token
                    snapshot_download(cfg.model_id, tqdm_class=_ProgressTqdm, **dl_kwargs)
                    self.on_log("Download complete — loading model into memory…")
                    self.on_progress(100, 100, -1.0)
                except Exception as e:
                    # Specific HTTP status handling via HfHubHTTPError
                    if _HfHTTPErr and isinstance(e, _HfHTTPErr):
                        status = e.response.status_code
                        tok_preview = (
                            f" (token: {cfg.hf_token[:8]}…{cfg.hf_token[-4:]})"
                            if cfg.hf_token and len(cfg.hf_token) > 12
                            else (" (token provided)" if cfg.hf_token else " (no token)")
                        )
                        if status == 401:
                            raise RuntimeError(
                                f"Authentication failed{tok_preview}. "
                                "Your token is invalid or expired. "
                                "Go to huggingface.co/settings/tokens, create a new token with Read permission, "
                                "and paste it on the Model screen."
                            ) from e
                        if status == 403:
                            raise RuntimeError(
                                f"Access denied for '{cfg.model_id}'. "
                                "Your IP may be blocked by HuggingFace, or this model requires special approval. "
                                "Fix: try a different network (phone hotspot), or download the model manually — "
                                f"git clone https://huggingface.co/{cfg.model_id} — then paste the folder path."
                            ) from e
                        if status == 429:
                            hint = (
                                "Your token may have hit its rate limit."
                                if cfg.hf_token
                                else "Add a HuggingFace token on the Model screen to bypass rate limits."
                            )
                            raise RuntimeError(
                                f"Rate limited by HuggingFace (429). {hint} "
                                "Wait 1 hour and try again, or use a different network."
                            ) from e
                        if status == 404:
                            raise RuntimeError(
                                f"Model '{cfg.model_id}' not found on HuggingFace (404). "
                                "Check the model name spelling — format is: username/model-name. "
                                "Browse available models at huggingface.co/models"
                            ) from e
                        raise RuntimeError(
                            f"Download failed (HTTP {status}) for '{cfg.model_id}'. "
                            f"Details: {str(e)[:200]}"
                        ) from e
                    # Fallback: string-pattern matching for non-HTTP exceptions
                    es, el = str(e), str(e).lower()
                    if "429" in es or "rate limit" in el or "too many requests" in el:
                        hint = (
                            "Try again in a few hours, or use a local model path."
                            if cfg.hf_token
                            else "Adding your HuggingFace token on the Model screen can bypass this limit."
                        )
                        raise RuntimeError(
                            f"Download blocked — too many requests (your IP may be rate-limited). {hint}"
                        ) from e
                    if "401" in es or "403" in es or "gated" in el:
                        tok_preview = (
                            f" (token: {cfg.hf_token[:8]}…{cfg.hf_token[-4:]})"
                            if cfg.hf_token and len(cfg.hf_token) > 12
                            else ""
                        )
                        hint = (
                            f"Check that your token{tok_preview} is correct and has access to this model."
                            if cfg.hf_token
                            else "Add your HuggingFace token on the Model screen to access private or gated models."
                        )
                        raise RuntimeError(
                            f"Authentication failed for '{cfg.model_id}'. {hint}"
                        ) from e
                    if "connection" in el or "errno" in el or "nodename" in el or "network" in el:
                        raise RuntimeError(
                            "Could not connect to HuggingFace. "
                            "Check your internet connection, or use a local model path instead."
                        ) from e
                    raise
                finally:
                    _ProgressTqdm._clear()

        # ── Load tokenizer ────────────────────────────────────────────────────
        self.on_log(f"Loading tokenizer: {cfg.model_id}")
        tok_kwargs: dict = {"trust_remote_code": True}
        if cfg.hf_token:
            tok_kwargs["token"] = cfg.hf_token

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.on_log(f"Loading model: {cfg.model_id}")

        # float16 causes segfaults in PyTorch's MPS backend during training;
        # float32 is stable and runs at the same ~3 it/s on Apple Silicon.
        model_dtype = torch.float32 if on_mps else torch.float16
        load_kwargs: dict = {"trust_remote_code": True, "torch_dtype": model_dtype}
        if cfg.hf_token:
            load_kwargs["token"] = cfg.hf_token

        if not on_mps and cfg.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
        elif not on_mps and cfg.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            if on_mps and cfg.quantization != "none":
                self.on_log("Apple Silicon detected — using float16 LoRA")
            load_kwargs["device_map"] = None

        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
        except Exception as e:
            es = str(e)
            el = es.lower()
            if "429" in es or "rate limit" in el or "too many requests" in el:
                hint = (
                    "Try again in a few hours, or use a local model path."
                    if cfg.hf_token
                    else "Adding your HuggingFace token on the Model screen can bypass this limit."
                )
                raise RuntimeError(
                    f"Download blocked — too many requests (your IP may be rate-limited). {hint}"
                ) from e
            if "401" in es or "403" in es or "gated" in el:
                hint = (
                    "Check that your HuggingFace token is correct and has access to this model."
                    if cfg.hf_token
                    else "Add your HuggingFace token on the Model screen to access private or gated models."
                )
                raise RuntimeError(
                    f"Authentication failed for '{cfg.model_id}'. {hint}"
                ) from e
            if "does not appear to have a file named" in es or (
                ("pytorch_model.bin" in es or "model.safetensors" in es) and "does not appear" in es
            ):
                raise RuntimeError(
                    f"Model '{cfg.model_id}' is not saved on this machine and could not be downloaded. "
                    "Go back and paste the full local folder path to a model you already have downloaded, "
                    "or check your internet connection and try again."
                ) from e
            if "connection" in el or "errno" in el or "nodename" in el or "network" in el:
                raise RuntimeError(
                    "Could not connect to download the model. "
                    "Check your internet connection, or use a local model path instead."
                ) from e
            raise

        if on_mps:
            model = model.to("mps")

        model.enable_input_require_grads()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        self.on_log(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        self.on_log(f"Loading dataset: {cfg.dataset_path}")
        raw = _load_raw_dataset(cfg)
        ds = raw.map(
            lambda ex: _format_example(ex, cfg.text_column),
            remove_columns=raw.column_names,
            num_proc=1,
        )

        # Mirror HF's optimizer-step count: ceil(batches / grad_accum) per epoch.
        batches_per_epoch = -(-len(ds) // cfg.per_device_train_batch_size)  # ceil div
        steps_per_epoch = -(-batches_per_epoch // cfg.gradient_accumulation_steps)
        total_steps = max(1, steps_per_epoch * cfg.num_epochs)

        stop_event = self._stop

        class _CB(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                # Fires every step → keeps the UI step counter live (loss = -1
                # means "step-only", the UI keeps the last real loss value).
                if stop_event.is_set():
                    control.should_training_stop = True
                    return
                self._on_progress(state.global_step, total_steps, -1.0)  # type: ignore[attr-defined]

            def on_log(self, args, state, control, logs=None, **kw):
                if stop_event.is_set():
                    control.should_training_stop = True
                    return
                logs = logs or {}
                if "loss" in logs:
                    loss = logs["loss"]
                    self._on_progress(state.global_step, total_steps, loss)  # type: ignore[attr-defined]
                    self._on_log(f"step {state.global_step}/{total_steps}  loss={loss:.4f}")  # type: ignore[attr-defined]
                elif "train_loss" in logs:
                    # Final summary event has no per-step "loss" key — report the
                    # real training loss instead of a misleading 0.0000.
                    loss = logs["train_loss"]
                    self._on_progress(state.global_step, total_steps, loss)  # type: ignore[attr-defined]
                    self._on_log(f"step {state.global_step}/{total_steps}  final loss={loss:.4f}")  # type: ignore[attr-defined]

        cb = _CB()
        cb._on_log = self.on_log          # type: ignore[attr-defined]
        cb._on_progress = self.on_progress  # type: ignore[attr-defined]

        output_dir = str(Path(cfg.output_dir).expanduser())
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sft_cfg = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            fp16=False,
            bf16=False,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            max_length=cfg.max_seq_length,
            dataset_text_field="text",
            report_to="none",
            optim="adamw_torch",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )

        self.on_log("Starting training…")
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds,
            processing_class=tokenizer,
            callbacks=[cb],
        )

        # Resume from the last checkpoint if requested and one exists in output_dir.
        resume = False
        if cfg.resume_from_checkpoint:
            checkpoints = list(Path(output_dir).glob("checkpoint-*"))
            if checkpoints:
                resume = True
                self.on_log("Resuming from last saved checkpoint…")
            else:
                self.on_log("Resume requested, but no checkpoint found — starting fresh.")

        trainer.train(resume_from_checkpoint=resume)

        if not self._stop.is_set():
            from .device_utils import safe_save_trainer
            self.on_log(f"Saving adapter → {output_dir}")
            safe_save_trainer(trainer, output_dir, device)
            tokenizer.save_pretrained(output_dir)

            if cfg.push_to_hub and cfg.hub_model_id:
                try:
                    self.on_log(f"Pushing to HuggingFace Hub: {cfg.hub_model_id}…")
                    from huggingface_hub import HfApi

                    api = HfApi(token=cfg.hf_token or None)
                    api.create_repo(cfg.hub_model_id, exist_ok=True, private=True)
                    api.upload_folder(folder_path=output_dir, repo_id=cfg.hub_model_id)
                    self.on_log(f"Pushed to Hub ✓  https://huggingface.co/{cfg.hub_model_id}")
                except Exception as e:
                    self.on_log(f"Push to Hub failed: {e}")

            self.on_done(output_dir)
        else:
            self.on_log("Training stopped by user.")
