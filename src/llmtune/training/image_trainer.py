"""CNN / image-classification fine-tuning.

Supports any HuggingFace AutoModelForImageClassification model:
  ResNet, EfficientNet, ConvNeXt, ViT, DeiT, Swin, and more.

Three training modes:
  feature_extraction — freeze backbone, train classifier head only   (fastest)
  full               — train all layers with a lower learning rate   (most accurate)
  lora               — LoRA adapters on attention (ViT / Swin models)(efficient)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from .device_utils import detect_device, safe_save_trainer
from .image_config import ImageTrainConfig


def _processor_size_kw(processor: Any, image_size: int) -> dict[str, int]:
    """Build a size dict that matches what this processor expects (shortest_edge vs H×W)."""
    raw = processor.size
    if hasattr(raw, "shortest_edge") and raw.shortest_edge is not None:
        return {"shortest_edge": image_size}
    if hasattr(raw, "height") and raw.height is not None:
        return {"height": image_size, "width": image_size}
    if isinstance(raw, dict):
        if raw.get("shortest_edge") is not None:
            return {"shortest_edge": image_size}
        if raw.get("height") is not None:
            return {"height": image_size, "width": image_size}
    # Safe default for ResNet / ConvNeXt style processors
    return {"shortest_edge": image_size}


class ImageFineTuner:
    """Runs CNN image-classification fine-tuning in a background thread."""

    def __init__(
        self,
        config: ImageTrainConfig,
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
        import json
        import os

        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            Trainer,
            TrainingArguments,
            TrainerCallback,
            default_data_collator,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        cfg = self.config
        device = detect_device()
        self.on_log(f"Device: {device.upper()}")

        # ── Image processor ───────────────────────────────────────────────────
        self.on_log(f"Loading image processor: {cfg.model_id}")
        proc_kwargs: dict = {"trust_remote_code": True}
        if cfg.hf_token:
            proc_kwargs["token"] = cfg.hf_token
        processor = AutoImageProcessor.from_pretrained(cfg.model_id, **proc_kwargs)
        size_kw = _processor_size_kw(processor, cfg.image_size)
        self.on_log(f"Image size: {size_kw}")

        # ── Dataset ───────────────────────────────────────────────────────────
        data_dir = str(Path(cfg.dataset_path).expanduser())
        if not Path(data_dir).is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {data_dir}\n"
                "Expected a folder with subfolders per class (e.g. cats/, dogs/)."
            )
        if self._stop.is_set():
            self.on_log("Stopped before dataset load.")
            return

        self.on_log(f"Loading dataset: {data_dir}")
        self.on_log("Scanning image folders…")
        raw = load_dataset("imagefolder", data_dir=data_dir)
        if self._stop.is_set():
            self.on_log("Stopped after dataset scan.")
            return

        if "train" in raw:
            train_ds = raw["train"]
            eval_ds = raw.get("validation") or raw.get("test")
            if eval_ds is None:
                self.on_log(
                    f"No validation split — holding out {int(cfg.val_split * 100)}% for validation"
                )
                split = train_ds.train_test_split(test_size=cfg.val_split, seed=42)
                train_ds = split["train"]
                eval_ds = split["test"]
        else:
            first = list(raw.values())[0]
            split = first.train_test_split(test_size=cfg.val_split, seed=42)
            train_ds = split["train"]
            eval_ds = split["test"]

        label_feat = train_ds.features.get("label")
        if label_feat and hasattr(label_feat, "names"):
            class_names: list[str] = label_feat.names
        else:
            num = max(train_ds["label"]) + 1
            class_names = [str(i) for i in range(num)]
        num_labels = len(class_names)

        preview = ", ".join(class_names[:6]) + ("…" if num_labels > 6 else "")
        self.on_log(f"Classes ({num_labels}): {preview}")
        self.on_log(f"Train: {len(train_ds)}  |  Val: {len(eval_ds)}")

        # ── Preprocess (single-process, one image at a time for Trainer compat) ─
        n_train, n_val = len(train_ds), len(eval_ds)
        self.on_log(f"Preprocessing {n_train + n_val} images — may take several minutes…")

        def preprocess(example: dict, idx: int | None = None) -> dict:
            if self._stop.is_set():
                return {"pixel_values": torch.zeros(3, cfg.image_size, cfg.image_size), "labels": 0}
            image = example["image"].convert("RGB")
            out = processor(image, size=size_kw, return_tensors="pt")
            return {
                "pixel_values": out["pixel_values"][0],
                "labels": example["label"],
            }

        def _map_with_progress(ds, name: str, total: int):
            last_pct = -1

            def _fn(example, idx):
                nonlocal last_pct
                if self._stop.is_set():
                    return preprocess(example)
                result = preprocess(example)
                if total > 0:
                    pct = int(100 * (idx + 1) / total)
                    if pct >= last_pct + 10:
                        last_pct = pct
                        self.on_log(f"{name}: {pct}% ({idx + 1}/{total})")
                        self.on_progress(pct, 100, -1.0)
                return result

            return ds.map(
                _fn,
                with_indices=True,
                remove_columns=ds.column_names,
                desc=name,
            )

        train_ds = _map_with_progress(train_ds, "Preprocessing train", n_train)
        if self._stop.is_set():
            self.on_log("Stopped during preprocessing.")
            return
        eval_ds = _map_with_progress(eval_ds, "Preprocessing val", n_val)
        if self._stop.is_set():
            self.on_log("Stopped during preprocessing.")
            return

        train_ds.set_format("torch")
        eval_ds.set_format("torch")
        self.on_log("Preprocessing complete.")
        self.on_progress(0, 100, -1.0)

        # ── Model ─────────────────────────────────────────────────────────────
        self.on_log(f"Loading model: {cfg.model_id}")
        id2label = {i: n for i, n in enumerate(class_names)}
        label2id = {n: i for i, n in enumerate(class_names)}
        model_kwargs: dict = {
            "trust_remote_code": True,
            "num_labels": num_labels,
            "id2label": id2label,
            "label2id": label2id,
            "ignore_mismatched_sizes": True,
        }
        if cfg.hf_token:
            model_kwargs["token"] = cfg.hf_token

        model = AutoModelForImageClassification.from_pretrained(cfg.model_id, **model_kwargs)

        if cfg.training_mode == "feature_extraction":
            self.on_log("Mode: feature extraction — backbone frozen, head only")
            for param in model.parameters():
                param.requires_grad = False
            head_keywords = ["classifier", "head", "fc", "cls_classifier", "distillation_classifier"]
            for name, param in model.named_parameters():
                if any(kw in name.lower() for kw in head_keywords):
                    param.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_p = sum(p.numel() for p in model.parameters())
            self.on_log(f"Trainable params: {trainable:,} / {total_p:,}  ({100 * trainable / total_p:.2f}%)")

        elif cfg.training_mode == "lora":
            self.on_log("Mode: LoRA — adapter layers on attention (ViT / Swin)")
            from peft import LoraConfig, TaskType, get_peft_model
            lora_cfg = LoraConfig(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            trainable, total_p = model.get_nb_trainable_parameters()
            self.on_log(f"Trainable params: {trainable:,} / {total_p:,}  ({100 * trainable / total_p:.2f}%)")

        else:
            self.on_log("Mode: full fine-tuning — all layers trained")
            total_p = sum(p.numel() for p in model.parameters())
            self.on_log(f"Total params: {total_p:,}")

        if device in ("mps", "cuda"):
            model = model.to(device)

        steps_per_epoch = -(-len(train_ds) // cfg.per_device_train_batch_size)  # ceil div
        total_steps = max(1, steps_per_epoch * cfg.num_epochs)
        stop_event = self._stop

        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if stop_event.is_set():
                    control.should_training_stop = True
                    return
                loss = (logs or {}).get("loss", 0.0)
                acc = (logs or {}).get("eval_accuracy")
                self._on_progress(state.global_step, total_steps, loss)
                line = f"step {state.global_step}/{total_steps}  loss={loss:.4f}"
                if acc is not None:
                    line += f"  val_acc={acc:.3f}"
                self._on_log(line)

        cb = _CB()
        cb._on_log = self.on_log
        cb._on_progress = self.on_progress

        output_dir = str(Path(cfg.output_dir).expanduser())
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        effective_lr = cfg.learning_rate if cfg.training_mode != "full" else min(cfg.learning_rate, 5e-5)

        import inspect
        _ta_sig = inspect.signature(TrainingArguments.__init__).parameters
        _eval_key = "eval_strategy" if "eval_strategy" in _ta_sig else "evaluation_strategy"

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            learning_rate=effective_lr,
            lr_scheduler_type=cfg.lr_scheduler_type,
            fp16=False,
            bf16=False,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_strategy="no",
            **{_eval_key: "epoch"},
            load_best_model_at_end=False,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": float((preds == labels).mean())}

        self.on_log("Starting training…")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[cb],
        )
        trainer.train()

        if self._stop.is_set():
            self.on_log("Training stopped by user.")
            return

        self.on_log("Training finished — saving model…")
        self.on_log(f"Output → {output_dir}")

        # Save lightweight files first.
        label_map = {"id2label": id2label, "label2id": label2id, "num_labels": num_labels}
        (Path(output_dir) / "label_map.json").write_text(json.dumps(label_map, indent=2))
        processor.save_pretrained(output_dir)
        self.on_log("Processor + label map saved.")

        safe_save_trainer(trainer, output_dir, device)
        self.on_log("Model weights saved.")

        self.on_done(output_dir)
