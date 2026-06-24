"""End-to-end fine-tune test on M1 Mac.

Model  : TinyLlama/TinyLlama-1.1B-Chat-v1.0  (~2.2 GB float16)
Dataset: databricks/databricks-dolly-15k      (real, 15k instructions)
Subset : 200 examples  (fast smoke test)
Device : MPS (Apple Silicon) → auto-detected
"""

import sys
import time
sys.path.insert(0, "src")

from llmtune.training.config import TrainConfig
from llmtune.training.trainer import FineTuner

logs = []
steps = []

def on_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    logs.append(line)

def on_progress(step: int, total: int, loss: float):
    steps.append({"step": step, "loss": loss})

def on_done(path: str):
    print(f"\n✅  Fine-tune complete. Adapter saved to: {path}", flush=True)

def on_error(e: Exception):
    print(f"\n❌  Error: {e}", flush=True)
    import traceback; traceback.print_exc()

cfg = TrainConfig(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization="none",          # MPS doesn't support bitsandbytes
    dataset_path="databricks/databricks-dolly-15k",  # HF Hub — real dataset
    dataset_format="jsonl",
    text_column="text",
    max_seq_length=256,           # keep memory low on M1
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    num_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    output_dir="./test-output",
)

# Monkey-patch: use only first 200 rows so the smoke test finishes quickly
import llmtune.training.trainer as _t
_original_load = _t._load_raw_dataset

def _patched_load(cfg):
    ds = _original_load(cfg)
    print(f"[dataset] Full size: {len(ds)} rows — using first 200 for smoke test", flush=True)
    return ds.select(range(200))

_t._load_raw_dataset = _patched_load

print("=" * 60)
print("llmtune — fine-tune smoke test")
print("=" * 60)

tuner = FineTuner(cfg, on_log=on_log, on_progress=on_progress,
                  on_done=on_done, on_error=on_error)
tuner.start()
tuner.join()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
if steps:
    first_loss = steps[0]["loss"]
    last_loss  = steps[-1]["loss"]
    print(f"  Steps logged : {len(steps)}")
    print(f"  First loss   : {first_loss:.4f}")
    print(f"  Final loss   : {last_loss:.4f}")
    improved = last_loss < first_loss
    print(f"  Loss trend   : {'↓ improving ✅' if improved else '→ flat / ↑ check config'}")
else:
    print("  No steps recorded — check logs above.")
