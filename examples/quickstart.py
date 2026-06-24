"""Quickstart: fine-tune a tiny LLM on the bundled sample dataset.

Runs entirely locally. On Apple Silicon it uses float16 LoRA on the MPS GPU;
on CPU/CUDA it adapts automatically. Expect a few minutes on a laptop.

    python examples/quickstart.py
"""

from pathlib import Path

from llmtune.training.config import TrainConfig
from llmtune.training.trainer import FineTuner

# The repo ships a 20-row instruction/response dataset you can train on directly.
SAMPLE_DATASET = Path(__file__).resolve().parent.parent / "sample_dataset.jsonl"


def main() -> None:
    cfg = TrainConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_path=str(SAMPLE_DATASET),
        dataset_format="jsonl",
        quantization="none",          # 4bit/8bit are CUDA-only; "none" works everywhere
        num_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=256,
        output_dir="./quickstart-adapter",
    )

    tuner = FineTuner(
        cfg,
        on_log=lambda msg: print(msg, flush=True),
        on_progress=lambda step, total, loss: None,
        on_done=lambda path: print(f"\n✅ Adapter saved to: {path}"),
        on_error=lambda e: print(f"\n❌ {e}"),
    )
    tuner.start()
    tuner.join()


if __name__ == "__main__":
    main()
