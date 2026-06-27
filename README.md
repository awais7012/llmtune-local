# llmtune

[![PyPI version](https://img.shields.io/pypi/v/llmtune-local.svg)](https://pypi.org/project/llmtune-local/)
[![Python versions](https://img.shields.io/pypi/pyversions/llmtune-local.svg)](https://pypi.org/project/llmtune-local/)
[![CI](https://github.com/awais7012/llmtune-local/actions/workflows/ci.yml/badge.svg)](https://github.com/awais7012/llmtune-local/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Fine-tune AI models (LLMs **and** image classifiers) on your own computer — no cloud, no subscription, no data leaving your machine.

```bash
pip install llmtune-local
llmtune run
```

> The PyPI package is **`llmtune-local`**; the command you run is **`llmtune`**.

---

## Screenshots

| Model picker | Training |
|---|---|
| ![Model selection](screenshots/model-picker.png) | ![Live training](screenshots/training.png) |

> _Screenshots live in [`screenshots/`](screenshots/). Drop in PNGs named
> `model-picker.png` and `training.png` to populate this section._

---

## What is this?

**llmtune** is a command-line tool that lets you take any open-source AI language model and train it further on your own data. This process is called *fine-tuning*.

Think of it like this: a language model (like Llama, Mistral, or Qwen) comes pre-trained on billions of words from the internet. It knows a lot about everything. But maybe you want it to:

- Answer questions specifically about your company's product
- Write in your brand's tone of voice
- Speak like a customer support agent trained on your FAQ
- Generate code in your specific codebase style
- Behave like a domain expert in your field

Instead of paying OpenAI or Anthropic, you take an open-source model and teach it yourself, using your own data. That's fine-tuning.

**llmtune makes that process as simple as pointing to a file.**

---

## Why "local"?

Everything runs 100% on your computer. That means:

- **Your data never leaves your machine.** No API calls, no cloud uploads, no third party sees it.
- **No ongoing cost.** You pay nothing per query, per token, or per training run.
- **You own the result.** The fine-tuned model is a file on your disk. Use it however you want.
- **Works offline.** Once the base model is downloaded, no internet is needed.

---

## Who is this for?

- **Developers** who want to customize an AI model without deep ML knowledge
- **Small teams** who can't afford enterprise AI costs but have their own data
- **Researchers** experimenting with model behavior
- **Hobbyists** who want to run their own AI locally
- **Companies** with sensitive data that must stay on-premises

---

## How is this different from Axolotl / LLaMA-Factory / Torchtune / TRL?

Those are excellent, powerful **research frameworks** — config-/CLI-driven, Linux + NVIDIA first, and aimed at ML engineers who want maximum control. llmtune is deliberately the opposite:

- **No config files, no ML expertise required.** Pick a model, point at a file, click train — in a browser UI or a terminal UI.
- **Runs on your laptop, including Apple Silicon.** float16 LoRA on MPS works out of the box; it transparently falls back when CUDA-only paths (4-bit QLoRA) aren't available.
- **Fully local & private.** Nothing about your data, models, or runs leaves the machine.
- **More than LLMs.** It also fine-tunes image classifiers (ResNet, ViT, ConvNeXt, …).

If you need DeepSpeed, multi-GPU sharding, or DPO/ORPO research pipelines, use one of the frameworks above. If you want the **fastest path from "a dataset" to "a working adapter" on the hardware you already own**, that's this.

| | **llmtune** | LLaMA-Factory | Axolotl | Torchtune |
|---|:---:|:---:|:---:|:---:|
| Zero-config (no YAML) | ✅ | ❌ | ❌ | ❌ |
| Terminal UI (TUI) | ✅ | ❌ | ❌ | ❌ |
| Browser GUI | ✅ | ✅ | ❌ | ❌ |
| First-class Apple Silicon (MPS) | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Image-classifier fine-tuning | ✅ | ❌ | ❌ | ❌ |
| One-command GGUF export | ✅ | ✅ | ⚠️ | ❌ |
| LoRA / QLoRA | ✅ | ✅ | ✅ | ✅ |
| Multi-GPU / DeepSpeed | ❌ | ✅ | ✅ | ✅ |
| Preference tuning (DPO/ORPO/…) | ❌ | ✅ | ✅ | ✅ |

> The frameworks above are more powerful for large-scale, multi-GPU, and research
> workflows — llmtune trades that for simplicity and local/laptop ergonomics.
> Comparison reflects each project's typical/default usage and may change; corrections welcome via an issue.

---

## How it works — plain English

1. **You pick a model.** You choose from a list of popular open-source models (TinyLlama, Llama 3, Mistral, Qwen, etc.) or point to one already on your disk.

2. **You provide a dataset.** This is a file — JSONL, JSON, CSV, or plain text — containing the examples you want the model to learn from. Each example is a piece of text: a question+answer pair, a document, a conversation, whatever you want the model to get good at.

3. **You set some numbers.** A few settings control how long and how intensely training runs. Beginners can leave everything at defaults. Advanced users get full control over every knob.

4. **Training runs.** The tool loads the model into your GPU/CPU memory and runs training in the background. You watch a live progress screen showing each step, the loss (a measure of how wrong the model still is — lower is better), and elapsed time.

5. **You get a file.** When training finishes, a small *adapter* file is saved to your disk. This adapter is not a full copy of the model — it's a compact set of modifications (typically 10–100 MB) that, when applied on top of the base model, makes it behave according to your data.

---

## What is LoRA / QLoRA?

Training a full language model requires hundreds of gigabytes of GPU memory and weeks of compute time. That's not feasible on a laptop.

**LoRA** (Low-Rank Adaptation) is a technique that sidesteps this. Instead of adjusting every single number inside the model (billions of parameters), LoRA inserts tiny extra "training layers" into specific parts of the model. Only those tiny layers are trained. The rest of the model stays frozen.

The result: you can fine-tune a 7-billion-parameter model using just 4–8 GB of RAM, in minutes to hours instead of weeks.

**QLoRA** takes it further — it first compresses ("quantizes") the base model to use 4-bit numbers instead of 16-bit, cutting memory usage roughly in half again. This is how you fine-tune large models on consumer hardware.

On Apple Silicon (M1/M2/M3 Macs), llmtune automatically uses regular LoRA in float16 — QLoRA's quantization library doesn't support Apple chips, so that step is skipped automatically.

---

## The interface

llmtune has a full terminal UI (TUI) — it's not just text scrolling in a shell. It has proper screens, inputs, buttons, and navigation, all rendered inside your terminal.

**Screen 1 — Model selection**
Two options:
- Paste a local folder path if you already have a HuggingFace model downloaded
- Pick from a list of popular models (they will be downloaded from HuggingFace the first time, then cached on disk forever)

> **Supported model sources:** HuggingFace download or a local HuggingFace folder.
> GGUF format (`.gguf` files, Ollama blobs) is **not supported** for training — GGUF is an
> inference-only format. If you have an Ollama model installed, the model selection screen
> will show its HuggingFace equivalent so you can use that instead.

**Screen 2 — Dataset**
Enter the path to your dataset file and choose the format. An "Advanced" section lets you configure the text field name and sequence length if needed.

**Screen 3 — Training settings**
Three core settings are shown immediately: epochs, batch size, and output folder. Below them, three collapsible sections let advanced users configure LoRA parameters, quantization mode, and the learning rate scheduler.

**Screen 4 — Training**
Live view of training. Shows current step, loss value, elapsed time, a progress bar, and a scrollable log of everything the trainer outputs. You can stop training early at any time.

---

## What is a "loss"?

During training, the model repeatedly tries to predict the next word in your dataset. The *loss* is a number that measures how often it gets it wrong. It starts high (bad) and should decrease over time as the model learns. A falling loss curve means training is working. A flat or rising loss means something is off (wrong learning rate, too few examples, etc.).

---

## What gets saved?

After training, two things are saved to your chosen output folder:

1. **Adapter weights** — the LoRA "diff" on top of the base model. A few files, usually under 100 MB.
2. **Tokenizer config** — the vocabulary settings needed to use the model correctly.

To use the fine-tuned model later, you load the base model and apply the adapter on top. No need to store a full copy of the base model for each fine-tune — adapters are tiny.

---

## Privacy

llmtune requires **no account and no login**. Everything runs locally on your
machine — your data, models, and training never leave the device, and nothing is
sent to any server. The tool is free with no usage limits.

---

## Hardware requirements

| Scenario | Minimum RAM | Notes |
|---|---|---|
| 1B parameter model (e.g. TinyLlama) | 6 GB | Works on most laptops |
| 3B parameter model | 8 GB | M1 MacBook Air with 8 GB is borderline |
| 7B parameter model | 14–16 GB | Needs 16 GB unified memory (M1 Pro / M2 etc.) |
| 13B+ parameter model | 24 GB+ | Desktop GPU recommended |

Apple Silicon Macs use "unified memory" — GPU and CPU share the same pool, so a 16 GB M1 Pro can handle a 7B model that would need a dedicated 16 GB NVIDIA card on a Windows PC.

---

## Performance

Speed depends heavily on model size, sequence length, and your hardware. One measured reference point:

| Model | Method | Hardware | Throughput | 50 steps |
|---|---|---|---|---|
| TinyLlama-1.1B | float16 LoRA | Apple M1 (MPS) | ~3.8 s/it | ~3.6 min |

Numbers on other setups will vary — run the example in [`examples/`](examples/) to benchmark your own machine. (Community-contributed benchmarks welcome.)

---

## Dataset format

Your dataset is a file on your computer. Supported formats:

**JSONL (recommended)** — one JSON object per line:
```jsonl
{"text": "Question: What is the capital of France? Answer: Paris."}
{"text": "Question: How do I reverse a list in Python? Answer: my_list[::-1]"}
```

**Instruction/Response format (auto-detected):**
```jsonl
{"instruction": "Summarize this article", "context": "...", "response": "..."}
```

**JSON** — an array of objects:
```json
[{"text": "..."}, {"text": "..."}]
```

**CSV** — with a column called `text` (configurable).

**Plain text** — one training sample per line.

A minimum of ~50–100 examples is recommended. More is better. Quality matters more than quantity.

---

## Technology stack

| Layer | Technology | Why |
|---|---|---|
| Terminal UI | [Textual](https://textual.textualize.io) | Modern Python TUI framework, looks great |
| Fine-tuning | [HuggingFace PEFT](https://github.com/huggingface/peft) + [TRL](https://github.com/huggingface/trl) | Industry standard LoRA implementation |
| Model loading | [Transformers](https://huggingface.co/docs/transformers) | Supports every major open-source model |
| Server | [FastAPI](https://fastapi.tiangolo.com) | Serves the UI + REST API |
| Frontend | [React](https://react.dev) + [Vite](https://vitejs.dev) + TypeScript | Browser/native-window UI |
| Packaging | [PyPI](https://pypi.org) via hatchling | Standard Python package distribution |

---

## Installing from PyPI

```bash
pip install llmtune-local
llmtune run
```

No account or login required — `llmtune run` takes you straight to the model selection screen.

---

## CLI commands

```bash
llmtune run                     # launch the app (browser/native window)
llmtune run --tui               # launch the terminal UI instead
llmtune version                 # print the installed version
llmtune export-gguf <path>      # convert a fine-tuned adapter to GGUF (llama.cpp / Ollama)
```

Export a trained adapter to a quantized GGUF in one step (llama.cpp is set up automatically on first use):

```bash
llmtune export-gguf ./my-adapter --quant q4_k_m -o my-model.gguf
```

---

## Python API (programmatic use)

Prefer code over the UI? Drive a training run directly:

```python
from llmtune.training.config import TrainConfig
from llmtune.training.trainer import FineTuner

cfg = TrainConfig(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_path="my_data.jsonl",      # jsonl / json / csv / txt, or a HF dataset id
    quantization="none",               # "none" (Mac/CPU/CUDA) | "4bit" | "8bit" (CUDA only)
    num_epochs=1,
    output_dir="./my-adapter",
)

tuner = FineTuner(
    cfg,
    on_log=print,
    on_progress=lambda step, total, loss: None,
    on_done=lambda path: print("Saved adapter to", path),
    on_error=lambda e: print("Error:", e),
)
tuner.start()   # runs in a background thread
tuner.join()    # block until finished
```

See [`examples/`](examples/) for a runnable end-to-end script.

---

## Using the fine-tuned model

After training completes, load your adapter in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "/path/to/llmtune-output")
tokenizer = AutoTokenizer.from_pretrained("/path/to/llmtune-output")

inputs = tokenizer("What is the capital of France?", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Roadmap

- [ ] Community-contributed benchmarks across more models/hardware
- [ ] DoRA and IA3 adapter methods (PEFT-backed)
- [ ] Resume-from-checkpoint in the UI
- [ ] More built-in dataset templates
- [ ] Optional multi-GPU path for non-Mac setups

Contributions and suggestions are welcome — open an issue.

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).
