# llmtune-local — Project Report

**Project:** llmtune-local — Local Fine-Tuning Toolkit for LLMs and Image Classifiers
**Package:** [`llmtune-local`](https://pypi.org/project/llmtune-local/) (PyPI) · CLI command: `llmtune`
**Repository:** https://github.com/awais7012/llmtune-local
**License:** MIT
**Status:** Published, working end-to-end (early-stage / actively developed)

---

## 1. Executive Summary

**llmtune-local** is a desktop tool that lets a developer fine-tune open-source AI
models — both large language models (LLMs) and image classifiers — entirely on
their own machine, with no cloud services and no data ever leaving the device.

Unlike existing fine-tuning frameworks, which are configuration-heavy, CLI-only,
and assume an NVIDIA GPU, llmtune-local is built around ease of use: a browser
GUI and a terminal UI, sensible defaults, first-class Apple Silicon support, and
a one-line installation (`pip install llmtune-local`). It targets developers who
want results without deep machine-learning expertise.

The project spans a Python training engine, a local web server, a React user
interface, an authentication layer, automated tests, continuous integration, and
distribution through the Python Package Index (PyPI).

---

## 2. Problem Statement & Motivation

Adapting a pre-trained model to a specific task ("fine-tuning") normally requires:

- Expensive cloud GPUs or paid APIs, and uploading potentially sensitive data to
  third parties.
- Deep familiarity with ML tooling — YAML configuration files, command-line
  training scripts, and an understanding of dozens of hyperparameters.
- NVIDIA/CUDA hardware; most consumer laptops (especially Apple Silicon Macs) are
  poorly supported by mainstream tools.

This excludes a large group of capable developers and small teams who have their
own data and modern laptops but neither the budget nor the ML background to use
the existing ecosystem. The motivation for llmtune-local is to remove those
barriers and make local, private fine-tuning approachable.

---

## 3. Objectives

1. Enable fine-tuning of open-source LLMs (LoRA/QLoRA) on consumer hardware.
2. Support image-classifier fine-tuning (CNNs / Vision Transformers).
3. Run fully locally — no data leaves the user's machine.
4. Provide both a graphical (browser) and terminal interface usable without ML expertise.
5. Support Apple Silicon (Metal / MPS) as a first-class target.
6. Package and distribute the tool so it installs with a single command.
7. Provide a path from a fine-tuned model to a deployable artifact (GGUF export).

---

## 4. Background

**Fine-tuning** continues training a pre-trained model on a smaller, task-specific
dataset so it adopts a desired behaviour, tone, or domain knowledge.

**LoRA (Low-Rank Adaptation)** avoids retraining all of a model's (billions of)
parameters. It freezes the base model and inserts small trainable "adapter"
matrices into selected layers. Only these are trained, reducing memory and time
by orders of magnitude — a multi-billion-parameter model can be tuned in a few
gigabytes of RAM. The output is a compact adapter (typically 10–100 MB) applied
on top of the base model.

**QLoRA** additionally quantizes the base model to 4-bit precision to cut memory
further (NVIDIA/CUDA only). On Apple Silicon, the tool automatically falls back
to float16 LoRA, since the quantization library does not support Metal.

---

## 5. System Architecture

The system is organised into four cooperating components, shipped as a single
Python package:

```
┌──────────────────────────────────────────────────────────────┐
│                    llmtune (CLI entry point)                   │
│              run · run --tui · version · export-gguf           │
└───────────────┬──────────────────────────────┬────────────────┘
                │                               │
        Browser / native window            Terminal UI (Textual)
                │                               │
                ▼                               ▼
┌──────────────────────────────┐    ┌──────────────────────────┐
│   Local FastAPI server        │    │   Training engine         │
│   • serves the React UI       │◄──►│   • LLM trainer (PEFT/TRL)│
│   • REST API (config, models, │    │   • Image trainer         │
│     dataset, jobs, library)   │    │   • device detection      │
│   • WebSocket live progress   │    │   • GGUF merge/export      │
└──────────────────────────────┘    └──────────────────────────┘
```

**a. CLI** (`src/llmtune/__main__.py`) — the `llmtune` command. Launches the app,
the terminal UI, prints the version, and exports trained models to GGUF.

**b. Local server** (`src/llmtune/server/`) — a FastAPI application that serves the
bundled React interface and exposes a REST API for configuration, model search,
dataset validation, training jobs, and library actions (inference, push to
HuggingFace Hub, GGUF export). It runs only on `localhost`.

**c. Frontend** (`frontend/`, built into `server/static/`) — a React + Vite +
TypeScript single-page application providing the graphical workflow.

**d. Training engine** (`src/llmtune/training/`) — the core ML logic: an LLM
fine-tuner built on HuggingFace PEFT + TRL, an image-classifier fine-tuner, device
detection (CUDA / Apple MPS / CPU), and model merge/GGUF export utilities.

---

## 6. Key Features

| Feature | Description |
|---|---|
| LLM fine-tuning | LoRA and QLoRA via HuggingFace PEFT + TRL |
| Image fine-tuning | ResNet, ViT, ConvNeXt, DeiT, Swin, EfficientNet, and any compatible HF model |
| Dual interface | Browser/native-window GUI **and** a full terminal UI |
| Apple Silicon | Automatic float16 LoRA on Metal (MPS); CPU/CUDA also supported |
| Dataset formats | JSONL, JSON, CSV, plain text, and HuggingFace Hub datasets |
| Live training view | Real-time loss, step counter, progress bar, elapsed time, full log |
| Model library | Inference test, push to HuggingFace Hub, GGUF export |
| GGUF export | One command/click to a quantized model for llama.cpp / Ollama |
| Privacy | All computation local; nothing about data/models is transmitted |
| Python API | Programmatic `FineTuner` / `TrainConfig` interface |

---

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10–3.12 (backend), TypeScript (frontend) |
| Fine-tuning | HuggingFace PEFT, TRL, Transformers |
| Numerical / ML | PyTorch, torchvision, datasets, accelerate |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite, TypeScript |
| Terminal UI | Textual, Rich |
| Export | llama.cpp (auto-installed) for GGUF conversion |
| Packaging | hatchling, distributed via PyPI |
| Quality | pytest, GitHub Actions CI |

---

## 8. Implementation Details

**Training pipeline.** A `TrainConfig` dataclass captures all hyperparameters
(model, dataset, LoRA rank/alpha/dropout, epochs, batch size, learning rate,
scheduler, sequence length, output directory). A `FineTuner` runs training on a
background thread, reporting progress via callbacks (log lines, step/loss
updates, completion, errors) so the UI can render a live view. Datasets are loaded
by file extension (JSONL/JSON/CSV/text) or pulled from the HuggingFace Hub, and an
instruction/response format is auto-detected.

**Device handling.** The engine detects CUDA, Apple MPS, or CPU and adapts:
4-bit/8-bit quantization on CUDA, float16 LoRA on Apple Silicon, with graceful
fallback. This is what makes the tool usable on ordinary laptops.

**No authentication.** The tool requires no account or login. It runs entirely on
the user's machine and communicates with no external service for access control;
the local API is reachable only on `localhost`. (Earlier iterations experimented
with a hosted identity provider, but for a fully local tool a login step added
friction without benefit, so it was removed.)

**GGUF export.** After fine-tuning, the tool can merge the LoRA adapter into its
base model and convert the result to a quantized GGUF file (q8_0 / q4_k_m / f16)
for use with llama.cpp or Ollama. llama.cpp's converter is set up automatically on
first use; if it cannot be installed, the merged model is still produced and
manual instructions are given.

---

## 9. Privacy & Security

- All training and inference run locally; user data never leaves the machine.
- No account or login is required.
- The local API binds to `localhost`, and Cross-Origin Resource Sharing is
  restricted to localhost origins so other websites cannot reach it from the
  browser.

---

## 10. Testing & Validation

- **End-to-end functional test:** a full LLM fine-tune was run on Apple Silicon
  (TinyLlama-1.1B, LoRA on MPS) and successfully produced a valid LoRA adapter
  (~4.5 MB of weights plus tokenizer/config), confirming the core pipeline works.
- **Unit tests:** automated tests cover the dataset validator across all supported
  formats and error paths.
- **Continuous integration:** a GitHub Actions workflow runs the test suite across
  Python 3.10, 3.11, and 3.12 on every push and pull request.
- **Distribution test:** a clean `pip install llmtune-local` was verified to
  install successfully on Apple Silicon, including correct platform-gating of a
  CUDA-only dependency.

---

## 11. Packaging & Distribution

The project is packaged with hatchling and published to PyPI as `llmtune-local`.
The built frontend ships inside the package, so a single `pip install llmtune-local`
provides the complete application; `llmtune run` launches it. The CUDA-only
quantization dependency is restricted to Linux so installation does not fail on
macOS or Windows. Releases follow semantic versioning and are documented in a
changelog.

---

## 12. Comparison with Existing Tools

| | llmtune-local | LLaMA-Factory | Axolotl | Torchtune |
|---|:---:|:---:|:---:|:---:|
| Zero-config (no YAML) | ✅ | ❌ | ❌ | ❌ |
| Terminal UI | ✅ | ❌ | ❌ | ❌ |
| Browser GUI | ✅ | ✅ | ❌ | ❌ |
| First-class Apple Silicon | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Image-classifier fine-tuning | ✅ | ❌ | ❌ | ❌ |
| One-command GGUF export | ✅ | ✅ | ⚠️ | ❌ |
| Multi-GPU / DeepSpeed | ❌ | ✅ | ✅ | ✅ |
| Preference tuning (DPO/ORPO) | ❌ | ✅ | ✅ | ✅ |

The established frameworks are more capable for large-scale, multi-GPU, and
research workflows. llmtune-local deliberately trades that breadth for simplicity,
privacy, and laptop-friendly (especially Apple Silicon) ergonomics.

---

## 13. Results

- A complete, installable application that fine-tunes real models locally, verified
  end-to-end on Apple Silicon.
- Published and publicly installable via `pip install llmtune-local`.
- Reference performance (single measured point): TinyLlama-1.1B, float16 LoRA on an
  Apple M1, approximately 3.8 s/iteration (~3.6 minutes for 50 steps). Performance
  on other models and hardware will vary.

---

## 14. Limitations

- No multi-GPU or distributed training; targeted at single-machine use.
- 4-bit/8-bit QLoRA requires NVIDIA/CUDA hardware.
- Advanced preference-tuning methods (DPO/ORPO/GRPO) are not yet implemented.
- Benchmark coverage is limited to a single measured configuration so far.

---

## 15. Future Work

- Community-contributed benchmarks across more models and hardware.
- Additional adapter methods (e.g. DoRA, IA3).
- Resume-from-checkpoint in the user interface.
- More built-in dataset templates and examples.
- An optional multi-GPU path for non-Mac setups.

---

## 16. Conclusion

llmtune-local demonstrates a complete, end-to-end software product: an ML training
engine, a local web service, a graphical and terminal user interface, an
authentication layer, automated testing and CI, and public distribution. It fills
a genuine gap — approachable, private, local fine-tuning on everyday hardware,
including Apple Silicon — and provides a foundation that can grow toward more
advanced training techniques over time.
