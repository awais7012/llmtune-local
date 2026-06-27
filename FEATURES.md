# llmtune — Feature Status

## ✅ All Features Working

---

### 1. No login required
- The app requires **no account and no authentication** — it runs entirely on the
  user's machine and opens straight into the studio.
- Works the same for the browser UI (`llmtune run`) and the TUI (`llmtune run --tui`).

---

### 2. LLM Fine-tuning  *(LoRA / QLoRA)*

**How it works:**
Select any HuggingFace language model → point to a text dataset → configure LoRA/training settings → training runs locally with a live progress screen.

**Model sources supported:**
| Source | Example | Notes |
|---|---|---|
| HuggingFace download | `Qwen/Qwen2.5-1.5B` | Auto-downloaded and cached |
| Local HF folder | `/Users/you/models/llama3` | Must be a full HF checkpoint folder |

> **Not supported:** GGUF files (`.gguf`, Ollama blobs).
> GGUF is an inference-only format. The model selection screen will show the
> HuggingFace equivalent for any Ollama model you have installed.

**Dataset formats:**
| Format | Extension | Notes |
|---|---|---|
| JSONL | `.jsonl` | Recommended — one JSON per line |
| JSON | `.json` | Array of objects |
| CSV | `.csv` | Must have a `text` column (configurable) |
| Plain text | `.txt` | One sample per line |
| HuggingFace Hub | `owner/dataset` | Downloaded automatically |

Auto-detected Dolly-style format: `{"instruction": "...", "context": "...", "response": "..."}`.

**Training modes (Quantization):**
| Mode | Hardware | Memory use |
|---|---|---|
| None (float16 LoRA) | Any — Mac, CPU, CUDA | Medium |
| 4-bit QLoRA | NVIDIA CUDA only | Low |
| 8-bit | NVIDIA CUDA only | Medium-low |

Apple Silicon (MPS): automatically uses float32 LoRA — no quantization (BitsAndBytes doesn't support MPS). Runs at ~3 it/s on M-series chips.

**Configurable hyperparameters:**
- LoRA rank (`r`), alpha, dropout, target modules
- Epochs, batch size, gradient accumulation
- Learning rate, warmup ratio, LR scheduler
- Max sequence length, output folder

**Output:**
- `adapter_model.safetensors` — LoRA weights (~10–100 MB)
- `tokenizer_config.json` and related tokenizer files

---

### 3. CNN / Image Classification Fine-tuning  *(NEW)*

**How it works:**
Select any HuggingFace image classification model → point to an image dataset folder → configure training mode and settings → training runs locally with live accuracy reporting.

**Supported models (and many more via HuggingFace ID):**
| Model | Architecture | Size |
|---|---|---|
| `microsoft/resnet-50` | ResNet | ~100 MB |
| `google/efficientnet-b0` | EfficientNet | ~21 MB |
| `google/efficientnet-b4` | EfficientNet | ~75 MB |
| `facebook/convnext-tiny-224` | ConvNeXt | ~113 MB |
| `facebook/convnext-base-224` | ConvNeXt | ~339 MB |
| `google/vit-base-patch16-224` | Vision Transformer | ~330 MB |
| `facebook/deit-small-patch16-224` | DeiT | ~87 MB |
| `facebook/deit-base-patch16-224` | DeiT | ~330 MB |
| `microsoft/swin-tiny-patch4-window7-224` | Swin Transformer | ~107 MB |

Any `AutoModelForImageClassification`-compatible model from HuggingFace will also work.

**Dataset format:**
```
dataset/
  class_a/   image1.jpg  image2.jpg  …
  class_b/   image3.jpg  image4.jpg  …
```
Or with pre-split train/validation:
```
dataset/
  train/class_a/  train/class_b/
  validation/class_a/  validation/class_b/
```
If no validation split is found, 10% of the training data is held out automatically.

Supported image formats: JPEG, PNG, BMP, WebP, TIFF (anything PIL can read).

**Training modes:**
| Mode | What trains | Best for |
|---|---|---|
| Feature extraction | Classifier head only | Small datasets, fast results |
| Full fine-tuning | All layers | Large datasets, maximum accuracy |
| LoRA | Attention adapters | ViT / DeiT / Swin models — efficient |

**Output:**
- `model.safetensors` — fine-tuned weights
- `preprocessor_config.json` — image processor config
- `label_map.json` — `{"id2label": {...}, "label2id": {...}, "num_labels": N}`

**Loading the fine-tuned CNN:**
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch, json

processor = AutoImageProcessor.from_pretrained("/path/to/llmtune-cnn-output")
model = AutoModelForImageClassification.from_pretrained("/path/to/llmtune-cnn-output")
model.eval()

label_map = json.loads(open("/path/to/llmtune-cnn-output/label_map.json").read())
id2label = label_map["id2label"]

image = Image.open("my_image.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = id2label[str(logits.argmax(-1).item())]
print(predicted_class)
```

---

### 4. Live Training Screen

Both LLM and CNN training share the same live training screen:
- Real-time loss display (color-coded: green < 0.5, yellow < 1.5, red ≥ 1.5)
- Step counter and progress bar
- Elapsed time
- Scrollable full training log
- Stop training at any time (saves nothing on early stop)
- CNN mode also reports validation accuracy per epoch

---

## Loading a Fine-tuned LLM Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
model = PeftModel.from_pretrained(base, "/path/to/llmtune-output")
tokenizer = AutoTokenizer.from_pretrained("/path/to/llmtune-output")

inputs = tokenizer("Hello, world!", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
```

---

## Hardware Requirements

| Task | Min RAM | Notes |
|---|---|---|
| CNN feature extraction (any) | 4 GB | CPU is fine for small models |
| CNN full fine-tuning (ResNet-50) | 6 GB | |
| LLM 1B (TinyLlama / Qwen-1.5B) | 6 GB | float16 LoRA |
| LLM 3B (Llama-3.2-3B) | 8 GB | |
| LLM 7B (Mistral, Llama-3.1-8B) | 16 GB | |
| LLM 7B 4-bit QLoRA | 8 GB | NVIDIA CUDA only |
