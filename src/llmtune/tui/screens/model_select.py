"""Model selection screen — LLM text models and CNN image models."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, Label, ListItem, ListView, Rule, Static

# ── LLM models ───────────────────────────────────────────────────────────────
POPULAR_MODELS: list[tuple[str, str, str]] = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat    — great for testing",     "~2.2 GB"),
    ("Qwen/Qwen2.5-1.5B",                  "Qwen 2.5 1.5B          — strong multilingual",    "~3.0 GB"),
    ("microsoft/phi-3-mini-4k-instruct",    "Phi-3 Mini 3.8B        — excellent for coding",   "~7.6 GB"),
    ("Qwen/Qwen2.5-7B",                     "Qwen 2.5 7B            — high quality",           "~14 GB"),
    ("meta-llama/Llama-3.2-1B",             "Llama 3.2 1B           — Meta's smallest",        "~2.0 GB"),
    ("meta-llama/Llama-3.2-3B",             "Llama 3.2 3B           — great balance",          "~6.4 GB"),
    ("meta-llama/Llama-3.1-8B",             "Llama 3.1 8B           — very capable",           "~16 GB"),
    ("mistralai/Mistral-7B-v0.1",           "Mistral 7B             — efficient reasoning",    "~14 GB"),
    ("google/gemma-2-2b",                   "Gemma 2 2B             — Google compact",         "~4.9 GB"),
    ("google/gemma-2-9b",                   "Gemma 2 9B             — Google large",           "~18 GB"),
]
_LLM_META: dict[str, tuple[str, str]] = {m[0]: (m[1], m[2]) for m in POPULAR_MODELS}

# ── CNN / Image classification models ────────────────────────────────────────
POPULAR_CNN_MODELS: list[tuple[str, str, str]] = [
    ("microsoft/resnet-50",                    "ResNet-50               — classic, battle-tested",    "~100 MB"),
    ("google/efficientnet-b0",                 "EfficientNet-B0         — compact & accurate",        "~21 MB"),
    ("google/efficientnet-b4",                 "EfficientNet-B4         — higher accuracy",           "~75 MB"),
    ("facebook/convnext-tiny-224",             "ConvNeXt Tiny           — modern conv net",           "~113 MB"),
    ("facebook/convnext-base-224",             "ConvNeXt Base           — larger conv net",           "~339 MB"),
    ("google/vit-base-patch16-224",            "ViT Base/16             — vision transformer",        "~330 MB"),
    ("facebook/deit-small-patch16-224",        "DeiT Small              — efficient ViT",             "~87 MB"),
    ("facebook/deit-base-patch16-224",         "DeiT Base               — standard ViT",             "~330 MB"),
    ("microsoft/swin-tiny-patch4-window7-224", "Swin Transformer Tiny   — hierarchical ViT",          "~107 MB"),
]
_CNN_META: dict[str, tuple[str, str]] = {m[0]: (m[1], m[2]) for m in POPULAR_CNN_MODELS}


def _is_gguf_path(path: str) -> bool:
    """Return True if the path points to a GGUF file (by extension or magic bytes)."""
    p = Path(path)
    if p.suffix.lower() == ".gguf":
        return True
    if p.is_file() and not p.suffix:
        try:
            with open(p, "rb") as f:
                return f.read(4) == b"GGUF"
        except Exception:
            pass
    return False


class ModelSelectScreen(Screen):
    CSS = """
    ModelSelectScreen { background: #0d1117; }

    #selected-label {
        color: #58a6ff;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    #local-path { margin-bottom: 0; }
    #search     { margin-bottom: 0; }
    #cnn-search { margin-bottom: 0; }

    ListView { height: 5; margin-bottom: 0; }

    #cnn-section { margin-top: 1; }

    #ollama-collapsible { margin-top: 1; margin-bottom: 0; }
    .ollama-ref-row { height: 2; padding: 0 1; color: #e6edf3; }

    #hf-token   { margin-bottom: 0; }
    #token-link { color: #8b949e; margin-top: 0; }

    #next-btn { width: 22; }
    """

    _selected_hf: str = ""
    _selected_cnn: str = ""

    def compose(self) -> ComposeResult:
        from llmtune.training.gguf_utils import is_ollama_installed, list_ollama_models
        self._selected_hf = ""
        self._selected_cnn = ""
        ollama_ok = is_ollama_installed()
        self._ollama_models: list[dict] = list_ollama_models() if ollama_ok else []

        with Horizontal(id="topbar"):
            yield Static("[bold]✦  llmtune[/]", id="topbar-logo")
            yield Static("Step 1 of 3", id="topbar-step")

        with Vertical(id="content"):
            yield Static("[bold]🤖  Select Base Model[/]", classes="screen-title")
            yield Rule()
            yield Static("No model selected", id="selected-label")

            # ── Local path ───────────────────────────────────────────────────
            yield Static("Already have a model downloaded?", classes="section-label")
            yield Input(
                placeholder="Local HuggingFace folder — e.g. /Users/you/models/llama3",
                id="local-path",
            )

            # ── LLM / text models ────────────────────────────────────────────
            yield Static("LLM / Text models  —  language fine-tuning", classes="section-label")
            yield Input(placeholder="Search LLM models…", id="search")
            yield ListView(
                *[ListItem(Label(label), name=mid) for mid, label, _ in POPULAR_MODELS],
                id="model-list",
            )

            # ── CNN / image models ───────────────────────────────────────────
            with Vertical(id="cnn-section"):
                yield Static("CNN / Image models  —  image classification fine-tuning", classes="section-label")
                yield Static(
                    "Fine-tune on your own labelled image dataset — cats vs dogs, defect detection, etc.",
                    classes="hint",
                )
                yield Input(placeholder="Search CNN models…", id="cnn-search")
                yield ListView(
                    *[ListItem(Label(label), name=mid) for mid, label, _ in POPULAR_CNN_MODELS],
                    id="cnn-model-list",
                )

            # ── Ollama reference ─────────────────────────────────────────────
            if ollama_ok and self._ollama_models:
                n = len(self._ollama_models)
                with Collapsible(
                    title=f"🦙 Ollama models installed ({n}) — HuggingFace equivalents",
                    collapsed=True,
                    id="ollama-collapsible",
                ):
                    yield Static(
                        "Ollama models run inference only — they cannot be fine-tuned directly.\n"
                        "Use the HuggingFace ID shown below as your model instead:",
                        classes="hint",
                    )
                    for m in self._ollama_models:
                        hf_id = m.get("hf_id") or "(no known HuggingFace equivalent)"
                        yield Static(
                            f"[bold]{m['name']}[/bold]"
                            f"  [dim]·  {m['size_str']}[/dim]"
                            f"  [dim]→[/dim]  [#58a6ff]{hf_id}[/]",
                            classes="ollama-ref-row",
                        )

            # ── HuggingFace token ────────────────────────────────────────────
            with Collapsible(
                title="🔑 HuggingFace Token  —  required for gated / private models",
                collapsed=True,
                id="token-collapsible",
            ):
                yield Static(
                    "Required for gated models (Llama, Gemma) and private repos.\n"
                    "  ✓ Bypasses rate limits  ✓ Access gated / private models\n"
                    "Skip if using a public model or a local folder path.",
                    classes="hint",
                )
                yield Input(
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    password=True,
                    id="hf-token",
                )
                yield Static(
                    "[dim]Get a free token at huggingface.co/settings/tokens  (choose Read permission)[/dim]",
                    id="token-link",
                )

        with Horizontal(id="bottom-nav"):
            yield Static("", id="status")
            yield Button("Next →", id="next-btn", variant="primary")

    # ── Event handlers ───────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "local-path":
            if event.value.strip():
                self._selected_hf = ""
                self._selected_cnn = ""
                self.query_one("#selected-label", Static).update("No model selected")
                self.query_one("#status", Static).update("")
            return

        if event.input.id == "search":
            query = event.value.lower()
            lv = self.query_one("#model-list", ListView)
            lv.clear()
            for mid, label, _ in POPULAR_MODELS:
                if query in mid.lower() or query in label.lower():
                    lv.append(ListItem(Label(label), name=mid))
            return

        if event.input.id == "cnn-search":
            query = event.value.lower()
            lv = self.query_one("#cnn-model-list", ListView)
            lv.clear()
            for mid, label, _ in POPULAR_CNN_MODELS:
                if query in mid.lower() or query in label.lower():
                    lv.append(ListItem(Label(label), name=mid))
            self.query_one("#selected-label", Static).update(
                "No model selected" if not query else "Select a model from the list above"
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not event.item.name:
            return
        self.query_one("#local-path", Input).value = ""
        self.query_one("#status", Static).update("")

        if event.list_view.id == "cnn-model-list":
            self._selected_hf = ""
            self._selected_cnn = event.item.name
            label, size = _CNN_META.get(event.item.name, ("", "unknown size"))
            self.query_one("#selected-label", Static).update(
                f"[bold]{event.item.name}[/bold]  [dim]·  {size}  ·  Image model[/dim]"
            )
        else:
            self._selected_cnn = ""
            self._selected_hf = event.item.name
            label, size = _LLM_META.get(event.item.name, ("", "unknown size"))
            self.query_one("#selected-label", Static).update(
                f"[bold]{event.item.name}[/bold]  [dim]·  {size} download[/dim]"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "next-btn":
            return

        raw_token = self.query_one("#hf-token", Input).value.strip()
        local = self.query_one("#local-path", Input).value.strip()
        hf_id = self.query_one("#search", Input).value.strip()
        cnn_id = self.query_one("#cnn-search", Input).value.strip()

        # ── Resolve model ID ─────────────────────────────────────────────────
        if local:
            expanded = str(Path(local).expanduser())
            if _is_gguf_path(expanded):
                self.query_one("#status", Static).update(
                    "❌ GGUF files cannot be used for training — "
                    "paste the HuggingFace model ID instead (e.g. Qwen/Qwen2.5-0.5B)"
                )
                return
            if not Path(expanded).exists():
                self.query_one("#status", Static).update("✗ Path not found — check it and try again.")
                return
            model_id = expanded
            mode = "llm"

        elif self._selected_cnn:
            model_id = self._selected_cnn
            mode = "cnn"

        elif self._selected_hf:
            model_id = self._selected_hf
            mode = "llm"

        elif hf_id:
            model_id = hf_id
            mode = "llm"

        elif cnn_id:
            model_id = cnn_id
            mode = "cnn"

        else:
            self.query_one("#status", Static).update("✗ Pick a model or enter a local path to continue.")
            return

        # ── Route to the right flow ──────────────────────────────────────────
        self.app.train_mode = mode

        if mode == "cnn":
            self.app.image_config.model_id = model_id
            self.app.image_config.hf_token = raw_token or None
            from llmtune.tui.screens.image_dataset import ImageDatasetScreen
            self.app.push_screen(ImageDatasetScreen())
        else:
            cfg = self.app.train_config
            cfg.hf_token = raw_token or None
            cfg.model_id = model_id
            from llmtune.tui.screens.dataset import DatasetScreen
            self.app.push_screen(DatasetScreen())
