"""Hyperparameter configuration screen."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, RadioButton, RadioSet, Rule, Static


class HyperparamsScreen(Screen):
    CSS = """
    HyperparamsScreen { background: #0d1117; }

    #summary {
        border: tall #30363d;
        background: #161b22;
        padding: 1 2;
        margin-top: 1;
        height: auto;
    }
    #summary-title { color: #58a6ff; text-style: bold; margin-bottom: 0; }
    #summary-text  { color: #8b949e; }

    #back-btn { width: 16; margin-right: 2; }
    #next-btn { width: 28; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("[bold]✦  llmtune[/]", id="topbar-logo")
            yield Static("Step 3 of 3", id="topbar-step")

        with Vertical(id="content"):
            yield Static("[bold]⚙  Training Configuration[/]", classes="screen-title")
            yield Rule()

            yield Static("Quick settings", classes="section-label")

            with Horizontal(classes="setting-row"):
                yield Static("Epochs", classes="setting-label")
                yield Input(value="3", id="epochs", classes="setting-input")

            with Horizontal(classes="setting-row"):
                yield Static("Batch size", classes="setting-label")
                yield Input(value="2", id="batch-size", classes="setting-input")

            with Horizontal(classes="setting-row"):
                yield Static("Grad accumulation", classes="setting-label")
                yield Input(value="4", id="grad-accum", classes="setting-input")

            with Horizontal(classes="setting-row"):
                yield Static("Output folder", classes="setting-label")
                yield Input(value=str(Path.home() / "llmtune-output"), id="output-path", classes="setting-input")

            with Collapsible(title="LoRA settings  (advanced)", collapsed=True):
                yield Static(
                    "LoRA trains small extra layers instead of the full model — far less memory needed.",
                    classes="hint",
                )
                with Horizontal(classes="setting-row"):
                    yield Static("Rank (r)", classes="setting-label")
                    yield Input(value="16", id="lora-r", classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Static("Alpha", classes="setting-label")
                    yield Input(value="32", id="lora-alpha", classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Static("Dropout", classes="setting-label")
                    yield Input(value="0.05", id="lora-dropout", classes="setting-input")

            with Collapsible(title="Quantization  (advanced)", collapsed=True):
                yield Static(
                    "None = float16 LoRA — works on all hardware including Apple Silicon.\n"
                    "4-bit / 8-bit = less VRAM but requires an NVIDIA CUDA GPU.",
                    classes="hint",
                )
                with RadioSet(id="quant-set"):
                    yield RadioButton("None  — float16 LoRA  (Mac, CPU, any GPU)", value=True, id="q-none")
                    yield RadioButton("4-bit QLoRA  — NVIDIA CUDA GPU only", id="q-4bit")
                    yield RadioButton("8-bit  — NVIDIA CUDA GPU only", id="q-8bit")

            with Collapsible(title="Optimizer  (advanced)", collapsed=True):
                with Horizontal(classes="setting-row"):
                    yield Static("Learning rate", classes="setting-label")
                    yield Input(value="2e-4", id="lr", classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Static("Warmup ratio", classes="setting-label")
                    yield Input(value="0.03", id="warmup", classes="setting-input")

            with Vertical(id="summary"):
                yield Static("Config summary", id="summary-title")
                yield Static(
                    "[dim]Fill in the fields above, then press Start Fine-tuning →[/dim]",
                    id="summary-text",
                )

        with Horizontal(id="bottom-nav"):
            yield Static("", id="status")
            yield Button("← Back", id="back-btn", variant="default")
            yield Button("Start Fine-tuning →", id="next-btn", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id != "next-btn":
            return

        cfg = self.app.train_config
        try:
            lora_r = int(self.query_one("#lora-r", Input).value)
            lora_alpha = int(self.query_one("#lora-alpha", Input).value)
            lora_dropout = float(self.query_one("#lora-dropout", Input).value)
            num_epochs = int(self.query_one("#epochs", Input).value)
            batch_size = int(self.query_one("#batch-size", Input).value)
            grad_accum = int(self.query_one("#grad-accum", Input).value)
            learning_rate = float(self.query_one("#lr", Input).value)
            warmup_ratio = float(self.query_one("#warmup", Input).value)
            output_dir = self.query_one("#output-path", Input).value.strip()
        except ValueError as e:
            self.query_one("#status", Static).update(f"✗ Invalid value: {e}")
            return

        if lora_r < 1 or lora_alpha < 1:
            self.query_one("#status", Static).update("✗ LoRA rank and alpha must be positive integers.")
            return
        if not 0 <= lora_dropout <= 1:
            self.query_one("#status", Static).update("✗ LoRA dropout must be between 0 and 1.")
            return
        if num_epochs < 1:
            self.query_one("#status", Static).update("✗ Number of epochs must be at least 1.")
            return
        if batch_size < 1:
            self.query_one("#status", Static).update("✗ Batch size must be at least 1.")
            return
        if grad_accum < 1:
            self.query_one("#status", Static).update("✗ Gradient accumulation steps must be at least 1.")
            return
        if learning_rate <= 0:
            self.query_one("#status", Static).update("✗ Learning rate must be positive.")
            return
        if not 0 <= warmup_ratio <= 1:
            self.query_one("#status", Static).update("✗ Warmup ratio must be between 0 and 1.")
            return
        if not output_dir:
            self.query_one("#status", Static).update("✗ Output folder cannot be empty.")
            return
        cfg.lora_r = lora_r
        cfg.lora_alpha = lora_alpha
        cfg.lora_dropout = lora_dropout
        cfg.num_epochs = num_epochs
        cfg.per_device_train_batch_size = batch_size
        cfg.gradient_accumulation_steps = grad_accum
        cfg.learning_rate = learning_rate
        cfg.warmup_ratio = warmup_ratio
        cfg.output_dir = output_dir

        quant_map = {"q-4bit": "4bit", "q-8bit": "8bit", "q-none": "none"}
        selected = self.query_one("#quant-set", RadioSet).pressed_button
        cfg.quantization = quant_map.get(selected.id if selected else "q-none", "none")

        import os
        model_short = os.path.basename(cfg.model_id.rstrip("/"))
        self.query_one("#summary-text", Static).update(
            f"[bold]{model_short}[/bold]  ·  {cfg.num_epochs} epoch(s)  ·  "
            f"batch {cfg.per_device_train_batch_size}×{cfg.gradient_accumulation_steps}  ·  "
            f"LoRA r={cfg.lora_r}  ·  {cfg.quantization} quant\n"
            f"[dim]Output → {cfg.output_dir}[/dim]"
        )

        from llmtune.tui.screens.training import TrainingScreen
        self.app.push_screen(TrainingScreen())
