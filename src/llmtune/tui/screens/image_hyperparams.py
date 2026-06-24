"""Hyperparameter screen for CNN / image-classification fine-tuning."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, RadioButton, RadioSet, Rule, Static


class ImageHyperparamsScreen(Screen):
    CSS = """
    ImageHyperparamsScreen { background: #0d1117; }

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
    #next-btn { width: 26; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("[bold]✦  llmtune[/]", id="topbar-logo")
            yield Static("Step 3 of 3", id="topbar-step")

        with Vertical(id="content"):
            yield Static("[bold]⚙  CNN Training Configuration[/]", classes="screen-title")
            yield Rule()

            # ── Training mode ────────────────────────────────────────────────
            yield Static("Training mode", classes="section-label")
            yield Static(
                "Controls which part of the model gets trained.",
                classes="hint",
            )
            with RadioSet(id="mode-set"):
                yield RadioButton(
                    "Feature extraction  — freeze backbone, train head only  (fastest, least RAM)",
                    value=True,
                    id="m-feature",
                )
                yield RadioButton(
                    "Full fine-tuning    — train all layers  (most accurate, most RAM)",
                    id="m-full",
                )
                yield RadioButton(
                    "LoRA               — adapter layers on attention  (ViT / Swin models only)",
                    id="m-lora",
                )

            # ── Quick settings ───────────────────────────────────────────────
            yield Static("Quick settings", classes="section-label")

            with Horizontal(classes="setting-row"):
                yield Static("Epochs", classes="setting-label")
                yield Input(value="10", id="epochs", classes="setting-input")

            with Horizontal(classes="setting-row"):
                yield Static("Batch size", classes="setting-label")
                yield Input(value="16", id="batch-size", classes="setting-input")

            with Horizontal(classes="setting-row"):
                yield Static("Learning rate", classes="setting-label")
                yield Input(value="1e-3", id="lr", classes="setting-input")

            with Horizontal(classes="setting-row"):
                yield Static("Output folder", classes="setting-label")
                yield Input(value=str(Path.home() / "llmtune-cnn-output"), id="output-path", classes="setting-input")

            # ── Advanced ─────────────────────────────────────────────────────
            with Collapsible(title="LoRA settings  (used when LoRA mode is selected)", collapsed=True):
                yield Static(
                    "LoRA works best on transformer-based image models (ViT, DeiT, Swin).\n"
                    "For ResNet / EfficientNet / ConvNeXt, use feature extraction or full fine-tuning instead.",
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

            with Collapsible(title="Image settings  (advanced)", collapsed=True):
                with Horizontal(classes="setting-row"):
                    yield Static("Image size (px)", classes="setting-label")
                    yield Input(value="224", id="image-size", classes="setting-input")
                with Horizontal(classes="setting-row"):
                    yield Static("Val split (0–1)", classes="setting-label")
                    yield Input(value="0.1", id="val-split", classes="setting-input")

            with Vertical(id="summary"):
                yield Static("Config summary", id="summary-title")
                yield Static(
                    "[dim]Fill in the fields above, then press Start Training →[/dim]",
                    id="summary-text",
                )

        with Horizontal(id="bottom-nav"):
            yield Static("", id="status")
            yield Button("← Back", id="back-btn", variant="default")
            yield Button("Start Training →", id="next-btn", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id != "next-btn":
            return

        cfg = self.app.image_config
        try:
            num_epochs = int(self.query_one("#epochs", Input).value)
            batch_size = int(self.query_one("#batch-size", Input).value)
            learning_rate = float(self.query_one("#lr", Input).value)
            output_dir = self.query_one("#output-path", Input).value.strip()
            lora_r = int(self.query_one("#lora-r", Input).value)
            lora_alpha = int(self.query_one("#lora-alpha", Input).value)
            lora_dropout = float(self.query_one("#lora-dropout", Input).value)
            image_size = int(self.query_one("#image-size", Input).value)
            val_split = float(self.query_one("#val-split", Input).value)
        except ValueError as e:
            self.query_one("#status", Static).update(f"✗ Invalid value: {e}")
            return

        if num_epochs < 1:
            self.query_one("#status", Static).update("✗ Number of epochs must be at least 1.")
            return
        if batch_size < 1:
            self.query_one("#status", Static).update("✗ Batch size must be at least 1.")
            return
        if learning_rate <= 0:
            self.query_one("#status", Static).update("✗ Learning rate must be positive.")
            return
        if not output_dir:
            self.query_one("#status", Static).update("✗ Output folder cannot be empty.")
            return
        if lora_r < 1 or lora_alpha < 1:
            self.query_one("#status", Static).update("✗ LoRA rank and alpha must be positive integers.")
            return
        if not 0 <= lora_dropout <= 1:
            self.query_one("#status", Static).update("✗ LoRA dropout must be between 0 and 1.")
            return
        if image_size < 32:
            self.query_one("#status", Static).update("✗ Image size must be at least 32 pixels.")
            return
        if not 0 < val_split < 1:
            self.query_one("#status", Static).update("✗ Validation split must be between 0 and 1 (exclusive).")
            return
        cfg.num_epochs = num_epochs
        cfg.per_device_train_batch_size = batch_size
        cfg.learning_rate = learning_rate
        cfg.output_dir = output_dir
        cfg.lora_r = lora_r
        cfg.lora_alpha = lora_alpha
        cfg.lora_dropout = lora_dropout
        cfg.image_size = image_size
        cfg.val_split = val_split

        mode_map = {"m-feature": "feature_extraction", "m-full": "full", "m-lora": "lora"}
        selected = self.query_one("#mode-set", RadioSet).pressed_button
        cfg.training_mode = mode_map.get(selected.id if selected else "m-feature", "feature_extraction")

        import os
        model_short = os.path.basename(cfg.model_id.rstrip("/"))
        self.query_one("#summary-text", Static).update(
            f"[bold]{model_short}[/bold]  ·  {cfg.num_epochs} epoch(s)  ·  "
            f"batch {cfg.per_device_train_batch_size}  ·  lr={cfg.learning_rate}  ·  {cfg.training_mode}\n"
            f"[dim]Output → {cfg.output_dir}[/dim]"
        )

        from llmtune.tui.screens.training import TrainingScreen
        self.app.push_screen(TrainingScreen())
