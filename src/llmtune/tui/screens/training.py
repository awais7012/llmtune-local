"""Live training screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, ProgressBar, RichLog, Rule, Static


def _loss_markup(loss: float) -> str:
    """Color-code loss value: red (high) → yellow → green (low)."""
    if loss <= 0:
        return "[dim]—[/dim]"
    if loss < 0.5:
        color = "#3fb950"
    elif loss < 1.5:
        color = "#d29922"
    else:
        color = "#f85149"
    return f"[bold {color}]{loss:.4f}[/bold {color}]"


def _friendly_error(msg: str) -> str:
    m = msg.lower()
    # Pass through already-friendly messages from trainer.py
    if "not saved on this machine" in m or "could not be downloaded" in m:
        return msg
    if "could not connect" in m or "check your internet" in m:
        return msg
    if "authentication failed" in m or "check that your huggingface" in m or "add your huggingface" in m:
        return msg
    if "rate-limited" in m or "try again in a few hours" in m or "too many requests" in m:
        return msg
    if "access denied" in m or "ip may be blocked" in m:
        return msg
    if "not found on huggingface" in m or "download failed (http" in m or "rate limited by huggingface" in m:
        return msg
    if "download blocked" in m or "download complete" in m:
        return msg
    # Catch-all patterns for raw/unexpected exceptions
    if "429" in msg or "rate limit" in m:
        return "Download blocked — your IP may be rate-limited. Add a HuggingFace token on the Model screen to bypass this."
    if "401" in msg or "403" in msg or "gated" in m:
        return "Authentication failed. Check your HuggingFace token on the Model screen, or choose a public model."
    if "out of memory" in m or "oom" in m:
        return "Not enough memory. Try a smaller model or reduce batch size in Training Settings."
    if "does not appear to have a file" in msg or "pytorch_model.bin" in msg:
        return "Model not found locally and could not be downloaded. Go back and enter a local folder path, or check your internet."
    if "no such file" in m:
        return "File not found — check your model or dataset path."
    if len(msg) <= 180:
        return msg
    return "Training stopped — an error occurred. See the log below for the full error message."


class TrainingScreen(Screen):
    CSS = """
    TrainingScreen { background: #0d1117; }

    #phase-header {
        text-style: bold;
        color: #e6edf3;
        height: 1;
        margin-bottom: 0;
    }

    #model-info {
        color: #8b949e;
        height: 1;
        margin-bottom: 0;
    }

    #device-label {
        color: #3fb950;
        height: 1;
        margin-bottom: 0;
    }

    #progress-label {
        color: #8b949e;
        height: 1;
        margin-bottom: 0;
    }

    #stop-btn { width: 26; }
    """

    def compose(self) -> ComposeResult:
        import os
        mode = getattr(self.app, "train_mode", "llm")
        if mode == "cnn":
            cfg = self.app.image_config
            mode_label = f"{cfg.training_mode}  ·  Image classification"
        else:
            cfg = self.app.train_config
            mode_label = f"{cfg.quantization} quant  ·  {cfg.num_epochs} epoch(s)  ·  LoRA r={cfg.lora_r}"
        model_short = os.path.basename(cfg.model_id.rstrip("/"))

        with Horizontal(id="topbar"):
            yield Static("[bold]✦  llmtune[/]", id="topbar-logo")
            yield Static("Step 4 of 4" if mode == "llm" else "Step 4 of 4", id="topbar-step")

        with Vertical(id="content"):
            yield Static("⏳  Loading…", id="phase-header")
            yield Static(f"{model_short}  ·  {mode_label}", id="model-info")
            yield Static("Detecting device…", id="device-label")

            yield Rule()

            with Horizontal(id="metrics-row"):
                with Vertical(classes="metric-card"):
                    yield Static("STEP", classes="metric-label")
                    yield Static("—", id="stat-step", classes="metric-value")
                with Vertical(classes="metric-card"):
                    yield Static("LOSS", classes="metric-label")
                    yield Static("[dim]—[/dim]", id="stat-loss", classes="metric-value")
                with Vertical(classes="metric-card"):
                    yield Static("ELAPSED", classes="metric-label")
                    yield Static("—", id="stat-time", classes="metric-value")

            yield Static("Waiting to start…", id="progress-label")
            yield ProgressBar(total=100, show_eta=True, id="progress-bar")
            yield Static("Training Log", id="log-header")
            yield RichLog(id="log", auto_scroll=True, wrap=True, markup=False, highlight=False)
            yield Static("", id="banner")

        with Horizontal(id="bottom-nav"):
            yield Button("⏹  Stop Training", id="stop-btn", variant="error")

    def on_mount(self) -> None:
        import time
        self._start_time = time.time()
        self._start_training()

    def _start_training(self) -> None:
        def _on_log(msg: str) -> None:
            self.app.call_from_thread(self._append_log, msg)

        def _on_progress(step: int, total: int, loss: float) -> None:
            self.app.call_from_thread(self._update_progress, step, total, loss)

        def _on_done(path: str) -> None:
            self.app.call_from_thread(self._training_done, path)

        def _on_error(e: Exception) -> None:
            self.app.call_from_thread(self._training_error, str(e))

        callbacks = dict(on_log=_on_log, on_progress=_on_progress, on_done=_on_done, on_error=_on_error)

        if getattr(self.app, "train_mode", "llm") == "cnn":
            from llmtune.training.image_trainer import ImageFineTuner
            self._tuner = ImageFineTuner(config=self.app.image_config, **callbacks)
        else:
            from llmtune.training import FineTuner
            self._tuner = FineTuner(config=self.app.train_config, **callbacks)

        self._tuner.start()

    def _elapsed(self) -> str:
        import time
        secs = int(time.time() - self._start_time)
        m, s = divmod(secs, 60)
        return f"{m}m {s:02d}s"

    def _append_log(self, msg: str) -> None:
        log = self.query_one("#log", RichLog)
        log.write(msg)

        # Parse device line and update device label
        if msg.startswith("Device:"):
            device = msg.replace("Device:", "").strip()
            label = {
                "MPS": "Apple Silicon GPU (MPS)",
                "CUDA": "NVIDIA GPU (CUDA)",
                "CPU": "CPU — no GPU acceleration",
            }.get(device, device)
            self.query_one("#device-label", Static).update(f"Running on: {label}")

        # Phase-based status updates
        if "not in local cache" in msg.lower():
            self.query_one("#phase-header", Static).update("⬇  Downloading")
            self.query_one("#progress-label", Static).update("Downloading from HuggingFace — this may take a few minutes…")
        elif "download complete" in msg.lower():
            self.query_one("#phase-header", Static).update("⏳  Loading Model")
            self.query_one("#progress-label", Static).update("Download complete — loading model into memory…")
        elif "Loading tokenizer" in msg:
            self.query_one("#phase-header", Static).update("⏳  Loading Model")
            self.query_one("#progress-label", Static).update("Loading tokenizer…")
        elif "Loading model" in msg:
            self.query_one("#phase-header", Static).update("⏳  Loading Model")
            self.query_one("#progress-label", Static).update("Loading model weights into memory…")
        elif "Loading dataset" in msg:
            self.query_one("#phase-header", Static).update("📂  Loading Dataset")
            self.query_one("#progress-label", Static).update("Loading and tokenizing dataset…")
        elif "Starting training" in msg:
            self.query_one("#phase-header", Static).update("🔥  Training")
            self.query_one("#progress-label", Static).update("Training started!")
        elif "Trainable params" in msg:
            log.write(f"  → {msg}")

        self.query_one("#stat-time", Static).update(self._elapsed())

    def _update_progress(self, step: int, total: int, loss: float) -> None:
        self.query_one("#stat-time", Static).update(self._elapsed())
        if loss < 0:
            # Download phase — step is percentage, total is 100
            if total > 0:
                self.query_one(ProgressBar).progress = step
                self.query_one("#progress-label", Static).update(f"Downloading…  ({step}%)")
            return
        # Training phase
        self.query_one("#stat-step", Static).update(f"{step}/{total}")
        self.query_one("#stat-loss", Static).update(_loss_markup(loss))
        if total > 0:
            pct = int(100 * step / total)
            self.query_one(ProgressBar).progress = pct
            self.query_one("#progress-label", Static).update(
                f"Step {step} of {total}  ({pct}%)"
            )

    def _training_done(self, path: str) -> None:
        self.query_one("#phase-header", Static).update("✅  Training Complete")
        self.query_one("#banner", Static).update(f"[#3fb950]Adapter saved → {path}[/]")
        self.query_one(ProgressBar).progress = 100
        btn = self.query_one("#stop-btn", Button)
        btn.label = "  Exit"
        btn.variant = "primary"

    def _training_error(self, msg: str) -> None:
        self.query_one("#phase-header", Static).update("❌  Training Failed")
        friendly = _friendly_error(msg)
        self.query_one("#banner", Static).update(f"[#f85149]{friendly}[/]")
        log = self.query_one("#log", RichLog)
        log.write("")
        log.write("━" * 50)
        log.write(f"Error: {msg}")
        btn = self.query_one("#stop-btn", Button)
        btn.label = "  Exit"
        btn.variant = "primary"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "stop-btn":
            return
        label = str(event.button.label).strip()
        if label == "Exit":
            self.app.exit()
        else:
            if hasattr(self, "_tuner"):
                self._tuner.stop()
            event.button.label = "Stopping…"
            event.button.disabled = True
