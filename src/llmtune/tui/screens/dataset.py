"""Dataset configuration screen."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Collapsible, Input, Label, RadioButton, RadioSet, Rule, Static


def _is_hf_dataset_id(path: str) -> bool:
    """Return True if path looks like a HuggingFace dataset ID (owner/dataset), not a local file."""
    p = path.strip()
    if not p:
        return False
    if p[0] in ("/", "~", ".") or (len(p) > 1 and p[1] == ":"):
        return False
    parts = p.split("/")
    return len(parts) == 2 and all(parts)


class DatasetScreen(Screen):
    CSS = """
    DatasetScreen { background: #0d1117; }

    #dataset-path { margin-bottom: 0; }

    #hf-auth {
        display: none;
        border: tall #d29922;
        background: #161b22;
        padding: 1 2;
        margin-top: 1;
        margin-bottom: 1;
        height: auto;
    }
    #hf-auth-title { color: #d29922; text-style: bold; margin-bottom: 0; }
    #hf-token      { margin-top: 0; margin-bottom: 0; }

    #back-btn { width: 16; margin-right: 2; }
    #next-btn { width: 22; }
    """

    def compose(self) -> ComposeResult:
        cfg = self.app.train_config

        with Horizontal(id="topbar"):
            yield Static("[bold]✦  llmtune[/]", id="topbar-logo")
            yield Static("Step 2 of 3", id="topbar-step")

        with Vertical(id="content"):
            yield Static("[bold]📊  Configure Dataset[/]", classes="screen-title")
            yield Rule()

            yield Static("Dataset source", classes="section-label")
            yield Static(
                "Enter a local file path  OR  a HuggingFace dataset ID  (e.g. tatsu-lab/alpaca)",
                classes="hint",
            )
            yield Input(
                placeholder="/Users/you/data/dataset.jsonl  or  owner/dataset-name",
                id="dataset-path",
            )

            # Appears automatically when path is a HF dataset ID
            with Vertical(id="hf-auth"):
                yield Static("⚠  HuggingFace dataset — token required", id="hf-auth-title")
                yield Static(
                    "This dataset will be downloaded. Add your token to authenticate.",
                    classes="hint",
                )
                yield Input(
                    value=cfg.hf_token or "",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    password=True,
                    id="hf-token",
                )
                yield Static(
                    "[dim]Get yours free at huggingface.co/settings/tokens[/dim]",
                    classes="hint",
                )

            yield Static("File format", classes="section-label")
            yield Static("Format of your local dataset file (ignored for HF datasets)", classes="hint")
            with RadioSet(id="format-set"):
                yield RadioButton("JSONL  — one JSON object per line  (recommended)", value=True, id="fmt-jsonl")
                yield RadioButton("JSON   — array of objects", id="fmt-json")
                yield RadioButton("CSV    — comma-separated", id="fmt-csv")
                yield RadioButton("Text   — one sample per line", id="fmt-text")

            with Collapsible(title="Advanced options", collapsed=True):
                yield Label("Text field name in your JSON (the key that holds the training text):")
                yield Input(value="text", id="text-col")
                yield Label("Max sequence length in tokens (256–512 recommended for M1/M2 Macs):")
                yield Input(value="512", id="max-seq")

        with Horizontal(id="bottom-nav"):
            yield Static("", id="status")
            yield Button("← Back", id="back-btn", variant="default")
            yield Button("Next →", id="next-btn", variant="primary")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "dataset-path":
            return
        self.query_one("#hf-auth").display = _is_hf_dataset_id(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id != "next-btn":
            return

        cfg = self.app.train_config
        path = self.query_one("#dataset-path", Input).value.strip()

        if not path:
            self.query_one("#status", Static).update("✗ Enter a file path or HuggingFace dataset ID.")
            return

        if _is_hf_dataset_id(path):
            token_override = self.query_one("#hf-token", Input).value.strip()
            if token_override:
                cfg.hf_token = token_override
            cfg.dataset_path = path
        else:
            expanded = str(Path(path).expanduser())
            if not Path(expanded).exists():
                self.query_one("#status", Static).update(
                    "✗ File not found. Check the path, or enter a HuggingFace dataset ID (owner/dataset)."
                )
                return
            cfg.dataset_path = expanded

        fmt_map = {"fmt-jsonl": "jsonl", "fmt-json": "json", "fmt-csv": "csv", "fmt-text": "text"}
        selected = self.query_one("#format-set", RadioSet).pressed_button
        cfg.dataset_format = fmt_map.get(selected.id if selected else "fmt-jsonl", "jsonl")

        text_col = self.query_one("#text-col", Input).value.strip() or "text"
        try:
            max_seq = int(self.query_one("#max-seq", Input).value.strip())
        except ValueError:
            max_seq = 512

        cfg.text_column = text_col
        cfg.max_seq_length = max_seq

        from llmtune.tui.screens.hyperparams import HyperparamsScreen
        self.app.push_screen(HyperparamsScreen())
