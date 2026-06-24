"""Dataset configuration screen for CNN / image-classification training."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Rule, Static


class ImageDatasetScreen(Screen):
    CSS = """
    ImageDatasetScreen { background: #0d1117; }

    #dataset-path { margin-bottom: 0; }

    #format-box {
        border: tall #30363d;
        background: #161b22;
        padding: 1 2;
        margin-top: 1;
        height: auto;
    }
    #format-title { color: #58a6ff; text-style: bold; margin-bottom: 0; }

    #back-btn { width: 16; margin-right: 2; }
    #next-btn { width: 22; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("[bold]✦  llmtune[/]", id="topbar-logo")
            yield Static("Step 2 of 3", id="topbar-step")

        with Vertical(id="content"):
            yield Static("[bold]🖼  Image Dataset[/]", classes="screen-title")
            yield Rule()

            yield Static("Dataset folder", classes="section-label")
            yield Static(
                "Organise images in subfolders — one folder per class:",
                classes="hint",
            )
            yield Static(
                "  dataset/\n"
                "    cat/   img1.jpg  img2.jpg  …\n"
                "    dog/   img3.jpg  img4.jpg  …\n\n"
                "Or use train/validation splits:\n"
                "  dataset/\n"
                "    train/cat/  train/dog/\n"
                "    validation/cat/  validation/dog/",
                classes="hint",
            )
            yield Input(
                placeholder="/Users/you/datasets/my_images",
                id="dataset-path",
            )

            with Vertical(id="format-box"):
                yield Static("What gets auto-detected", id="format-title")
                yield Static(
                    "• Class names  — from subfolder names\n"
                    "• Number of classes  — counted automatically\n"
                    "• Train / val split  — from folder structure or 90/10 random split\n"
                    "• Image size  — configurable in the next screen (default 224×224)",
                    classes="hint",
                )

        with Horizontal(id="bottom-nav"):
            yield Static("", id="status")
            yield Button("← Back", id="back-btn", variant="default")
            yield Button("Next →", id="next-btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
            return
        if event.button.id != "next-btn":
            return

        path = self.query_one("#dataset-path", Input).value.strip()
        if not path:
            self.query_one("#status", Static).update("✗ Enter the path to your image dataset folder.")
            return

        expanded = str(Path(path).expanduser())
        if not Path(expanded).is_dir():
            self.query_one("#status", Static).update(
                "✗ Folder not found — check the path. Must be a directory containing class subfolders."
            )
            return

        self.app.image_config.dataset_path = expanded

        from llmtune.tui.screens.image_hyperparams import ImageHyperparamsScreen
        self.app.push_screen(ImageHyperparamsScreen())
