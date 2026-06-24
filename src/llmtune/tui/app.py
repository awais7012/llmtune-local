"""Root Textual application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from llmtune.training.config import TrainConfig
from llmtune.training.image_config import ImageTrainConfig

_THEME_CSS = Path(__file__).parent / "theme.css"


class LLMTuneApp(App):
    TITLE = "llmtune"
    THEME = "catppuccin-mocha"
    CSS_PATH = _THEME_CSS

    def __init__(self):
        super().__init__()
        self.train_config = TrainConfig()
        self.image_config = ImageTrainConfig()
        self.train_mode: str = "llm"  # "llm" | "cnn"

    def on_mount(self) -> None:
        from llmtune.auth import is_authenticated
        from llmtune.tui.screens.login import LoginScreen
        from llmtune.tui.screens.model_select import ModelSelectScreen

        if is_authenticated():
            self.push_screen(ModelSelectScreen())
        else:
            self.push_screen(LoginScreen())
