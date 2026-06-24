"""Login screen — shown when no valid session exists."""

from __future__ import annotations

import threading

from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.screen import Screen
from textual.widgets import Button, Static


class LoginScreen(Screen):
    CSS = """
    LoginScreen {
        background: #0d1117;
        align: center middle;
    }

    #login-title {
        text-align: center;
        text-style: bold;
        color: #58a6ff;
        margin-bottom: 0;
        width: 40;
    }

    #login-tagline {
        text-align: center;
        color: #8b949e;
        margin-bottom: 2;
        width: 40;
    }

    #login-desc {
        text-align: center;
        color: #e6edf3;
        margin-bottom: 2;
        width: 40;
    }

    #login-btn {
        width: 36;
        margin-bottom: 1;
    }

    #status {
        text-align: center;
        height: 1;
        color: #8b949e;
        margin-bottom: 1;
        width: 40;
    }

    #privacy-note {
        text-align: center;
        color: #8b949e;
        width: 40;
    }
    """

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Static("[bold]✦  llmtune[/]", id="login-title")
                yield Static("Local LLM Fine-Tuning", id="login-tagline")
                yield Static(
                    "Fine-tune any AI model on your own computer.\n[dim]No cloud. No cost. 100% private.[/dim]",
                    id="login-desc",
                )
                yield Button("🌐  Login with Browser", id="login-btn", variant="primary")
                yield Static("", id="status")
                yield Static(
                    "[dim]Log in once to activate. Your training data and models never leave this machine.[/dim]",
                    id="privacy-note",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "login-btn":
            self._start_login()

    def _start_login(self) -> None:
        self.query_one("#login-btn", Button).disabled = True
        self.query_one("#status", Static).update("[dim]Opening browser…[/dim]")

        def _do_login():
            from llmtune.auth import login
            try:
                login(open_browser=True)
                self.app.call_from_thread(self._on_success)
            except Exception as e:
                self.app.call_from_thread(self._on_error, str(e))

        threading.Thread(target=_do_login, daemon=True).start()

    def _on_success(self) -> None:
        self.query_one("#status", Static).update("[green]✓ Logged in — loading…[/green]")
        from llmtune.tui.screens.model_select import ModelSelectScreen
        self.app.push_screen(ModelSelectScreen())

    def _on_error(self, msg: str) -> None:
        btn = self.query_one("#login-btn", Button)
        btn.disabled = False
        m = msg.lower()
        if "connect" in m or "network" in m or "errno" in m or "nodename" in m:
            friendly = "Could not connect. Check your internet connection and try again."
        elif "timeout" in m or "timed out" in m:
            friendly = "Login timed out. Please try again."
        else:
            friendly = "Login failed. Please try again."
        self.query_one("#status", Static).update(f"[red]✗ {friendly}[/red]")
