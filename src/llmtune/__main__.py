"""Entry point: `llmtune` CLI command."""

from __future__ import annotations

import sys
import typer

app = typer.Typer(help="Fine-tune any LLM locally with LoRA/QLoRA.")


@app.command()
def run(
    tui: bool = typer.Option(
        False, "--tui", help="Use the terminal UI instead of the browser app."
    ),
    port: int = typer.Option(
        8765, "--port", help="Port for the local web server."
    ),
    no_window: bool = typer.Option(
        False, "--no-window", help="Start server without opening a window."
    ),
    browser: bool = typer.Option(
        False, "--browser", help="Open in system browser instead of native app window."
    ),
) -> None:
    """Launch llmtune — opens a native desktop window by default. Sign-in required."""
    if tui:
        from llmtune.tui.app import LLMTuneApp

        tui_app = LLMTuneApp()
        tui_app.run()
        return

    from llmtune.server.launcher import launch

    launch(port=port, open_window=not no_window, use_browser=browser)


@app.command()
def logout() -> None:
    """Clear stored login credentials."""
    from llmtune.auth import logout as _logout

    _logout()
    typer.echo("Logged out.")


@app.command()
def version() -> None:
    """Print the installed llmtune version."""
    from llmtune import __version__

    typer.echo(f"llmtune {__version__}")


@app.command(name="export-gguf")
def export_gguf(
    model_path: str = typer.Argument(
        ..., help="Path to a fine-tuned adapter or a full model folder."
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Output .gguf path (default: <model>.<quant>.gguf)."
    ),
    quant: str = typer.Option(
        "q8_0", "--quant", "-q", help="Quantization type: q8_0, q4_k_m, or f16."
    ),
    hf_token: str = typer.Option(
        None, "--hf-token", help="HuggingFace token, if the base model is gated."
    ),
) -> None:
    """Merge a fine-tuned adapter into its base model and convert it to GGUF
    (for llama.cpp / Ollama). llama.cpp is set up automatically on first use."""
    from llmtune.training.merge_utils import export_gguf as _export

    out = output or f"{model_path.rstrip('/')}.{quant}.gguf"
    result = _export(model_path, out, quant=quant, hf_token=hf_token, on_log=typer.echo)
    if result.get("status") == "ok":
        typer.echo(f"\n✅ {result['message']}")
    else:
        # merged_only / failure — the merged model is still on disk.
        typer.echo(f"\n⚠️  {result['message']}")
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
