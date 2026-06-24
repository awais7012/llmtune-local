"""Start the local server and open a native desktop window."""

from __future__ import annotations

import platform
import socket
import threading
import time
from pathlib import Path

import uvicorn

from llmtune.server.app import create_app
from llmtune.server.config import DEFAULT_PORT, get_port

# Dark theme background while the React app loads.
_WINDOW_BG = "#12151c"


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_port(start: int = DEFAULT_PORT) -> int:
    for port in range(start, start + 20):
        if _port_available(port):
            return port
    return start


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _window_size() -> tuple[int, int, tuple[int, int]]:
    """Default and minimum window size — fits 1366×768 laptops and up."""
    return 1024, 700, (720, 520)


def _gui_backends() -> list[str | None]:
    """
    Native GUI backends to try, in order.

    None lets pywebview pick its platform default. We still try explicit
    backends first so failures are predictable and we can fall through.
    """
    system = platform.system()
    if system == "Darwin":
        return ["cocoa", None]
    if system == "Windows":
        # edgechromium needs Edge WebView2; mshtml works on every Windows install.
        return ["edgechromium", "mshtml", None]
    # Linux: GTK is common on Ubuntu/Debian; Qt on some distros / KDE.
    return ["gtk", "qt", None]


def _native_window_help() -> str:
    system = platform.system()
    if system == "Darwin":
        return (
            "On macOS, reinstall pywebview: pip install --force-reinstall pywebview\n"
            "If the problem persists, run from Terminal.app (not over SSH)."
        )
    if system == "Windows":
        return (
            "On Windows, install Microsoft Edge WebView2 Runtime:\n"
            "  https://developer.microsoft.com/microsoft-edge/webview2/\n"
            "Or reinstall: pip install --force-reinstall pywebview"
        )
    return (
        "On Linux, install WebKit/GTK or Qt bindings, then reinstall pywebview.\n"
        "  Debian/Ubuntu: sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-webkit2-4.1\n"
        "  Fedora: sudo dnf install python3-gobject gtk3 webkit2gtk4.1\n"
        "  Arch: sudo pacman -S python-gobject gtk3 webkit2gtk-4.1"
    )


def _open_native_window(url: str) -> None:
    """
    Open a native desktop window. Tries each platform GUI backend in turn.

    Raises RuntimeError with install guidance if no backend can start.
    """
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "pywebview is not installed. Run: pip install pywebview"
        ) from exc

    # Extract port from URL for diagnostic
    import re
    port_match = re.search(r":(\d+)", url)
    port = port_match.group(1) if port_match else "8765"

    width, height, min_size = _window_size()
    window = webview.create_window(
        "llmtune",
        url,
        width=width,
        height=height,
        min_size=min_size,
        resizable=True,
        text_select=True,
        background_color=_WINDOW_BG,
    )

    # Inject API base URL before React loads
    def on_loaded():
        try:
            # Inject the server URL so React can use absolute URLs
            window.evaluate_js(f"window.__LLMTUNE_API_BASE__ = 'http://127.0.0.1:{port}';")
            
            # Diagnostic: test if fetch() works inside the webview
            result = window.evaluate_js(f"""
                (async () => {{
                    try {{
                        const r = await fetch('http://127.0.0.1:{port}/health');
                        return 'OK:' + r.status;
                    }} catch(e) {{
                        return 'ERR:' + e.message;
                    }}
                }})()
            """)
            print(f"[llmtune] fetch diagnostic: {result}")
        except Exception as e:
            print(f"[llmtune] fetch diagnostic failed: {e}")

    window.events.loaded += on_loaded

    storage = Path.home() / ".llmtune" / "webview"
    storage.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for gui in _gui_backends():
        label = gui or "default"
        try:
            start_kwargs: dict = {
                "private_mode": False,
                "storage_path": str(storage),
                "http_server": True,  # Allow fetch() to http:// URLs
            }
            if gui is not None:
                start_kwargs["gui"] = gui
            webview.start(**start_kwargs)
            return
        except Exception as exc:
            errors.append(f"  {label}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(
        "llmtune could not open a native window on this machine.\n"
        f"{_native_window_help()}\n"
        f"Backends tried:\n{detail}\n"
        "As a last resort you can use: llmtune run --browser"
    )


def _keep_server_alive(thread: threading.Thread, server: uvicorn.Server) -> None:
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True


def launch(
    port: int | None = None,
    open_window: bool = True,
    use_browser: bool = False,
    host: str = "127.0.0.1",
) -> None:
    """Start uvicorn and open the app in a native window or browser."""
    chosen = port or get_port()
    if not _port_available(chosen):
        chosen = _find_port(chosen)

    url = f"http://{host}:{chosen}"
    app = create_app()

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=chosen, log_level="info"),
    )

    def _run_server() -> None:
        server.run()

    # Non-daemon so the server stays up until we shut it down explicitly.
    thread = threading.Thread(target=_run_server, daemon=False, name="llmtune-server")
    thread.start()

    if not open_window:
        _keep_server_alive(thread, server)
        return

    if not _wait_for_server(url):
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError(f"Server did not start on {url}")

    if use_browser:
        import webbrowser

        webbrowser.open(url)
        _keep_server_alive(thread, server)
        return

    try:
        _open_native_window(url)
    except RuntimeError:
        server.should_exit = True
        thread.join(timeout=5)
        raise

    server.should_exit = True
    thread.join(timeout=5)
