# Changelog

All notable changes to **llmtune-local** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.5]

### Changed
- **Removed login entirely — the app now runs with no authentication.** Dropped
  the Auth0 device flow from the server, CLI/TUI, and React frontend; the app
  opens straight into the studio.

### Removed
- `llmtune logout` command and the TUI login screen.
- Auth-only dependencies (`python-jose`, `keyring`) and the Auth0 unit test.

## [0.1.4]

### Added
- README: honest feature-comparison table vs LLaMA-Factory / Axolotl / Torchtune,
  a Screenshots section, a Performance reference point, and a Roadmap.
- `screenshots/` directory for curated UI images.

## [0.1.3]

### Added
- **`llmtune export-gguf` CLI command** — merge a fine-tuned adapter into its
  base model and convert it to a quantized GGUF (`q8_0` / `q4_k_m` / `f16`) for
  use with llama.cpp / Ollama. llama.cpp is set up automatically on first use.
- README badges (PyPI, Python versions, CI, license) and a CLI command reference.

### Changed
- Synced `__version__` with the packaged version.

## [0.1.2]

### Added
- `LICENSE` (MIT) file.
- Unit test suite (`tests/`) and a GitHub Actions CI workflow (Python 3.10–3.12).
- `examples/quickstart.py` — runnable end-to-end fine-tune on the bundled dataset.
- Richer PyPI metadata: classifiers, keywords, and project URLs.

### Changed
- Trimmed the README to user-facing content; moved internal/developer notes out.
- Broadened the package summary to mention image-classifier fine-tuning.

## [0.1.1]

### Changed
- **Authentication migrated from Clerk to Auth0** using the OAuth 2.0 Device
  Authorization Flow. The local server brokers the flow and verifies the Auth0
  ID token via JWKS; the frontend no longer bundles an auth SDK.
- Rewrote the README to reflect the current architecture and install name.

### Fixed
- Dataset validation now supports **CSV, JSON, and plain-text** files in addition
  to JSONL (previously only JSONL was accepted, despite all four being trainable).
- Training **loss display** no longer reports `0.0000` on the final step — it now
  shows the real training loss.
- **CORS** is restricted to `localhost` / `127.0.0.1` instead of reflecting any
  origin, closing a cross-origin access vector for the local server.
- `bitsandbytes` is now gated to Linux so `pip install` no longer fails on macOS
  and Windows (4-bit/8-bit QLoRA is CUDA-only regardless).

## [0.1.0]

### Added
- Initial release: local LoRA/QLoRA fine-tuning for LLMs and fine-tuning for
  image classifiers, with a browser UI and an optional terminal UI.
