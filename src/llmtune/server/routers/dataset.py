"""Dataset validation endpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

PREVIEW_ROWS = 3


class ValidateRequest(BaseModel):
    path: str = ""


def _load_jsonl(expanded: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total = 0
    with open(expanded, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(rows) < PREVIEW_ROWS:
                rows.append(json.loads(line))
    return rows, total


def _load_json(expanded: Path) -> tuple[list[dict], int]:
    with open(expanded, "r", encoding="utf-8") as f:
        data = json.load(f)
    # An array of records, or a single record object.
    records = data if isinstance(data, list) else [data]
    rows = [r for r in records[:PREVIEW_ROWS] if isinstance(r, dict)]
    return rows, len(records)


def _load_csv(expanded: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    total = 0
    with open(expanded, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            total += 1
            if len(rows) < PREVIEW_ROWS:
                rows.append(dict(record))
    return rows, total


def _load_txt(expanded: Path) -> tuple[list[dict], int]:
    # Plain text: one sample per line, exposed under a "text" column.
    rows: list[dict] = []
    total = 0
    with open(expanded, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            if len(rows) < PREVIEW_ROWS:
                rows.append({"text": line})
    return rows, total


# Matches the trainer's extension handling (training/trainer.py::_load_raw_dataset).
_LOADERS = {
    ".jsonl": _load_jsonl,
    ".json": _load_json,
    ".csv": _load_csv,
    ".txt": _load_txt,
}


@router.post("/validate")
def validate_dataset(body: ValidateRequest):
    path = body.path.strip()

    if not path:
        return {"status": "empty"}

    expanded = Path(path).expanduser()

    if not expanded.exists():
        return {"status": "not_found", "message": f"File not found: {path}"}

    if not expanded.is_file():
        return {"status": "not_file", "message": "Path is a directory, not a file"}

    # Unknown/extension-less files fall back to JSONL parsing, like the trainer.
    loader = _LOADERS.get(expanded.suffix.lower(), _load_jsonl)

    try:
        rows, total = loader(expanded)

        # Detect columns
        columns = list(rows[0].keys()) if rows else []

        # Detect format type
        fmt = "unknown"
        if "instruction" in columns and "response" in columns:
            fmt = "instruction + response"
        elif "instruction" in columns and "output" in columns:
            fmt = "instruction + output"
        elif "text" in columns:
            fmt = "text"
        elif "messages" in columns:
            fmt = "chat messages"

        # Estimate avg tokens (chars / 4)
        avg_tokens = 0
        if rows:
            total_chars = sum(len(str(r)) for r in rows)
            avg_tokens = round((total_chars / len(rows)) / 4)

        return {
            "status": "ok",
            "row_count": total,
            "columns": columns,
            "format": fmt,
            "preview": rows[:3],
            "avg_tokens": avg_tokens,
        }
    except json.JSONDecodeError as e:
        return {"status": "parse_error", "message": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
