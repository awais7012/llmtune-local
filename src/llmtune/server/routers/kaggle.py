"""Kaggle dataset search + download via the public Kaggle REST API.

Auth is HTTP Basic (username + API key). Credentials may come from the request,
the KAGGLE_USERNAME / KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json. Downloaded
datasets are extracted under ~/.llmtune/kaggle/ — everything stays on-device.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/kaggle", tags=["kaggle"])

_BASE = "https://www.kaggle.com/api/v1"
_DOWNLOAD_DIR = Path.home() / ".llmtune" / "kaggle"
_DATA_EXTS = {".csv", ".json", ".jsonl", ".txt", ".tsv", ".parquet"}


class Creds(BaseModel):
    username: str | None = None
    key: str | None = None


def _resolve_creds(c: Creds) -> tuple[str, str]:
    user = (c.username or "").strip() or os.getenv("KAGGLE_USERNAME", "").strip()
    key = (c.key or "").strip() or os.getenv("KAGGLE_KEY", "").strip()
    if not (user and key):
        kp = Path.home() / ".kaggle" / "kaggle.json"
        if kp.is_file():
            try:
                data = json.loads(kp.read_text(encoding="utf-8"))
                user = user or str(data.get("username", "")).strip()
                key = key or str(data.get("key", "")).strip()
            except Exception:
                pass
    if not (user and key):
        raise HTTPException(
            status_code=400,
            detail="Kaggle username and API key required. Create one at "
            "kaggle.com/settings → API → Create New Token.",
        )
    return user, key


class SearchReq(Creds):
    query: str


@router.post("/search")
def search(body: SearchReq):
    user, key = _resolve_creds(body)
    q = body.query.strip()
    if not q:
        return {"datasets": []}
    try:
        r = httpx.get(
            f"{_BASE}/datasets/list",
            params={"search": q},
            auth=(user, key),
            timeout=20,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Kaggle: {e}") from e
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="Kaggle rejected your credentials — check your username and API key.")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Kaggle search failed ({r.status_code}).")
    try:
        items = r.json()
    except Exception:
        items = []
    out = []
    for d in items[:25]:
        ref = d.get("ref", "")
        out.append(
            {
                "ref": ref,
                "title": d.get("title", "") or ref,
                "subtitle": d.get("subtitle", "") or "",
                "size_bytes": int(d.get("totalBytes") or 0),
                "downloads": int(d.get("downloadCount") or 0),
                "updated": d.get("lastUpdated", "") or "",
                "url": d.get("url") or f"https://www.kaggle.com/datasets/{ref}",
            }
        )
    return {"datasets": out}


class DownloadReq(Creds):
    ref: str  # owner/dataset-slug


@router.post("/download")
def download(body: DownloadReq):
    user, key = _resolve_creds(body)
    ref = body.ref.strip().strip("/")
    if "/" not in ref:
        raise HTTPException(status_code=400, detail="Invalid dataset ref — expected owner/dataset.")
    owner, slug = ref.split("/", 1)
    dest = _DOWNLOAD_DIR / f"{owner}__{slug}"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "_download.zip"

    try:
        with httpx.stream(
            "GET",
            f"{_BASE}/datasets/download/{owner}/{slug}",
            auth=(user, key),
            follow_redirects=True,
            timeout=None,
        ) as r:
            if r.status_code == 401:
                raise HTTPException(status_code=401, detail="Kaggle rejected your credentials.")
            if r.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied — open this dataset on kaggle.com and accept its rules first.",
                )
            if r.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Dataset '{ref}' not found on Kaggle.")
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Kaggle download failed ({r.status_code}).")
            with zip_path.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Download error: {e}") from e

    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink()
    except zipfile.BadZipFile:
        # Some single-file downloads aren't zipped — keep the raw file.
        pass

    files = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.suffix.lower() in _DATA_EXTS:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            files.append({
                "path": str(p),
                "name": p.name,
                "ext": p.suffix.lower().lstrip("."),
                "size_bytes": size,
            })
    files.sort(key=lambda f: f["size_bytes"], reverse=True)
    return {"folder": str(dest), "files": files}
