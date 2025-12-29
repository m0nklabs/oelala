"""Civitai API client utilities.

Minimal wrapper used by Oelala backend to:
- Search checkpoints on Civitai
- Download a chosen model file into ComfyUI model folders

Auth:
- Set CIVITAI_API_TOKEN in the backend environment for private/age-gated downloads.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


CIVITAI_BASE_URL = "https://civitai.com/api/v1"


def _safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:240] if len(name) > 240 else name


@dataclass(frozen=True)
class CivitaiFileChoice:
    file_id: int
    name: str
    size_kb: Optional[int]
    download_url: str
    primary: bool


class CivitaiClient:
    def __init__(self, token: Optional[str] = None, timeout: int = 60):
        self.token = token or os.getenv("CIVITAI_API_TOKEN") or None
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "oelala-backend/1.0",
                "Accept": "application/json",
            }
        )

    def search_models(self, query: str, limit: int = 10, types: Optional[List[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"query": query, "limit": limit}
        if types:
            # Civitai supports repeated 'types' query params; requests handles list values
            params["types"] = types
        resp = self.session.get(f"{CIVITAI_BASE_URL}/models", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_model_version(self, version_id: int) -> Dict[str, Any]:
        resp = self.session.get(f"{CIVITAI_BASE_URL}/model-versions/{version_id}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def choose_file(
        self,
        version_payload: Dict[str, Any],
        file_id: Optional[int] = None,
        prefer_primary: bool = True,
    ) -> CivitaiFileChoice:
        files = version_payload.get("files") or []
        if not files:
            raise ValueError("Civitai model version has no files")

        def to_choice(f: Dict[str, Any]) -> CivitaiFileChoice:
            return CivitaiFileChoice(
                file_id=int(f.get("id")),
                name=str(f.get("name") or "model.safetensors"),
                size_kb=f.get("sizeKB"),
                download_url=str(f.get("downloadUrl") or ""),
                primary=bool(f.get("primary", False)),
            )

        choices = [to_choice(f) for f in files if f.get("downloadUrl")]
        if not choices:
            raise ValueError("Civitai model version has no downloadable files")

        if file_id is not None:
            for c in choices:
                if c.file_id == int(file_id):
                    return c
            raise ValueError(f"file_id={file_id} not found in version files")

        if prefer_primary:
            primaries = [c for c in choices if c.primary]
            if primaries:
                return primaries[0]

        # Fallback: pick the largest file (often the actual checkpoint)
        choices_sorted = sorted(
            choices,
            key=lambda c: (c.size_kb or 0),
            reverse=True,
        )
        return choices_sorted[0]

    def _download_url_with_token(self, url: str) -> str:
        if not self.token:
            return url
        # Civitai expects token as query param
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}token={self.token}"

    def download_file(
        self,
        url: str,
        dest_path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")

        download_url = self._download_url_with_token(url)
        with self.session.get(download_url, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        tmp_path.replace(dest_path)
        return dest_path

    def download_checkpoint_from_version(
        self,
        version_id: int,
        dest_dir: Path,
        file_id: Optional[int] = None,
        filename_hint: Optional[str] = None,
    ) -> Path:
        payload = self.get_model_version(version_id)
        choice = self.choose_file(payload, file_id=file_id)

        version_name = str(payload.get("name") or f"version_{version_id}")
        base_name = filename_hint or choice.name or version_name
        safe_name = _safe_filename(base_name)

        # Ensure we keep the original extension if present
        if "." not in safe_name and "." in choice.name:
            safe_name = safe_name + Path(choice.name).suffix

        # Namespace by version id to avoid collisions
        final_name = _safe_filename(f"civitai_{version_id}_{safe_name}")
        dest_path = dest_dir / final_name

        # If already downloaded, return as-is
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path

        # Small backoff for transient CDN hiccups
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                return self.download_file(choice.download_url, dest_path)
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt)

        raise RuntimeError(f"Failed to download after retries: {last_err}")
