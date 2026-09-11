"""
Storage layer for AI Mosaic Builder.
Plain JSON files — no encryption, no database, no cloud.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from engine.models import AppSettings

logger = logging.getLogger("storage")

DATA_DIR     = Path(__file__).parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Settings ─────────────────────────────────────────────────────────────────

def load_settings() -> AppSettings:
    ensure_data_dir()
    if SETTINGS_FILE.exists():
        try:
            text = SETTINGS_FILE.read_text(encoding="utf-8")
            payload = json.loads(text)
            if "n_ctx" not in payload:
                payload["n_ctx"] = 4096
            settings = AppSettings.from_dict(payload)
            # max_tokens was added after the original dataclass schema; keep it
            # backward-compatible with existing settings.json files.
            settings.max_tokens = max(64, min(4096, int(payload.get("max_tokens", 576))))
            return settings
        except Exception as e:
            logger.warning(f"[storage] Failed to load settings: {e} — using defaults")
    settings = AppSettings()
    settings.n_ctx = 4096
    settings.max_tokens = 576
    return settings


def save_settings(settings: AppSettings) -> None:
    ensure_data_dir()
    try:
        payload = settings.to_dict()
        payload["n_ctx"] = int(getattr(settings, "n_ctx", 4096))
        payload["max_tokens"] = max(64, min(4096, int(getattr(settings, "max_tokens", 576))))
        SETTINGS_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error(f"[storage] Failed to save settings: {e}")


# ─── GGUF discovery ───────────────────────────────────────────────────────────

def list_gguf_models(directory: str) -> list[str]:
    """Return all .gguf files found recursively under directory."""
    if not directory or not os.path.isdir(directory):
        return []
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".gguf"):
                result.append(os.path.join(root, f))
    return sorted(result)
