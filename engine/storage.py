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

DATA_DIR = Path(__file__).parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_custom_prompt(value: Any) -> str:
    return str(value or "").strip()[:2000]


def _normalize_custom_prompt_weight(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 30.0


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
            settings.max_tokens = max(64, min(4096, int(payload.get("max_tokens", 576))))
            settings.min_subject_percent = max(1.0, min(50.0, float(payload.get("min_subject_percent", 10.0))))
            settings.target_subject_percent = max(
                settings.min_subject_percent,
                min(75.0, float(payload.get("target_subject_percent", 15.0))),
            )
            # Custom prompt settings are intentionally attached here rather than
            # changing the core dataclass schema, keeping older settings files compatible.
            settings.custom_prompt = _normalize_custom_prompt(payload.get("custom_prompt", ""))
            settings.custom_prompt_weight = _normalize_custom_prompt_weight(payload.get("custom_prompt_weight", 30.0))
            return settings
        except Exception as e:
            logger.warning(f"[storage] Failed to load settings: {e} — using defaults")
    settings = AppSettings()
    settings.n_ctx = 4096
    settings.max_tokens = 576
    settings.min_subject_percent = 10.0
    settings.target_subject_percent = 15.0
    settings.custom_prompt = ""
    settings.custom_prompt_weight = 30.0
    return settings


def save_settings(settings: AppSettings) -> None:
    ensure_data_dir()
    try:
        payload = settings.to_dict()
        payload["n_ctx"] = int(getattr(settings, "n_ctx", 4096))
        payload["max_tokens"] = max(64, min(4096, int(getattr(settings, "max_tokens", 576))))
        min_subject_percent = max(1.0, min(50.0, float(getattr(settings, "min_subject_percent", 10.0))))
        target_subject_percent = max(
            min_subject_percent,
            min(75.0, float(getattr(settings, "target_subject_percent", 15.0))),
        )
        payload["min_subject_percent"] = min_subject_percent
        payload["target_subject_percent"] = target_subject_percent
        payload["custom_prompt"] = _normalize_custom_prompt(getattr(settings, "custom_prompt", ""))
        payload["custom_prompt_weight"] = _normalize_custom_prompt_weight(getattr(settings, "custom_prompt_weight", 30.0))
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
