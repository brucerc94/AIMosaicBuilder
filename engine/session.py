"""
Session persistence for AI Mosaic Builder.

Saves and restores the full list of ImageRecords and source-folder exclusions
next to the analysis cache so a project is self-contained.

Session file:
    <source>/.aimosaic/session.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from engine.models import ImageRecord

logger = logging.getLogger("session")

_SESSION_FILENAME = "session.json"
_SESSION_DIRNAME = ".aimosaic"


def _session_path(source_folder: str) -> Path:
    source = Path(source_folder).expanduser().resolve()
    return source / _SESSION_DIRNAME / _SESSION_FILENAME


def _normalize_excluded_folder(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.strip("/")
    if not text or text == "." or text == ".." or text.startswith("../"):
        return ""
    return text


def _read_existing_exclusions(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        values = parsed.get("excluded_folders", []) if isinstance(parsed, dict) else []
        if not isinstance(values, list):
            return []
        return sorted({item for item in (_normalize_excluded_folder(v) for v in values) if item})
    except Exception:
        return []


def save_session(
    source_folder: str,
    records: list[ImageRecord],
    excluded_folders: list[str] | None = None,
) -> None:
    """Persist records and source-folder exclusions inside the source folder."""
    path = _session_path(source_folder)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        exclusions = (
            sorted({item for item in (_normalize_excluded_folder(v) for v in excluded_folders or []) if item})
            if excluded_folders is not None
            else _read_existing_exclusions(path)
        )
        data = {
            "source_folder": str(Path(source_folder).expanduser().resolve()),
            "excluded_folders": exclusions,
            "records": [r.to_dict() for r in records],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.info(
            "[session] Saved %d records and %d excluded folder(s) to %s",
            len(records),
            len(exclusions),
            path,
        )
    except Exception as e:
        logger.error("[session] Failed to save session: %s", e)


def load_session(source_folder: str) -> Optional[list[ImageRecord]]:
    """Load previously saved ImageRecords for the given source folder."""
    path = _session_path(source_folder)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected_source = str(Path(source_folder).expanduser().resolve())
        saved_source = str(data.get("source_folder", ""))
        if saved_source and str(Path(saved_source).expanduser().resolve()) != expected_source:
            logger.warning("[session] Source folder mismatch — ignoring saved session")
            return None
        records = [ImageRecord.from_dict(r) for r in data.get("records", [])]
        logger.info("[session] Loaded %d records from %s", len(records), path)
        return records
    except Exception as e:
        logger.warning("[session] Cannot load %s: %s", path, e)
        return None


def load_excluded_folders(source_folder: str) -> list[str]:
    """Load normalized relative subfolders excluded for this source folder."""
    path = _session_path(source_folder)
    exclusions = _read_existing_exclusions(path)
    logger.info("[session] Loaded %d excluded folder(s) from %s", len(exclusions), path)
    return exclusions


def delete_session(source_folder: str) -> None:
    path = _session_path(source_folder)
    if path.exists():
        path.unlink()
        logger.info("[session] Deleted session %s", path)


def session_exists(source_folder: str) -> bool:
    return _session_path(source_folder).exists()


def count_processed(records: list[ImageRecord]) -> int:
    """Number of records that have already been through analysis."""
    from engine.models import ImageStatus
    return sum(1 for r in records if r.status != ImageStatus.PENDING)
