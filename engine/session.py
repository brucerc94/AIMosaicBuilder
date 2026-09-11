"""
Session persistence for AI Mosaic Builder.

Saves and restores the full list of ImageRecords next to the source folder's
analysis cache so a project is self-contained and can be moved/copied with
its state.

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


def save_session(source_folder: str, records: list[ImageRecord]) -> None:
    """Persist the current list of ImageRecords inside the source folder."""
    path = _session_path(source_folder)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "source_folder": str(Path(source_folder).expanduser().resolve()),
            "records": [r.to_dict() for r in records],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.info("[session] Saved %d records to %s", len(records), path)
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
