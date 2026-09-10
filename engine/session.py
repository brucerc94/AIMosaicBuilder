"""
Session persistence for AI Mosaic Builder.

Saves and restores the full list of ImageRecords so the user can
stop mid-analysis and resume from where they left off.

Session file: data/session.json
One file per source folder (keyed by folder path hash so different
folders don't overwrite each other).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from engine.models import ImageRecord

logger = logging.getLogger("session")

_SESSION_DIR = Path(__file__).parent.parent / "data" / "sessions"


def _session_path(source_folder: str) -> Path:
    key = hashlib.md5(source_folder.encode()).hexdigest()[:16]
    return _SESSION_DIR / f"session_{key}.json"


def save_session(source_folder: str, records: list[ImageRecord]) -> None:
    """Persist the current list of ImageRecords to disk."""
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(source_folder)
    try:
        data = {
            "source_folder": source_folder,
            "records": [r.to_dict() for r in records],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.info(f"[session] Saved {len(records)} records for {source_folder}")
    except Exception as e:
        logger.error(f"[session] Failed to save session: {e}")


def load_session(source_folder: str) -> Optional[list[ImageRecord]]:
    """
    Load previously saved ImageRecords for the given source folder.
    Returns None if no session exists or it cannot be parsed.
    """
    path = _session_path(source_folder)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("source_folder") != source_folder:
            logger.warning("[session] Source folder mismatch — ignoring saved session")
            return None
        records = [ImageRecord.from_dict(r) for r in data.get("records", [])]
        logger.info(f"[session] Loaded {len(records)} records for {source_folder}")
        return records
    except Exception as e:
        logger.warning(f"[session] Cannot load session: {e}")
        return None


def delete_session(source_folder: str) -> None:
    path = _session_path(source_folder)
    if path.exists():
        path.unlink()
        logger.info(f"[session] Deleted session for {source_folder}")


def session_exists(source_folder: str) -> bool:
    return _session_path(source_folder).exists()


def count_processed(records: list[ImageRecord]) -> int:
    """Number of records that have already been through analysis (any state other than PENDING)."""
    from engine.models import ImageStatus
    return sum(1 for r in records if r.status != ImageStatus.PENDING)
