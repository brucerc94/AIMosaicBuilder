"""
AI Mosaic Builder — entry point.

Launches the PySide6 desktop application.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    # Silence overly verbose third-party loggers unless debug mode
    if not debug:
        for noisy in ("PIL", "PIL.Image", "PIL.PngImagePlugin"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    debug = "--debug" in sys.argv or "-d" in sys.argv

    _setup_logging(debug=debug)
    log = logging.getLogger("main")
    log.info("AI Mosaic Builder starting up")

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError as exc:
        print(f"[ERROR] PySide6 not found: {exc}")
        print("Install with: pip install PySide6")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("AI Mosaic Builder")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Mosaic Builder")

    # High-DPI support
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    from ui.main import MainWindow
    window = MainWindow()
    window.show()

    log.info("UI ready — entering event loop")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
