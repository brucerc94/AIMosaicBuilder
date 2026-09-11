"""AI Mosaic Builder entry point."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from engine.target_subject import install_target_subject_bias

# Install before ui.main imports its workers so every layout optimization call
# uses the requested Target Subject value when comparing layouts.
install_target_subject_bias()

from ui.main import MainWindow
from ui.styles import DARK_STYLESHEET


def main() -> int:
    debug = "--debug" in sys.argv or "-d" in sys.argv
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AI Mosaic Builder")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("AI Mosaic Builder")
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
