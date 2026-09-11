"""Safe re-analysis controller for AI Mosaic Builder."""
from __future__ import annotations

from typing import Iterable

from PySide6.QtWidgets import QMessageBox, QPushButton

from engine.models import ImageRecord, ImageStatus
from ui.workers import AnalysisWorker


class _CacheBypass:
    """Cache adapter that forces fresh inference while preserving cache writes."""

    def __init__(self, cache) -> None:
        self._cache = cache

    def get(self, *_args, **_kwargs):
        return None

    def put(self, *args, **kwargs):
        return self._cache.put(*args, **kwargs)

    def flush(self):
        return self._cache.flush()


class ReanalysisController:
    """Install and handle the guarded Re-analyze action for the settings panel."""

    def __init__(self, panel) -> None:
        self.panel = panel
        self.button = QPushButton("↻  Re-analyze")
        self.button.setToolTip("Run the vision model again and replace the cached analysis.")
        self.button.setEnabled(True)
        self.button.clicked.connect(self.confirm_and_run)
        panel.layout().insertWidget(1, self.button)

    def _window(self):
        return self.panel.window()

    def confirm_and_run(self) -> None:
        window = self._window()
        if any(
            getattr(window, attr, None) is not None
            for attr in ("_loader", "_analysis_worker", "_post_worker", "_generate_worker", "_export_worker")
        ):
            return

        records: list[ImageRecord] = list(getattr(window, "_records", []))
        analyzed = [r for r in records if r.analysis is not None]
        if not analyzed:
            QMessageBox.information(
                window,
                "Re-analyze",
                "There are no analyzed images to re-analyze. Run Analyze first.",
            )
            return
        if not getattr(window._engine, "model_loaded", False):
            QMessageBox.warning(
                window,
                "Re-analyze",
                "The vision model is not loaded. Run Analyze first, then use Re-analyze.",
            )
            return

        selected = [r for r in analyzed if r.status == ImageStatus.SELECTED]
        box = QMessageBox(window)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Re-analyze images?")
        box.setText("This will ignore the cached analysis and run the vision model again.")
        box.setInformativeText(
            "The previous cache entries are not deleted; successful new analyses will replace them."
        )

        all_button = box.addButton(
            f"Re-analyze All ({len(analyzed)})",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        selected_button = None
        if selected:
            selected_button = box.addButton(
                f"Re-analyze Selected ({len(selected)})",
                QMessageBox.ButtonRole.AcceptRole,
            )
        cancel_button = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked is cancel_button:
            return
        if clicked is all_button:
            paths = [r.path for r in analyzed]
            self._start(window, paths, records, selected_only=False)
        elif selected_button is not None and clicked is selected_button:
            paths = [r.path for r in selected]
            self._start(window, paths, records, selected_only=True)

    def _start(
        self,
        window,
        paths: list[str],
        original_records: list[ImageRecord],
        selected_only: bool,
    ) -> None:
        if not paths:
            return

        self.button.setEnabled(False)
        window.settings_panel.set_analyzing(True)
        window.settings_panel.set_generate_enabled(False)
        window.settings_panel.set_preview_enabled(False)
        window.settings_panel.set_export_mosaic_enabled(False)
        window._summary.setText(
            f"Re-analyzing {'selected' if selected_only else 'all'} images… {len(paths)} image(s)"
        )

        bypass_cache = _CacheBypass(window._cache)
        worker = AnalysisWorker(window._engine, bypass_cache, window._settings, paths)
        worker.signals.progress.connect(window._on_analysis_progress)
        worker.signals.finished.connect(
            lambda fresh: self._finished(window, fresh, original_records, selected_only)
        )
        worker.signals.error.connect(lambda message: self._error(window, message))
        window._analysis_worker = worker
        window._pool.start(worker)

    def _finished(
        self,
        window,
        fresh: list[ImageRecord],
        original_records: list[ImageRecord],
        selected_only: bool,
    ) -> None:
        if selected_only:
            by_path = {record.path: record for record in fresh}
            merged: list[ImageRecord] = []
            for original in original_records:
                replacement = by_path.get(original.path)
                if replacement is None:
                    merged.append(original)
                    continue
                replacement.manually_included = original.manually_included
                replacement.manually_excluded = original.manually_excluded
                merged.append(replacement)
        else:
            merged = fresh

        self.button.setEnabled(True)
        window._on_analysis_finished(merged)

    def _error(self, window, message: str) -> None:
        self.button.setEnabled(True)
        window._on_worker_error(message)
