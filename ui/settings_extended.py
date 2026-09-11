"""Compatibility export for the organized settings panel."""

from PySide6.QtWidgets import QLabel

from ui.reanalysis import ReanalysisController
from ui.settings import SettingsPanel as _BaseSettingsPanel


class SettingsPanel(_BaseSettingsPanel):
    """Settings panel with guarded workflow-state buttons and English UI copy."""

    def __init__(self, *args, **kwargs):
        self._analysis_completed = False
        self._mosaic_generated = False
        super().__init__(*args, **kwargs)
        self.reanalysis_controller = ReanalysisController(self)
        self._localize_custom_request_help()

    def _localize_custom_request_help(self) -> None:
        """Ensure the Custom Request helper text is presented in English."""
        english = (
            "Optional. Gemma evaluates this request during Analyze together with the normal photo analysis. "
            "Examples: ‘only walking’, ‘everyone jumping’, or ‘photos at the beach’."
        )
        for label in self.findChildren(QLabel):
            if label.text().startswith("Optional. This is evaluated by Gemma during Analyze"):
                label.setText(english)
                break

    def set_source_folder(self, folder: str) -> None:
        """Opening a different source starts a fresh UI workflow state."""
        self._analysis_completed = False
        self._mosaic_generated = False
        super().set_source_folder(folder)
        super().set_generate_enabled(False)
        super().set_preview_enabled(False)
        super().set_export_mosaic_enabled(False)

    def set_analyzing(self, active: bool) -> None:
        """Analyze disables downstream actions until analysis completes."""
        if active:
            self._analysis_completed = False
            self._mosaic_generated = False
        super().set_analyzing(active)

    def set_generate_enabled(self, enabled: bool) -> None:
        """Generate is available only after a successful analysis pass exposes candidates."""
        if enabled:
            self._analysis_completed = True
        super().set_generate_enabled(bool(enabled) and self._analysis_completed)

    def set_preview_enabled(self, enabled: bool) -> None:
        """Preview is available only after Generate Mosaic has completed."""
        self._mosaic_generated = bool(enabled)
        super().set_preview_enabled(bool(enabled) and self._mosaic_generated)

    def set_export_mosaic_enabled(self, enabled: bool) -> None:
        """Export is available only after Generate Mosaic has completed."""
        self._mosaic_generated = bool(enabled)
        super().set_export_mosaic_enabled(bool(enabled) and self._mosaic_generated)


__all__ = ["SettingsPanel"]
