"""Compatibility export for the organized settings panel."""

from PySide6.QtWidgets import QLabel

from ui.reanalysis import ReanalysisController
from ui.settings import SettingsPanel as _BaseSettingsPanel


class SettingsPanel(_BaseSettingsPanel):
    """Settings panel with the guarded Re-analyze action and English UI copy."""

    def __init__(self, *args, **kwargs):
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


__all__ = ["SettingsPanel"]
