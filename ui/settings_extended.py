"""Compatibility export for the organized settings panel."""

from ui.reanalysis import ReanalysisController
from ui.settings import SettingsPanel as _BaseSettingsPanel


class SettingsPanel(_BaseSettingsPanel):
    """Settings panel with the guarded Re-analyze action."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reanalysis_controller = ReanalysisController(self)


__all__ = ["SettingsPanel"]
