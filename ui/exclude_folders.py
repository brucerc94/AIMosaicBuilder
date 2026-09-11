"""Dialog for selecting source subfolders excluded from image discovery."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)


class ExcludeFoldersDialog(QDialog):
    """Let the user check/uncheck source subfolders to ignore during Analyze."""

    def __init__(
        self,
        source_folder: str,
        excluded_folders: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exclude Folders")
        self.resize(560, 620)
        self._source = Path(source_folder).expanduser().resolve()
        self._updating = False
        self._excluded = {
            self._normalize_relative(value)
            for value in (excluded_folders or [])
            if self._normalize_relative(value)
        }
        self._items_by_path: dict[str, QTreeWidgetItem] = {}
        self._build_ui()
        self._populate()

    @staticmethod
    def _normalize_relative(value: str) -> str:
        text = str(value or "").strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        text = text.strip("/")
        if not text or text == "." or text.startswith("../") or text == "..":
            return ""
        return text

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            "Checked folders will be ignored by Analyze and will not appear in the image list. "
            "Your existing analysis cache is kept."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Folder (checked = excluded)"])
        self._tree.setAlternatingRowColors(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree, 1)

        actions = QHBoxLayout()
        self._check_all = QPushButton("Check All")
        self._uncheck_all = QPushButton("Uncheck All")
        self._check_all.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        self._uncheck_all.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        actions.addWidget(self._check_all)
        actions.addWidget(self._uncheck_all)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        directories: list[tuple[str, Path]] = []
        for path in sorted(self._source.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_dir():
                continue
            try:
                relative = path.relative_to(self._source).as_posix()
            except ValueError:
                continue
            # Never expose the application's own project metadata directory.
            if relative == ".aimosaic" or relative.startswith(".aimosaic/"):
                continue
            directories.append((relative, path))

        self._updating = True
        try:
            self._tree.clear()
            roots: dict[str, QTreeWidgetItem] = {}
            for relative, _path in directories:
                parts = relative.split("/")
                parent_item = None
                accumulated: list[str] = []
                for part in parts:
                    accumulated.append(part)
                    key = "/".join(accumulated)
                    item = self._items_by_path.get(key) if parent_item is None else None
                    if item is None:
                        if parent_item is None:
                            item = roots.get(key)
                        else:
                            # Find an existing child under this parent.
                            item = next(
                                (parent_item.child(i) for i in range(parent_item.childCount()) if parent_item.child(i).text(0) == part),
                                None,
                            )
                        if item is None:
                            item = QTreeWidgetItem(parent_item if parent_item is not None else self._tree)
                            item.setText(0, part)
                            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                            item.setCheckState(0, Qt.CheckState.Unchecked)
                    self._items_by_path[key] = item
                    if parent_item is None:
                        roots[key] = item
                    parent_item = item

            # Apply saved exclusions. A checked parent automatically checks all descendants.
            for relative in sorted(self._excluded, key=lambda value: (value.count("/"), value)):
                item = self._items_by_path.get(relative)
                if item is not None:
                    item.setCheckState(0, Qt.CheckState.Checked)
        finally:
            self._updating = False
        self._tree.expandAll()

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating:
            return
        state = item.checkState(0)
        if state not in (Qt.CheckState.Checked, Qt.CheckState.Unchecked):
            return
        self._updating = True
        try:
            for index in range(item.childCount()):
                item.child(index).setCheckState(0, state)
        finally:
            self._updating = False

    def _set_all(self, state: Qt.CheckState) -> None:
        self._updating = True
        try:
            for index in range(self._tree.topLevelItemCount()):
                self._tree.topLevelItem(index).setCheckState(0, state)
        finally:
            self._updating = False

    def excluded_folders(self) -> list[str]:
        """Return the smallest set of checked folders (parents replace checked children)."""
        result: list[str] = []
        for relative, item in sorted(self._items_by_path.items(), key=lambda pair: (pair[0].count("/"), pair[0])):
            if item.checkState(0) != Qt.CheckState.Checked:
                continue
            if any(relative == parent or relative.startswith(parent + "/") for parent in result):
                continue
            result.append(relative)
        return result
