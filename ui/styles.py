"""
UI style constants for AI Mosaic Builder.
"""

DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QLabel {
    background: transparent;
    color: #cdd6f4;
}

QLabel#title_label {
    font-size: 16px;
    font-weight: bold;
    color: #cba6f7;
}

QLabel#section_label {
    font-size: 11px;
    font-weight: bold;
    color: #6c7086;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 28px;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #7f849c;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #45475a;
    border-color: #313244;
}

QPushButton#primary_btn {
    background-color: #cba6f7;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#primary_btn:hover {
    background-color: #d0b0fa;
}

QPushButton#primary_btn:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#danger_btn {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#danger_btn:hover {
    background-color: #f5a0b7;
}

QPushButton#success_btn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
}

QPushButton#success_btn:disabled {
    background-color: #45475a;
    color: #6c7086;
    border: none;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #cba6f7;
    selection-color: #1e1e2e;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #cba6f7;
}

QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 28px;
}

QComboBox:hover { border-color: #7f849c; }
QComboBox:focus { border-color: #cba6f7; }

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
}

QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 28px;
}

QSpinBox:focus, QDoubleSpinBox:focus { border-color: #cba6f7; }

QCheckBox {
    color: #cdd6f4;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #cba6f7;
    border-color: #cba6f7;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #313244;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #cba6f7;
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}

QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    min-height: 18px;
}

QProgressBar::chunk {
    background-color: #cba6f7;
    border-radius: 3px;
}

QScrollArea {
    background-color: #1e1e2e;
    border: none;
}

QScrollBar:vertical {
    background: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover { background: #7f849c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

QScrollBar:horizontal {
    background: #1e1e2e;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

QGroupBox {
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #cba6f7;
}

QSplitter::handle {
    background: #313244;
    width: 2px;
    height: 2px;
}

QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}

QStatusBar {
    background-color: #181825;
    color: #7f849c;
    border-top: 1px solid #313244;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
}

QMenuBar::item:selected {
    background-color: #313244;
}

QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}

QMenu::item:selected {
    background-color: #45475a;
}

QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #181825;
    color: #7f849c;
    border: 1px solid #313244;
    border-bottom: none;
    padding: 6px 14px;
    border-radius: 4px 4px 0 0;
}

QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border-color: #cba6f7;
}

QFrame#separator {
    background-color: #313244;
    max-height: 1px;
}

/* Thumbnail grid card */
QFrame#image_card {
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 8px;
}

QFrame#image_card:hover {
    border-color: #7f849c;
}

QFrame#image_card[selected=true] {
    border-color: #cba6f7;
}

QFrame#image_card[rejected=true] {
    border-color: #f38ba8;
    opacity: 0.7;
}
"""
