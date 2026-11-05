#!/usr/bin/env python3
"""Color themes and Qt styling for the Tricho desktop app."""

from string import Template

from PySide6.QtGui import QColor, QPalette

LIGHT = {
    "page_bg": "#f5f6f8",
    "card_bg": "#ffffff",
    "card_border": "#e4e7ec",
    "title": "#111827",
    "heading": "#1f2937",
    "body": "#111827",
    "muted": "#6b7280",
    "footer": "#9ca3af",
    "link": "#2a6fb0",
    "success": "#2e7d32",
    "danger": "#c62828",
    "streak": "#14532d",
    "separator": "#eceff3",
    "primary_bg": "#2e7d32",
    "primary_hover": "#276b2b",
    "primary_fg": "#ffffff",
    "secondary_bg": "#ffffff",
    "secondary_fg": "#374151",
    "secondary_border": "#d1d5db",
    "secondary_hover": "#f3f4f6",
    "input_bg": "#ffffff",
    "input_text": "#111827",
    "input_border": "#d1d5db",
    "alert_bg": "#c62828",
    "alert_fg": "#ffffff",
    "safe_bg": "#2e7d32",
    "safe_fg": "#ffffff",
    "tooltip_bg": "#ffffff",
    "tooltip_fg": "#111827",
    "graph_grid": "#e8e8e8",
    "graph_axis": "#a0a0a0",
    "graph_text": "#5a5a5a",
    "graph_nodata": "#8c8c8c",
    "graph_bar": "#c62828",
    "heatmap_label": "#6e6e6e",
    "heatmap_month": "#505050",
    "heatmap_border": "#dcdcdc",
}

DARK = {
    "page_bg": "#15171c",
    "card_bg": "#1b1e24",
    "card_border": "#2a2f38",
    "title": "#f3f4f6",
    "heading": "#e5e7eb",
    "body": "#e5e7eb",
    "muted": "#9ca3af",
    "footer": "#6b7280",
    "link": "#6ea8fe",
    "success": "#4ade80",
    "danger": "#f87171",
    "streak": "#86efac",
    "separator": "#2a2f38",
    "primary_bg": "#16a34a",
    "primary_hover": "#15803d",
    "primary_fg": "#ffffff",
    "secondary_bg": "#1b1e24",
    "secondary_fg": "#e5e7eb",
    "secondary_border": "#3a404b",
    "secondary_hover": "#252a32",
    "input_bg": "#111318",
    "input_text": "#e5e7eb",
    "input_border": "#3a404b",
    "alert_bg": "#b91c1c",
    "alert_fg": "#ffffff",
    "safe_bg": "#166534",
    "safe_fg": "#ffffff",
    "tooltip_bg": "#1b1e24",
    "tooltip_fg": "#e5e7eb",
    "graph_grid": "#2a2f38",
    "graph_axis": "#4b5563",
    "graph_text": "#d1d5db",
    "graph_nodata": "#9ca3af",
    "graph_bar": "#ef4444",
    "heatmap_label": "#9ca3af",
    "heatmap_month": "#cbd5e1",
    "heatmap_border": "#2a2f38",
}

_STYLE = Template("""
QWidget {
    color: $body;
}
QMainWindow, QTabWidget, QTabBar,
QWidget#activityPage, QWidget#settingsPage, QWidget#cameraPage {
    background: $page_bg;
}
QTabWidget::pane {
    border: none;
    background: $page_bg;
}
QFrame#card {
    background: $card_bg;
    border: 1px solid $card_border;
    border-radius: 12px;
}
QWidget#settingsFooter {
    background: $card_bg;
    border-top: 1px solid $card_border;
}
QLabel[role="pageTitle"] {
    font-size: 22px;
    font-weight: 800;
    color: $title;
}
QLabel[role="subtitle"] {
    font-size: 12px;
    color: $muted;
}
QLabel[role="cardTitle"] {
    font-size: 14px;
    font-weight: 700;
    color: $heading;
}
QLabel[role="hint"] {
    color: $muted;
}
QLabel[role="total"] {
    font-size: 13px;
    font-weight: 600;
    color: $danger;
}
QLabel[role="streak"] {
    font-size: 20px;
    font-weight: 700;
    color: $streak;
}
QLabel[role="footer"] {
    color: $footer;
    font-size: 11px;
    padding-top: 4px;
}
QPushButton#saveButton {
    background: $primary_bg;
    color: $primary_fg;
    font-weight: 600;
    padding: 8px 24px;
    border: none;
    border-radius: 8px;
}
QPushButton#saveButton:hover {
    background: $primary_hover;
}
QPushButton#resetButton {
    background: $secondary_bg;
    color: $secondary_fg;
    font-weight: 600;
    padding: 8px 20px;
    border: 1px solid $secondary_border;
    border-radius: 8px;
}
QPushButton#resetButton:hover {
    background: $secondary_hover;
}
QCheckBox {
    color: $body;
}
QToolTip {
    background: $tooltip_bg;
    color: $tooltip_fg;
    border: 1px solid $card_border;
    padding: 4px;
}
""")


def stylesheet(theme):
    return _STYLE.substitute(theme)


def build_palette(theme):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme["page_bg"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme["body"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme["input_bg"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme["card_bg"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme["input_text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme["card_bg"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme["body"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme["primary_bg"]))
    palette.setColor(QPalette.ColorRole.HighlightedText,
                     QColor(theme["primary_fg"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme["tooltip_bg"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme["tooltip_fg"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText,
                     QColor(theme["muted"]))
    palette.setColor(QPalette.ColorRole.Link, QColor(theme["link"]))

    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role,
                         QColor(theme["muted"]))
    return palette
