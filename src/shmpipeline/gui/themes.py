"""GUI theme definitions for the Qt desktop app."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class ThemeDefinition:
    """Color tokens shared by the GUI and viewer widgets."""

    name: str
    window: str
    base: str
    alternate_base: str
    text: str
    muted_text: str
    button: str
    button_text: str
    highlight: str
    highlight_text: str
    accent: str
    border: str
    plot_background: str
    plot_foreground: str
    success: str
    error: str
    success_bg: str
    error_bg: str


_THEMES = {
    "light": ThemeDefinition(
        name="light",
        window="#f3f0e8",
        base="#fffdf8",
        alternate_base="#ebe6da",
        text="#1f262d",
        muted_text="#6e756f",
        button="#e7e1d1",
        button_text="#1f262d",
        highlight="#d87a3a",
        highlight_text="#fffdf8",
        accent="#2d6a8a",
        border="#c9c0ae",
        plot_background="#fffdf8",
        plot_foreground="#24313a",
        success="#1f6a3c",
        error="#8d2a2a",
        success_bg="#d9f0df",
        error_bg="#f5dddd",
    ),
    "dark": ThemeDefinition(
        name="dark",
        window="#1d2126",
        base="#14181d",
        alternate_base="#232a31",
        text="#e7e2d8",
        muted_text="#a5aa9d",
        button="#2a3138",
        button_text="#e7e2d8",
        highlight="#d98946",
        highlight_text="#11151a",
        accent="#68b0c9",
        border="#3b4650",
        plot_background="#14181d",
        plot_foreground="#d7d5cf",
        success="#73c08e",
        error="#f28a8a",
        success_bg="#21422c",
        error_bg="#512424",
    ),
}


def resolve_theme(theme_name: str | None) -> ThemeDefinition:
    """Return one of the supported GUI themes."""
    if theme_name is None:
        return _THEMES["light"]
    return _THEMES.get(theme_name, _THEMES["light"])


def build_palette(theme: ThemeDefinition) -> QPalette:
    """Build the Qt palette for one theme."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(theme.window))
    palette.setColor(QPalette.WindowText, QColor(theme.text))
    palette.setColor(QPalette.Base, QColor(theme.base))
    palette.setColor(QPalette.AlternateBase, QColor(theme.alternate_base))
    palette.setColor(QPalette.ToolTipBase, QColor(theme.base))
    palette.setColor(QPalette.ToolTipText, QColor(theme.text))
    palette.setColor(QPalette.Text, QColor(theme.text))
    palette.setColor(QPalette.Button, QColor(theme.button))
    palette.setColor(QPalette.ButtonText, QColor(theme.button_text))
    palette.setColor(QPalette.Highlight, QColor(theme.highlight))
    palette.setColor(QPalette.HighlightedText, QColor(theme.highlight_text))
    palette.setColor(QPalette.Light, QColor(theme.alternate_base))
    palette.setColor(QPalette.Mid, QColor(theme.border))
    palette.setColor(QPalette.Dark, QColor(theme.border))
    palette.setColor(QPalette.PlaceholderText, QColor(theme.muted_text))
    return palette


def apply_application_theme(app, theme_name: str | None) -> ThemeDefinition:
    """Apply a theme to Qt and pyqtgraph surfaces."""
    theme = resolve_theme(theme_name)
    app.setPalette(build_palette(theme))
    app.setStyleSheet(
        "QGroupBox {"
        f" border: 1px solid {theme.border};"
        " border-radius: 6px;"
        " margin-top: 12px;"
        " padding-top: 8px;"
        "}"
        "QGroupBox::title {"
        " subcontrol-origin: margin;"
        " left: 10px;"
        " padding: 0 4px;"
        "}"
    )
    pg.setConfigOption("background", theme.plot_background)
    pg.setConfigOption("foreground", theme.plot_foreground)
    return theme
