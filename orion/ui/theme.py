"""Orion UI visual language: mark, light-blue palette, dense adaptive chrome."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from nicegui import ui

PRIMARY = "#3D7FBF"
SECONDARY = "#7EB6D9"
ACCENT = "#5BB8D4"
PAGE_BG = "#E8F2FA"
HEADER_FG = "#F7FCFF"

ICON_PATH = Path(__file__).resolve().parent / "static" / "orion-mark.png"
FAVICON_PATH = Path(__file__).resolve().parent / "static" / "orion-mark.svg"

DENSE_CSS = f"""
.orion-app .q-header {{
  min-height: 40px;
  color: {HEADER_FG};
}}
.orion-app .q-tab {{
  min-height: 32px !important;
  padding: 0 8px;
  font-size: 12px;
}}
.orion-app .q-tab-panel {{
  padding: 4px 8px;
}}
.orion-app .q-field--standard .q-field__control {{
  min-height: 28px;
}}
.orion-app .q-btn {{
  min-height: 26px;
  padding: 0 8px;
}}
.orion-app .q-card {{
  border-radius: 6px;
}}
.orion-app .q-item {{
  min-height: 22px;
  padding: 0 6px;
}}
.orion-app .q-checkbox {{
  min-height: 18px;
  margin: 0;
}}
.orion-app .q-checkbox .q-checkbox__inner {{
  font-size: 16px;
  width: 16px;
  min-width: 16px;
  height: 16px;
}}
.orion-app .q-checkbox .q-checkbox__label,
.orion-app .q-toggle .q-toggle__label {{
  font-size: 12px;
  padding: 0 0 0 4px;
  line-height: 16px;
}}
.orion-app .q-toggle {{
  margin: 0;
}}
.orion-app .q-tree .q-tree__node-header {{
  min-height: 20px;
  padding: 0 4px;
}}
.orion-setting {{
  display: flex;
  align-items: center;
  gap: 8px !important;
  width: max-content;
  max-width: 100%;
}}
.orion-setting-name {{
  flex: 0 0 13rem;
  font-size: 12px;
  line-height: 1.2;
  color: #4b5563;
}}
.orion-setting-value {{
  width: 10rem;
  max-width: 10rem;
  flex: 0 0 10rem;
}}
.orion-setting-value--wide {{
  width: 28rem;
  max-width: min(28rem, 70vw);
  flex: 0 1 28rem;
}}
.orion-setting .q-field {{
  width: 100%;
}}
.orion-setting .q-field__control {{
  min-height: 28px !important;
}}
.dataset-tree {{
  line-height: 1.15;
}}
.dataset-tree .q-checkbox {{
  padding: 0 2px;
  height: 18px;
}}
.orion-config-list-item {{
  padding: 2px 6px;
  border-radius: 4px;
}}
.orion-config-list-item--selected {{
  background: rgba(61, 127, 191, 0.18);
}}
.orion-input-slot {{
  padding: 2px 6px;
  border-radius: 4px;
}}
.orion-input-slot--selected {{
  background: rgba(61, 127, 191, 0.18);
}}
.orion-mark {{
  width: 28px;
  height: 28px;
  border-radius: 6px;
  flex-shrink: 0;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.25);
}}
.plot-drag-handle {{
  user-select: none;
  touch-action: none;
}}
.orion-fetch-bar,
.orion-fetch-bar .q-linear-progress__model,
.orion-fetch-bar .q-linear-progress__track {{
  transition: none !important;
  animation: none !important;
}}
@media (max-width: 800px) {{
  .orion-tagline {{ display: none; }}
  .orion-app .q-tab {{ padding: 0 4px; font-size: 12px; }}
  .orion-title {{ font-size: 1rem; }}
  .orion-setting-name {{ flex-basis: 9rem; }}
}}
"""


@contextmanager
def setting_row(name: str, *, wide: bool = False) -> Iterator[None]:
    """One compact row: name on the left, control boxed to the value (not the page)."""
    value_class = "orion-setting-value--wide" if wide else "orion-setting-value"
    with ui.row().classes("orion-setting items-center no-wrap"):
        ui.label(name).classes("orion-setting-name")
        with ui.element("div").classes(value_class):
            yield


def apply_theme() -> None:
    """Apply light-blue primary colour, page background, and dense controls."""
    ui.dark_mode().disable()
    ui.colors(primary=PRIMARY, secondary=SECONDARY, accent=ACCENT)
    ui.query("body").classes("orion-app")
    ui.query("body").style(f"background:{PAGE_BG}")
    ui.add_css(DENSE_CSS)
