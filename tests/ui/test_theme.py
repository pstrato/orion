"""Behaviour: Orion mark, light-blue palette, and dense adaptive chrome."""

from __future__ import annotations

import re

from orion.ui.configuration import DEFAULT_CONFIGURATION_COLORS, default_configuration
from orion.ui.theme import DENSE_CSS, FAVICON_PATH, ICON_PATH, PAGE_BG, PRIMARY


def test_app_mark_files_exist():
    assert ICON_PATH.is_file()
    assert FAVICON_PATH.is_file()
    assert ICON_PATH.suffix == ".png"
    assert FAVICON_PATH.suffix == ".svg"


def test_favicon_svg_draws_a_leaf_and_seven_orion_stars():
    svg = FAVICON_PATH.read_text(encoding="utf-8")
    assert "leaf" in svg.lower()
    assert "orion" in svg.lower()
    assert svg.count("<circle") == 7


def test_orion_stars_are_vertices_of_the_leaf():
    svg = FAVICON_PATH.read_text(encoding="utf-8")
    stars = [(int(float(x)), int(float(y))) for x, y in re.findall(r'<circle\b[^>]*\bcx="([\d.]+)"[^>]*\bcy="([\d.]+)"', svg)]
    assert len(stars) == 7
    path_data = "".join(re.findall(r'\bd="([^"]+)"', svg)).replace(" ", "")
    for x, y in stars:
        assert f"{x},{y}" in path_data, f"star {(x, y)} is not a vertex of the leaf path"


def test_theme_palette_is_light_blue():
    assert PRIMARY.lower() == "#3d7fbf"
    assert PAGE_BG.lower() == "#e8f2fa"


def test_dense_css_tightens_chrome_and_adapts_on_narrow_viewports():
    assert ".orion-app .q-tab" in DENSE_CSS
    assert "min-height: 32px" in DENSE_CSS
    assert ".orion-app .q-tab-panel" in DENSE_CSS
    assert "padding: 4px 8px" in DENSE_CSS
    assert "@media (max-width: 800px)" in DENSE_CSS
    assert ".orion-tagline { display: none; }" in DENSE_CSS


def test_dense_css_keeps_setting_values_next_to_their_names():
    assert ".orion-setting" in DENSE_CSS
    assert ".orion-setting-name" in DENSE_CSS
    assert "max-width: 10rem" in DENSE_CSS


def test_dense_css_packs_dataset_tree_checkboxes():
    assert ".dataset-tree" in DENSE_CSS
    assert ".q-checkbox__inner" in DENSE_CSS
    assert "font-size: 16px" in DENSE_CSS


def test_default_configuration_color_matches_theme_primary():
    assert DEFAULT_CONFIGURATION_COLORS[0] == PRIMARY
    assert default_configuration().color == PRIMARY
