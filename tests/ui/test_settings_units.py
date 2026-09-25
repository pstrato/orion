"""Behaviour: Settings exposes unit-alternative choices for display."""

from __future__ import annotations

from orion.ui.preferences import UiPreferences
from orion.ui.results_data import ScalarDatum, apply_unit_alternatives
from orion.ui.settings_units import unit_alternative_options
from orion.ui.units import UNIT_ALTERNATIVES, convert_for_display, display_unit_for


def test_settings_lists_a_selector_per_display_class_including_water():
    options = unit_alternative_options()
    assert [row["canonical"] for row in options] == list(UNIT_ALTERNATIVES)
    water = next(row for row in options if row["canonical"] == "water")
    assert "Water" in str(water["label"])
    water_choices = water["choices"]
    assert isinstance(water_choices, dict)
    assert water_choices == {"mm": "mm", "kg/m^2": "kg/m^2", "kg/ha": "kg/ha"}
    mass = next(row for row in options if row["canonical"] == "kg/m^2")
    mass_choices = mass["choices"]
    assert isinstance(mass_choices, dict)
    assert "mm" not in mass_choices
    assert "biomass" in str(mass["label"]).lower()


def test_settings_tab_keeps_each_control_beside_its_name():
    import inspect

    from orion.ui.app import _render_settings_tab

    source = inspect.getsource(_render_settings_tab)
    assert "setting_row" in source
    assert "render_entity_values" in source
    assert "validate_inputs" not in source
    assert "Cache location" not in source
    assert '.classes("w-full")' not in source


def test_dataset_tree_uses_dense_checkboxes_without_extra_row_chrome():
    import inspect

    from orion.ui.app import _render_dataset_inputs, _render_hierarchy_nodes

    tree = inspect.getsource(_render_hierarchy_nodes)
    host = inspect.getsource(_render_dataset_inputs)
    assert 'props("dense")' in tree
    assert "ui.row(" not in tree
    assert "dataset-tree" in host


def test_apply_unit_alternatives_uses_agronomic_defaults_when_prefs_empty():
    rows = (
        ScalarDatum(name="soil.layers.0.clay", unit="kg/kg", value="0.12", description="Clay"),
        ScalarDatum(name="crop.biomass", unit="kg/m^2", value="0.1", description="Biomass"),
        ScalarDatum(name="soil.layers.0.water", unit="kg/m^2", value="0.1", description="Water"),
        ScalarDatum(name="weather.Ps", unit="kg/m^2", value="2.5", description="Precipitation"),
        ScalarDatum(name="clock.step", unit="step", value="3", description="Step"),
    )
    converted = apply_unit_alternatives(rows, {})
    assert converted[0].unit == "%"
    assert converted[0].value == "12"
    assert converted[1].unit == "kg/ha"
    assert converted[1].value == "1000"
    assert converted[2].unit == "mm"
    assert converted[2].value == "0.1"
    assert converted[3].unit == "mm"
    assert converted[3].value == "2.5"
    assert converted[4] == rows[4]


def test_apply_unit_alternatives_honours_separate_water_preference():
    rows = (
        ScalarDatum(name="crop.biomass", unit="kg/m^2", value="0.1", description="Biomass"),
        ScalarDatum(name="irrigation.amount", unit="kg/m^2", value="0.1", description="Irrigation"),
    )
    prefs = UiPreferences(unit_alternatives={"kg/m^2": "t/ha", "water": "kg/ha"})
    converted = apply_unit_alternatives(rows, prefs.unit_alternatives)
    assert converted[0].unit == "t/ha"
    assert converted[0].value == "1"
    assert converted[1].unit == "kg/ha"
    assert converted[1].value == "1000"


def test_apply_unit_alternatives_rewrites_scalar_value_and_unit():
    rows = (
        ScalarDatum(name="soil.layers.0.clay", unit="kg/kg", value="0.12", description="Clay"),
        ScalarDatum(name="clock.step", unit="step", value="3", description="Step"),
    )
    prefs = UiPreferences(unit_alternatives={"kg/kg": "g/kg"})
    converted = apply_unit_alternatives(rows, prefs.unit_alternatives)
    assert converted[0].unit == "g/kg"
    assert converted[0].value == "120"
    assert converted[1] == rows[1]


def test_display_unit_for_prefers_settings_choice():
    assert display_unit_for("kg/m^2", {"kg/m^2": "t/ha"}) == "t/ha"
    value, unit = convert_for_display(0.32, "kg/m^2", "t/ha")
    assert unit == "t/ha"
    assert abs(value - 3.2) < 1e-9
