"""Behaviour: the settings view follows the Settings entity."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from orion.core.entity import entity
from orion.core.entity import Entity
from orion.core.setting import Settings
from orion.ui.app import build_model, ui_settings
from orion.ui.configuration import default_configuration
from orion.ui.reflect import list_entity_values, set_entity_value


def test_settings_view_lists_every_field_on_the_settings_entity(tmp_path: Path):
    settings = ui_settings(cache_path=tmp_path, validate_inputs=True)
    listed = list_entity_values(settings)

    declared = [field.name for field in fields(Settings) if field.name != "name"]
    assert [item.name for item in listed] == declared
    by_name = {item.name: item for item in listed}
    assert by_name["cache_path"].value == tmp_path
    assert by_name["cache_path"].kind == "path"
    assert "cache" in by_name["cache_path"].description.lower()
    assert by_name["validate_inputs"].value is True
    assert by_name["validate_inputs"].kind == "bool"
    assert by_name["validate_inputs"].description.startswith("Validate input")
    assert by_name["validate_initial_states"].value is False
    assert by_name["validate_simulation_states"].value is False


def test_nested_entity_fields_are_discovered_with_the_entity_walk():
    @entity()
    class Noted(Entity):
        note: str
        enabled: bool

    @entity()
    class Bundle(Entity):
        noted: Noted

    bundle = Bundle(name="bundle", noted=Noted(name="noted", note="a", enabled=True))
    assert [item.path for item in list_entity_values(bundle)] == ["noted.note", "noted.enabled"]


def test_editing_a_settings_field_updates_the_model_settings(tmp_path: Path):
    from orion.ui.app import AppState

    settings = set_entity_value(ui_settings(cache_path=tmp_path), "validate_simulation_states", True)
    assert settings.validate_simulation_states is True
    built = build_model(AppState(settings=settings), default_configuration())
    assert built.settings.validate_simulation_states is True
    assert built.settings.cache_path == tmp_path
    assert built.settings.validate_inputs is False
