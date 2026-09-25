"""Behaviour: Configurations tab lists variants and edits weather, soil, and crop."""

from __future__ import annotations

import inspect

from orion.core.input import Input
from orion.ui.configuration import Configuration, default_configuration, next_configuration_color, provider_fields
from orion.ui.configuration_tab import add_configuration, clamp_configuration_index, clamp_input_key, process_input_slots


def test_adding_a_configuration_appends_it_and_selects_the_new_row():
    configs = [default_configuration()]
    configs, index = add_configuration(configs)
    assert len(configs) == 2
    assert index == 1
    assert configs[1].readonly is False
    assert configs[1].clock == "Clock"
    assert configs[1].weather == "Open-Meteo"
    assert configs[1].soil == "SoilGrids"
    assert configs[1].crop == "Wheat"
    assert configs[1].light_interception == "Beer-Lambert"
    assert configs[1].color == next_configuration_color((default_configuration(),))


def test_configuration_index_stays_inside_the_list():
    configs = [default_configuration(), Configuration(name="trial", enabled=True, color="#C45C26", processes=(), readonly=False)]
    assert clamp_configuration_index(configs, 0) == 0
    assert clamp_configuration_index(configs, 1) == 1
    assert clamp_configuration_index(configs, 9) == 1
    assert clamp_configuration_index(configs, -1) == 0


def test_process_input_slots_follow_inputs_and_optional_processes():
    defaults = default_configuration()
    slots = process_input_slots(defaults)
    assert [slot.key for slot in slots] == [name for name, _label in provider_fields()]
    assert [slot.label for slot in slots] == ["Clock", "Weather", "Soil", "Crop", "LightInterception"]
    crop = next(slot for slot in slots if slot.key == "crop")
    assert crop.label == "Crop"
    assert crop.implementation == "Wheat"
    soil = next(slot for slot in slots if slot.key == "soil")
    assert soil.implementation == "SoilGrids"

    class PhenologyInput(Input):
        """Optional process input used only by this test."""

    with_phenology = Configuration(
        name="trial",
        enabled=True,
        color="#C45C26",
        processes=(PhenologyInput("phenology"),),
        readonly=False,
    )
    extra = process_input_slots(with_phenology)
    assert extra[-1].key == "phenology"
    assert extra[-1].label == "Phenology"
    assert clamp_input_key(defaults, "missing") == slots[0].key
    assert clamp_input_key(defaults, "crop") == "crop"


def test_configuration_tab_lists_process_inputs_then_edits_the_selected_one():
    import orion.ui.configuration_tab as tab

    source = inspect.getsource(tab)
    assert "orion-config-list" in source
    assert "process_input_slots" in source
    assert "Implementation" in source
    assert "render_entity_editor" in source
    assert "k_leaves" not in source
    assert 'icon="add"' in source


def test_build_inputs_keeps_clock_delta_editable(tmp_path):
    from orion.processes.clock import ClockInput
    from orion.ui.app import AppState, build_inputs, ui_settings

    built = build_inputs(AppState(settings=ui_settings(cache_path=tmp_path), step_hours=3), default_configuration().with_parameter("clock", "delta", 6))
    clock = next(entity for _path, entity in built.all_entities() if isinstance(entity, ClockInput))
    assert int(clock.delta.value) == 6
