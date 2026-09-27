"""Behaviour: process discovery edits inputs, including constants, and only reads states."""

from __future__ import annotations

import pytest

from orion.ui.process_explorer import (
    ParamSpec,
    ProcessInfo,
    abstract_process_types,
    discover_process_classes,
    editable_fields_for,
    get_process,
    implementations_of,
    process_hierarchy_tree,
    run_process_sweep,
)


def test_discover_process_classes_includes_clock_and_weather():
    processes = discover_process_classes()
    labels = {p.label for p in processes}
    assert "ClockProcess" in labels
    assert "WeatherProcess" in labels
    clock = next(p for p in processes if p.label == "ClockProcess")
    assert clock.implemented is True
    assert any(cls.__name__ == "Clock" for cls in clock.input_states)
    assert any(cls.__name__ == "Clock" for cls in clock.output_states)


def test_playground_lists_light_interception_and_its_beer_lambert_implementation():
    processes = discover_process_classes()
    light = next(item for item in abstract_process_types(processes) if item.label == "LightInterceptionProcess")
    assert light.implemented is False
    assert [item.label for item in implementations_of(light, processes)] == ["BeerLambertLightInterceptionProcess"]


def test_light_interception_experiment_changes_soil_light_with_extinction():
    info = get_process("orion.processes.crop.light_interception.BeerLambertLightInterceptionProcess")
    paths = {field.path for field in editable_fields_for(info)}
    assert "canopy.leaves.k" in paths
    assert "canopy.radiation" in paths

    dim = run_process_sweep(info, [ParamSpec(path="canopy.leaves.k", mode="value", value=0.1)])
    dense = run_process_sweep(info, [ParamSpec(path="canopy.leaves.k", mode="value", value=0.9)])

    assert dense.outputs[0]["light_interception.soil"] < dim.outputs[0]["light_interception.soil"]


def test_playground_lists_abstract_types_and_groups_implementations():
    processes = discover_process_classes()
    types = abstract_process_types(processes)
    labels = {item.label for item in types}
    assert "ClockProcess" in labels
    assert "WeatherProcess" in labels
    clock = next(item for item in types if item.label == "ClockProcess")
    assert [item.label for item in implementations_of(clock, processes)] == ["ClockProcess"]

    abstract = ProcessInfo(
        key="orion.processes.crop.light_interception.LightInterceptionProcess",
        cls=object,
        label="LightInterceptionProcess",
        module="orion.processes.crop.light_interception",
        doc="",
        bases=("Process",),
        implemented=False,
        input_states=(),
        output_states=(),
    )
    concrete = ProcessInfo(
        key="orion.processes.crop.light_interception.BeerLambertLightInterceptionProcess",
        cls=object,
        label="BeerLambertLightInterceptionProcess",
        module="orion.processes.crop.light_interception",
        doc="",
        bases=("LightInterceptionProcess", "Process"),
        implemented=True,
        input_states=(),
        output_states=(),
    )
    family = (abstract, concrete)
    assert [item.label for item in abstract_process_types(family)] == ["LightInterceptionProcess"]
    assert [item.label for item in implementations_of(abstract, family)] == ["BeerLambertLightInterceptionProcess"]


def test_process_tab_selects_an_implementation_of_the_abstract_type():
    import inspect

    import orion.ui.process_tab as tab

    source = inspect.getsource(tab)
    assert "abstract_process_types" in source
    assert "implementations_of" in source
    assert "Implementation" in source


def test_process_hierarchy_groups_clock_by_package():
    tree = process_hierarchy_tree()
    clock_group = next(node for node in tree if node["label"] == "clock")
    child_labels = {child["label"] for child in clock_group.get("children", [])}
    assert "ClockProcess" in child_labels


def test_clock_input_constant_is_editable_and_state_variables_are_not():
    info = get_process("orion.processes.clock.ClockProcess")
    fields = editable_fields_for(info)
    paths = {field.path for field in fields}
    delta = next(field for field in fields if field.path == "clock.delta")

    assert "clock.delta" in paths
    assert "clock.step" not in paths
    assert delta.kind == "constant"
    assert delta.unit == "hours"


def test_sweep_rebuilds_state_from_the_edited_input_constant():
    info = get_process("orion.processes.clock.ClockProcess")
    result = run_process_sweep(
        info,
        [ParamSpec(path="clock.delta", mode="range", start=1.0, stop=4.0, steps=4)],
    )

    assert len(result.outputs) == 4
    assert [row["clock.step"] for row in result.outputs] == [1.0, 1.0, 1.0, 1.0]


def test_sweep_rejects_edits_aimed_at_a_state():
    info = get_process("orion.processes.clock.ClockProcess")

    with pytest.raises(TypeError, match="read-only"):
        run_process_sweep(info, [ParamSpec(path="clock.step", mode="value", value=4.0)])


def test_abstract_process_cannot_be_swept():
    info = ProcessInfo(
        key="abstract.WeatherProcess",
        cls=object,
        label="WeatherProcess",
        module="orion.processes.weather",
        doc="",
        bases=(),
        implemented=False,
        input_states=(),
        output_states=(),
    )
    with pytest.raises(ValueError, match="abstract"):
        run_process_sweep(info, [])
