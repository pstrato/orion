"""Behaviour: process discovery edits inputs, including constants, and only reads states."""

from __future__ import annotations

import pytest

from orion.ui.process_explorer import (
    ParamSpec,
    ProcessInfo,
    discover_process_classes,
    editable_fields_for,
    get_process,
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
