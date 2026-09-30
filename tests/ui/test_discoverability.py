"""Behavioural tests for UI discovery of model structure and history."""

from __future__ import annotations

import pytest

from orion.ui.introspect import history_series, inspect_model, list_plottable_variables
from orion.ui.reflect import apply_quantity_edit


def test_model_view_exposes_run_context_from_input(clock_model):
    view = inspect_model(clock_model.model, clock_model.inputs, clock_model.processes)

    assert view.name == "clock only"
    assert view.days == 10
    assert view.step_hours == 3
    assert view.location_name == "parcel"
    assert view.latitude == pytest.approx(51.5)
    assert view.longitude == pytest.approx(0.1)
    assert view.device


def test_model_view_discovers_processes_and_states_by_name(clock_model):
    view = inspect_model(clock_model.model, clock_model.inputs, clock_model.processes)

    assert "clock" in [p.name for p in view.processes]
    assert "clock" in [s.name for s in view.states]


def test_state_view_exposes_variables_with_units(clock_model):
    view = inspect_model(clock_model.model, clock_model.inputs, clock_model.processes)
    clock_state = next(s for s in view.states if s.name == "clock")
    variables = {v.name: v for v in clock_state.fields if v.kind == "variable"}

    assert "step" in variables
    assert variables["step"].unit == "step"
    assert variables["step"].kind == "variable"


def test_inputs_are_editable_including_constants(clock_model):
    from orion.processes.clock import ClockInput

    clock_input = next(entity for _path, entity in clock_model.inputs.all_entities() if isinstance(entity, ClockInput))
    edited = apply_quantity_edit(clock_input, "delta", 6)
    state = edited.states()

    assert int(edited.delta.value) == 6
    assert int(clock_input.delta.value) == 3
    assert int(state.delta.value) == 6


def test_states_are_readonly(clock_model):
    clock = next(state for state in clock_model.states if state.name == "clock")

    with pytest.raises(TypeError, match="read-only"):
        apply_quantity_edit(clock, "delta", 6)
    with pytest.raises(TypeError, match="read-only"):
        apply_quantity_edit(clock, "step", 9)
    assert int(clock.delta.value) == 3


def test_simulation_history_lists_plottable_variables(clock_model):
    _, history = clock_model.run(5)

    assert ("clock", "step", "step") in list(list_plottable_variables(history))


def test_simulation_history_exposes_the_steps_on_its_step_axis(clock_model):
    final, history = clock_model.run(5)
    assert final.axes == ()
    assert [item.name for item in history.axes] == ["step"]
    assert history.axes[0].values == (1, 2, 3, 4, 5)


def test_simulation_history_provides_monotonic_clock_series(clock_model):
    steps = 5
    _, history = clock_model.run(steps)

    series, unit, _dimension = history_series(history, "clock", "step")

    assert unit == "step"
    assert len(series) == steps
    assert series == sorted(series)
    assert series[0] == 1.0
    assert series[-1] == float(steps)


def test_missing_history_variable_is_an_error(clock_model):
    _, history = clock_model.run(2)

    with pytest.raises(KeyError):
        history_series(history, "clock", "not_a_variable")
