"""Behaviour: one editor edits input quantities in configuration and the process lab."""

from __future__ import annotations

import inspect

from orion.core.entity import entity
from orion.core.input import Input
from orion.core.parameter import Parameter, param
from orion.core.quantity import between_0_1_inc
from orion.processes.clock import ClockInput
from orion.ui.reflect import list_input_fields, slider_limits


@entity()
class SliderProbeInput(Input):
    """Probe input with one share constrained to the unit interval."""

    share: Parameter


def test_entity_editor_is_the_quantity_editor_for_configuration_and_processes():
    import orion.ui.configuration_tab as configuration_tab
    import orion.ui.entity_editor as editor_module
    import orion.ui.process_tab as process_tab

    editor = inspect.getsource(editor_module)
    assert "list_input_fields" in editor
    assert "ui.number" in editor
    assert "render_entity_editor" in inspect.getsource(configuration_tab)
    assert "render_entity_editor" in inspect.getsource(process_tab)


def test_a_quantity_bounded_between_zero_and_one_reports_slider_limits():
    sample = SliderProbeInput("probe", share=param("share", "1", 0.4, "Share of light", constraint=between_0_1_inc))
    field = list_input_fields(sample)[0]

    assert field.name == "share"
    assert field.lower is not None and field.upper is not None
    assert (field.lower, field.upper) == (0.0, 1.0)
    assert slider_limits(field.lower, field.upper, strict=field.strict) == (0.0, 1.0)


def test_bounded_quantities_use_a_slider():
    import orion.ui.entity_editor as editor_module

    source = inspect.getsource(editor_module)
    assert "ui.slider" in source
    assert "field.lower" in source


def test_entity_editor_edits_input_constants_including_clock_delta():
    from datetime import date

    from orion.core.constant import const
    from orion.core.variable import var

    clock = ClockInput(
        name="clock",
        start=var("start", "isodate", date(2024, 1, 1), "Simulation start date"),
        end=var("end", "isodate", date(2024, 1, 11), "Simulation end date"),
        delta=const("delta", "hours", 3, "Simulation delta step in hours"),
    )
    names = [field.name for field in list_input_fields(clock)]
    assert "delta" in names
    assert "step" not in names
