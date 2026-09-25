"""Behaviour: one editor edits input quantities in configuration and the process lab."""

from __future__ import annotations

import inspect

from orion.processes.clock import ClockInput
from orion.ui.reflect import list_input_fields


def test_entity_editor_is_the_quantity_editor_for_configuration_and_processes():
    import orion.ui.configuration_tab as configuration_tab
    import orion.ui.entity_editor as editor_module
    import orion.ui.process_tab as process_tab

    editor = inspect.getsource(editor_module)
    assert "list_input_fields" in editor
    assert "ui.number" in editor
    assert "render_entity_editor" in inspect.getsource(configuration_tab)
    assert "render_entity_editor" in inspect.getsource(process_tab)


def test_entity_editor_edits_input_constants_including_clock_delta():
    from datetime import date

    from orion.core.constant import const

    clock = ClockInput(
        name="clock",
        start=const("start", "isodate", date(2024, 1, 1), "Simulation start date"),
        end=const("end", "isodate", date(2024, 1, 11), "Simulation end date"),
        delta=const("delta", "hours", 3, "Simulation delta step in hours"),
    )
    names = [field.name for field in list_input_fields(clock)]
    assert "delta" in names
    assert "step" not in names
