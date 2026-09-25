"""Shared editor for numeric quantities on an input.

States are read-only. Configuration and the process lab both use this editor.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from orion.core.entity import Entity
from orion.ui.reflect import InputField, input_field_label, list_input_fields
from orion.ui.theme import setting_row


def render_entity_editor(
    entity: Entity,
    on_change: Callable[[str, float], None],
    *,
    live: bool = False,
    after_field: Callable[[InputField], None] | None = None,
) -> None:
    """Edit every numeric quantity on an input, constants included."""
    for field in list_input_fields(entity):
        _render_quantity_field(field, on_change, live=live)
        if after_field is not None:
            after_field(field)


def _render_quantity_field(field: InputField, on_change: Callable[[str, float], None], *, live: bool) -> None:
    with setting_row(input_field_label(field.name)):
        box = ui.number(value=field.value, format="%.4g").props("dense")
        hint = field.description
        if field.unit:
            hint = f"{hint} ({field.unit})" if hint else field.unit
        if hint:
            box.tooltip(hint)

        def commit(_=None, name: str = field.name, current: float = field.value) -> None:
            raw = box.value
            if raw is None:
                return
            next_value = float(raw)
            if next_value != current:
                on_change(name, next_value)

        if live:
            box.on_value_change(commit)
        else:
            box.on("blur", commit)
            box.on("keydown.enter", commit)
