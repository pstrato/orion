"""Shared editor for numeric quantities on an input.

States are read-only. Configuration and the process lab both use this editor.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from datetime import date

from orion.core.entity import Entity
from orion.ui.reflect import EntityValue, InputField, input_field_label, list_entity_values, list_input_fields
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


def render_entity_values(entity: Entity, on_change: Callable[[str, object], None]) -> None:
    """Edit scalar fields discovered on an entity and its nested entities."""
    for item in list_entity_values(entity):
        _render_entity_value(item, on_change)


def _render_entity_value(item: EntityValue, on_change: Callable[[str, object], None]) -> None:
    with setting_row(input_field_label(item.path), wide=item.kind == "path"):
        if item.kind == "bool":
            box = ui.switch(value=bool(item.value)).props("dense")
        elif item.kind == "int":
            box = ui.number(value=float(item.value), format="%.0f", step=1).props("dense")  # type: ignore[arg-type]
        elif item.kind == "float":
            box = ui.number(value=float(item.value), format="%.4g").props("dense")  # type: ignore[arg-type]
        elif item.kind == "date":
            shown = item.value.isoformat() if isinstance(item.value, date) else str(item.value)
            box = ui.input(value=shown).props("dense type=date")
        else:
            box = ui.input(value=str(item.value)).props("dense")
        if item.description:
            box.tooltip(item.description)

        def commit(_=None, path: str = item.path, kind: str = item.kind, current: object = item.value) -> None:
            raw = box.value
            if kind == "bool":
                next_value: object = bool(raw)
            elif kind == "int":
                if raw is None:
                    return
                next_value = int(raw)
            elif kind == "float":
                if raw is None:
                    return
                next_value = float(raw)
            elif kind == "date":
                if not raw:
                    return
                next_value = date.fromisoformat(str(raw))
            else:
                next_value = "" if raw is None else str(raw)
            if next_value != current:
                on_change(path, next_value)

        if item.kind == "bool":
            box.on_value_change(commit)
        else:
            box.on("blur", commit)
            box.on("keydown.enter", commit)
