"""Shared editor for numeric quantities on an input.

States are read-only. Configuration and the process lab both use this editor.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from nicegui import ui

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
        box = _entity_value_box(item)
        if item.description:
            box.tooltip(item.description)

        def commit(_=None, path: str = item.path, kind: str = item.kind, current: object = item.value) -> None:
            next_value = _read_entity_value(kind, box.value)
            if next_value is None or next_value == current:
                return
            on_change(path, next_value)

        if item.kind == "bool":
            box.on_value_change(commit)
        else:
            box.on("blur", commit)
            box.on("keydown.enter", commit)


def _entity_value_box(item: EntityValue):
    if item.kind == "bool":
        return ui.switch(value=bool(item.value)).props("dense")
    if item.kind == "int":
        return ui.number(value=float(item.value), format="%.0f", step=1).props("dense")  # type: ignore[arg-type]
    if item.kind == "float":
        return ui.number(value=float(item.value), format="%.4g").props("dense")  # type: ignore[arg-type]
    if item.kind == "date":
        shown = item.value.isoformat() if isinstance(item.value, date) else str(item.value)
        return ui.input(value=shown).props("dense type=date")
    return ui.input(value=str(item.value)).props("dense")


def _read_entity_value(kind: str, raw: object) -> object | None:
    if kind == "bool":
        return bool(raw)
    if kind in {"int", "float"}:
        if raw is None:
            return None
        return int(raw) if kind == "int" else float(raw)  # type: ignore[arg-type]
    if kind == "date":
        if not raw:
            return None
        return date.fromisoformat(str(raw))
    return "" if raw is None else str(raw)
