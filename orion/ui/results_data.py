"""Extract display-friendly scalar/const rows and plot keys from models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import jax.numpy as jnp

from orion.core.entity import Entity
from orion.core.model import Model
from orion.core.quantity import Quantity
from orion.core.state import State
from orion.ui.reflect import format_path, on_within_step, owned_quantities
from orion.ui.units import convert_for_display, display_class_for, display_unit_for


@dataclass(frozen=True)
class ScalarDatum:
    """One constant or scalar quantity for table display."""

    name: str
    unit: str
    value: str
    description: str = ""
    resource: str | None = None


def _format_value(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    try:
        arr = jnp.asarray(value)
    except Exception:  # noqa: BLE001
        return None
    if arr.shape != () and arr.size != 1:
        return None
    item = arr.reshape(()).item()
    if isinstance(item, (bool,)):
        return str(item)
    if isinstance(item, int):
        return str(item)
    if isinstance(item, float):
        return f"{item:.6g}"
    return str(item)


def apply_unit_alternatives(rows: Sequence[ScalarDatum], preferences: Mapping[str, str]) -> tuple[ScalarDatum, ...]:
    """Rewrite scalar rows into the user's chosen display units."""
    converted: list[ScalarDatum] = []
    for row in rows:
        if not row.unit:
            converted.append(row)
            continue
        try:
            numeric = float(row.value)
        except ValueError:
            converted.append(row)
            continue
        display = display_unit_for(display_class_for(row.unit, row.resource), preferences)
        new_value, new_unit = convert_for_display(numeric, row.unit, display)
        if new_unit == row.unit and new_value == numeric:
            converted.append(row)
            continue
        converted.append(replace(row, unit=new_unit, value=_format_value(new_value) or row.value))
    return tuple(converted)


def _walk_scalars(prefix: str, entity: Entity) -> list[ScalarDatum]:
    rows: list[ScalarDatum] = []
    for path, quantity in owned_quantities(entity, Quantity):
        if on_within_step(quantity):
            continue
        formatted = _format_value(quantity.value)
        if formatted is None:
            continue
        label = format_path(path)
        name = f"{prefix}.{label}" if prefix else label
        rows.append(
            ScalarDatum(
                name=name,
                unit=quantity.unit or "",
                value=formatted,
                description=quantity.description or "",
                resource=quantity.resource,
            )
        )
    return rows


def collect_scalar_data(model: Model) -> tuple[ScalarDatum, ...]:
    """Constants and scalar variables from all model states (nested included)."""
    rows: list[ScalarDatum] = []
    for state in model.states:
        rows.extend(_walk_scalars(state.name, state))
    return tuple(rows)


def collect_scalar_data_for_state(state: State) -> tuple[ScalarDatum, ...]:
    return tuple(_walk_scalars(state.name, state))


def plot_key(state_name: str, variable_name: str) -> str:
    return f"{state_name}.{variable_name}"


def list_plot_keys(history: Model) -> list[str]:
    from orion.ui.introspect import list_plottable_variables

    return [plot_key(s, v) for s, v, _ in list_plottable_variables(history)]
