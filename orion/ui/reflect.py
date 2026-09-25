"""Walk Orion entities for the UI.

Discovery uses ``direct_entities`` and ``all_entities``. States are read-only.
Inputs are editable, including their ``Constant`` values.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from typing import TypeVar, cast

import jax.numpy as jnp

from orion.core.axis import Axis
from orion.core.constant import Constant
from orion.core.constraint import Constraint
from orion.core.entity import Entity, EntityPath, EntityRelation
from orion.core.model import Model
from orion.core.parameter import Parameter
from orion.core.quantity import Quantity
from orion.core.state import State
from orion.core.variable import Variable

_SKIP_RELATIONS = (Constraint, Axis)


@dataclass(frozen=True)
class InputField:
    """One numeric quantity the configuration editor can change."""

    name: str
    value: float
    description: str
    unit: str | None


def quantity_kind(entity: Entity) -> str:
    """Role of a quantity as the UI should present it."""
    if isinstance(entity, Variable):
        return "variable"
    if isinstance(entity, Parameter):
        return "parameter"
    if isinstance(entity, Constant):
        return "constant"
    if isinstance(entity, Quantity):
        return "quantity"
    return "entity"


def format_relation(relation: EntityRelation) -> str:
    if relation.index is None:
        return relation.name
    return f"{relation.name}.{relation.index}"


def format_path(path: EntityPath) -> str:
    return ".".join(format_relation(relation) for relation in path)


T = TypeVar("T", bound=Entity)


def owned_quantities(entity: Entity, kind: type | tuple[type, ...] = Quantity) -> Iterable[tuple[EntityPath, Quantity]]:
    """Quantities owned by ``entity``, not quantities that belong to a nested state."""

    def walk(node: Entity, prefix: EntityPath) -> Iterable[tuple[EntityPath, Quantity]]:
        for relation, child in node.direct_entities():
            if isinstance(child, _SKIP_RELATIONS):
                continue
            path = (*prefix, relation)
            if isinstance(child, State):
                continue
            if isinstance(child, Quantity) and isinstance(child, kind):
                yield path, child
                continue
            if isinstance(child, Entity):
                yield from walk(child, path)

    return walk(entity, ())


def entity_at(root: Entity, path: EntityPath) -> Entity:
    current: Entity = root
    for relation in path:
        value = getattr(current, relation.name)
        current = value if relation.index is None else value[relation.index]
    return current


def replace_entity(root: Entity, path: EntityPath, updated: Entity) -> Entity:
    """Return a copy of ``root`` with the entity at ``path`` replaced."""
    if not path:
        return updated
    relation, *rest = path
    value = getattr(root, relation.name)
    if relation.index is None:
        child = replace_entity(value, tuple(rest), updated)
        return replace(root, **{relation.name: child})
    items = list(value)
    items[relation.index] = replace_entity(items[relation.index], tuple(rest), updated)
    return replace(root, **{relation.name: tuple(items)})


def set_quantity_value(quantity: Quantity, value: float) -> Quantity:
    """Return a copy of any quantity, including constants, with a new value."""
    if isinstance(quantity, (Variable, Parameter)):
        return quantity.set(jnp.asarray(value))
    current = quantity.value
    if isinstance(current, bool):
        return replace(quantity, value=bool(value))
    if isinstance(current, int):
        return replace(quantity, value=int(value))
    if isinstance(current, float):
        return replace(quantity, value=float(value))
    if hasattr(current, "shape"):
        return replace(quantity, value=jnp.asarray(value))
    return replace(quantity, value=value)


def apply_quantity_edits(root: T, edits: tuple[tuple[str, str, float], ...], role: str) -> T:
    """Apply ``(role, quantity, value)`` edits to an input. States are read-only."""
    if isinstance(root, State):
        raise TypeError("States are read-only in the UI. Edit the Input that creates them, including constants.")
    wanted = {name: value for edit_role, name, value in edits if edit_role == role}
    if not wanted:
        return root
    updated: Entity = root
    matches = [(path, quantity) for path, quantity in owned_quantities(root) if _edit_name(path, quantity) in wanted or quantity.name in wanted]
    for path, quantity in matches:
        key = _edit_name(path, quantity)
        new_value = wanted[key] if key in wanted else wanted[quantity.name]
        updated = replace_entity(updated, path, set_quantity_value(quantity, new_value))
    return cast(T, updated)


def apply_quantity_edit(root: T, name: str, value: float) -> T:
    """Replace one named quantity on ``root``."""
    return apply_quantity_edits(root, ((root.name, name, value),), root.name)


def is_numeric_scalar(value: object) -> bool:
    if isinstance(value, (bool, str, date)):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        arr = jnp.asarray(value)
    except TypeError, ValueError:
        return False
    return arr.shape == () or arr.size == 1


def numeric_scalar(value: object) -> float | None:
    if not is_numeric_scalar(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    arr = jnp.asarray(value).reshape(())
    return float(arr)


def list_input_fields(inp: Entity) -> tuple[InputField, ...]:
    """Numeric scalars on an input, constants included. States have none."""
    if isinstance(inp, State):
        raise TypeError("States are read-only in the UI. Edit the Input that creates them, including constants.")
    fields: list[InputField] = []
    for path, quantity in owned_quantities(inp):
        number = numeric_scalar(quantity.value)
        if number is None:
            continue
        fields.append(
            InputField(
                name=_edit_name(path, quantity),
                value=number,
                description=quantity.description,
                unit=quantity.unit or None,
            )
        )
    return tuple(fields)


def input_field_label(name: str) -> str:
    return name.replace("_", " ").replace(".", " ")


def run_steps(model: Model, steps: int) -> tuple[Model, Model]:
    """Step ``model`` and stack variable values into a history model for plotting."""
    if steps < 1:
        return model, model
    current = model
    snapshots: list[Model] = []
    for _ in range(steps):
        current = current.step()
        snapshots.append(current)
    return current, stack_variables(snapshots)


def stack_variables(snapshots: list[Model]) -> Model:
    """Copy the last snapshot, replacing each variable with its series."""
    template = snapshots[-1]
    stacked: Model = template
    for path, entity in template.all_entities(of_type=Variable):
        if not isinstance(entity, Variable):
            continue
        series = []
        for snapshot in snapshots:
            current = entity_at(snapshot, path)
            if not isinstance(current, Variable):
                raise TypeError(f"Expected a variable at {format_path(path)}")
            series.append(jnp.asarray(current.value))
        updated = replace_entity(stacked, path, replace(entity, value=jnp.stack(series)))
        stacked = cast(Model, updated)
    return stacked


def _edit_name(path: EntityPath, quantity: Quantity) -> str:
    label = format_path(path)
    return label or quantity.name
