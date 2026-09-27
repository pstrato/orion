"""Named axes for quantity arrays and trajectory metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from orion.core.entity import Entity, EntityRelation, entity


@entity()
class Axis(Entity):
    """One ndarray axis; ``name`` is the id (inputs, step, layer, …)."""

    description: str
    """Human-readable meaning of this axis."""

    labels: tuple[str, ...] | None = None
    """Optional categorical coordinates along this axis."""

    values: tuple[Any, ...] = ()
    """Coordinates along this axis: the inputs, or the step numbers."""

    def direct_entities(self) -> Iterable[tuple[EntityRelation, Entity]]:
        """Coordinates stay on the axis; they are not child entities."""
        return iter(())


def axis(
    name: str,
    *,
    description: str = "",
    labels: tuple[str, ...] | None = None,
    values: tuple[Any, ...] = (),
) -> Axis:
    """Create an axis."""
    return Axis(name, description, labels, values)


INPUTS = axis("inputs", description="One member per inputs in a batched simulation")
"""Leading axis when several inputs are simulated together."""

STEP = axis("step", description="Simulation step")
"""Trajectory axis for scanned simulation history."""


def inputs_axis(members: tuple[Any, ...]) -> Axis:
    """Inputs axis whose values are the inputs, in ndarray order."""
    return axis("inputs", description=INPUTS.description, values=members)


def step_axis(count: int) -> Axis:
    """Step axis whose values are the completed steps, 1 through ``count``."""
    return axis("step", description=STEP.description, values=tuple(range(1, count + 1)))


LAYER = axis("layer", description="Soil / root layer index")
"""Depth or root-density layer axis."""

CELL = axis("cell", description="Geographic grid cell index")

WITHIN_STEP = axis("within_step", description="Sub-step samples within one clock step")
"""Intra-step weather / flux samples (e.g. hourly within a multi-hour step)."""
