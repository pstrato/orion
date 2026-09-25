"""Named axes for quantity arrays and trajectory metadata."""

from __future__ import annotations

from orion.core.entity import Entity, entity


@entity()
class Axis(Entity):
    """One ndarray axis; ``name`` is the id (batch, step, layer, …)."""

    description: str
    """Human-readable meaning of this axis."""

    labels: tuple[str, ...] | None = None
    """Optional categorical coordinates along this axis."""


def axis(
    name: str,
    *,
    description: str = "",
    labels: tuple[str, ...] | None = None,
) -> Axis:
    """Create an axis."""
    return Axis(name, description, labels)


BATCH = axis("batch", description="Parallel simulation members (locations, scenarios, …)")
"""Leading axis for batched runs."""

STEP = axis("step", description="Simulation step")
"""Trajectory axis for scanned simulation history."""

LAYER = axis("layer", description="Soil / root layer index")
"""Depth or root-density layer axis."""

CELL = axis("cell", description="Geographic grid cell index")

WITHIN_STEP = axis("within_step", description="Sub-step samples within one clock step")
"""Intra-step weather / flux samples (e.g. hourly within a multi-hour step)."""
