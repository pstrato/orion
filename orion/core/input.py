from __future__ import annotations

from typing import TYPE_CHECKING

from shapely import Point

from orion.core.constant import Constant, const
from orion.core.entity import Entity, entity

if TYPE_CHECKING:
    from orion.core.state import State


class Input(Entity):
    """Protocol for a model input."""

    def states(self, *args, **kwargs) -> None | State | tuple[State, ...]:
        """Return the initial state(s)."""


@entity()
class LocationInput(Input):
    """A geographical location for a simulation."""

    geometry: Constant[Point]
    """Geometry of the location."""

    @property
    def centroid(self) -> Constant[Point]:
        """Centroid of the location."""
        return const("centroid", "coordinate", self.geometry.value.centroid)


@entity()
class Inputs(Entity):
    """Top-level model inputs.

    - location: geographical location (point or polygon → batched members)
    - start / end / step: clock horizon (step size in hours)
    - inputs: all model inputs (weather, soil, crop, light interception, and additional processes)
    """

    inputs: tuple[Input, ...]
    """Model inputs: weather, soil, crop, light interception, and additional processes."""
