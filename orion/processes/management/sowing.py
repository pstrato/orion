"""Abstract sowing process and outcome state."""

from __future__ import annotations

from orion.core.jax import entity

from orion.core.constant import Constant
from orion.core.entity import Entity
from orion.core.input import Input, Inputs
from orion.core.process import Process
from orion.core.quantity import is_non_negative, is_scalar
from orion.core.state import State, state_meta_fields
from orion.core.variable import var
from orion.processes.clock import Clock


@entity(data=("amount",), meta=state_meta_fields)
class Sowing(State):
    """Sowing information for one step."""

    density: Constant[float]
    """Sowing density (seeds/m²)."""

    variety: Constant[str]
    """Sowing variety."""

    species: Constant[str]
    """Sowing species."""


@entity()
class SowingEvent(Entity):
    """Sowing event information."""

    density: Constant[float]
    """Sowing density (seeds/m²)."""

    variety: Constant[str]
    """Sowing variety."""

    species: Constant[str]
    """Sowing species."""


@entity()
class SowingProcess(Entity, Process):
    """Sowing process."""

    events: Constant[dict[int, SowingEvent]]
    """Sowing events to perform per hour after start."""

    @property
    def entity_kind(self) -> str:
        return "process"

    def step(self, clock: Clock) -> Sowing:
        """Compute sowing outcome for this step."""
        event = self.events.value.get(clock.has.value)
        if event is None:
            return Sowing(
                name="sowing",
                constraint=None,
                amount=var(
                    "amount",
                    "seeds/m^2",
                    0.0,
                    description="Seed amount sown",
                    constraint=is_scalar + is_non_negative,
                ),
            )
        else:
            return Sowing(
                name="sowing",
                constraint=None,
                amount=var(
                    "amount",
                    "seeds/m^2",
                    event.density.value,
                    description="Seed amount sown",
                    constraint=is_scalar + is_non_negative,
                ),
            )


@entity()
class SowingInput(Entity, Input):
    """Seeds sowing outcome and process."""

    def states(self, inputs: Inputs) -> tuple[State, ...]:
        return (
            Sowing(
                name="sowing",
                constraint=None,
                amount=var(
                    "amount",
                    "seeds/m^2",
                    description="Seed amount sown",
                    constraint=is_scalar + is_non_negative,
                ),
            ),
        )

    def processes(self, inputs: Inputs) -> tuple[Process, ...]:
        return (SowingProcess(self.name),)
