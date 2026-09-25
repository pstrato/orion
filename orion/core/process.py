"""Process contract: states in → updated state(s) out."""

from __future__ import annotations

from orion.core.entity import Entity
from orion.core.state import State


class Process(Entity):
    """A simulation process."""

    def step(self, *args, **kargs) -> State | tuple[State, ...]:
        """Advance the process by one step.

        Args:
            ...: States required by this process.

        Returns:
            Updated state(s).
        """
        raise NotImplementedError()
