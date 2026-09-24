"""Behaviour: @entity registers dataclass fields as JAX data or meta."""

from __future__ import annotations

from datetime import date

from orion.core.constant import Constant
from orion.core.entity import Entity, entity
from orion.core.input import Input
from orion.core.parameter import Parameter
from orion.core.process import Process
from orion.core.state import State
from orion.core.variable import Variable


def test_fields_are_registered_as_data_or_meta():
    @entity()
    class Sample(Entity):
        weather: Input
        crop: State
        clock: Process
        biomass: Variable
        rate: Parameter
        layers: tuple[State, ...]
        start: Constant[date]
        label: str

    assert Sample.__entity_data_fields__ == ("weather", "crop", "clock", "biomass", "rate", "layers")
    assert Sample.__entity_meta_fields__ == ("name", "start", "label")
