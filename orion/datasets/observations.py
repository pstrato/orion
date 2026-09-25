"""Dated observed quantities on a wheat experiment."""

from __future__ import annotations

from datetime import date

from orion.core.constant import Constant, const
from orion.core.entity import Entity, entity
from orion.core.quantity import is_non_negative, is_scalar
from orion.core.units import Unit
from orion.datasets.convert import to_area_index, to_kg_per_kg, to_kg_per_m2


@entity()
class Observation(Entity):
    """One observed value at a calendar date."""

    on: Constant[date]
    """Date of the observation."""

    reading: Constant[float]
    """Observed quantity. ``reading.name`` is the quantity, ``reading.unit`` its unit."""


def make_observation(name: str, on: date, quantity: str, unit: Unit, value: float) -> Observation:
    """Observation whose date and reading are constants."""
    return Observation(
        name,
        on=const("on", "isodate", on, description="Date of the observation"),
        reading=const(quantity, unit, value, description=quantity, constraint=is_scalar + is_non_negative),
    )


_ALIASES = {
    "yield": "yield",
    "grain_yield": "yield",
    "gwad": "yield",
    "biomass": "biomass",
    "aboveground_biomass": "biomass",
    "cwad": "biomass",
    "leaf_biomass": "leaf_biomass",
    "leaf_dm": "leaf_biomass",
    "lwad": "leaf_biomass",
    "stem_biomass": "stem_biomass",
    "stem_dm": "stem_biomass",
    "swad": "stem_biomass",
    "ear_biomass": "ear_biomass",
    "spike_biomass": "ear_biomass",
    "ewad": "ear_biomass",
    "root_biomass": "root_biomass",
    "root_dm": "root_biomass",
    "rwad": "root_biomass",
    "grain_n": "grain_n",
    "grain_protein": "grain_protein",
    "canopy_n": "canopy_n",
    "crop_n": "canopy_n",
    "lai": "lai",
    "leaf_area_index": "lai",
    "laid": "lai",
    "phenology_stage": "phenology_stage",
    "bbch": "phenology_stage",
    "zadoks": "phenology_stage",
}

_MASS = {"yield", "biomass", "leaf_biomass", "stem_biomass", "ear_biomass", "root_biomass"}
_FRACTION = {"grain_n", "grain_protein", "canopy_n"}


def observation_from_published(on: date, variable: str, value: float, unit: str) -> Observation | None:
    """Map a published variable name and unit onto one Orion observation.

    ICASA codes (GWAD, CWAD, LWAD, SWAD, EWAD, RWAD, LAID) are accepted.
    Unknown variables are skipped.
    """
    quantity = _ALIASES.get(variable.strip().lower())
    if quantity is None:
        return None
    published = unit.strip()
    stored: float
    stored_unit: Unit
    if quantity in _MASS:
        stored = to_kg_per_m2(value, published or "kg/m^2")
        stored_unit = "kg/m^2"
    elif quantity in _FRACTION:
        stored = to_kg_per_kg(value, published or "kg/kg")
        stored_unit = "kg/kg"
    elif quantity == "lai":
        stored = to_area_index(value, published or "m^2/m^2")
        stored_unit = "m^2/m^2"
    else:
        stored = value
        stored_unit = "step"
    return make_observation(f"{quantity}-{on.isoformat()}", on, quantity, stored_unit, stored)
