"""Dated management events on a wheat experiment."""

from __future__ import annotations

from datetime import date

from orion.core.constant import Constant, const
from orion.core.entity import Entity, entity
from orion.core.quantity import is_non_negative, is_scalar
from orion.core.resource import Resource


def _on(when: date) -> Constant[date]:
    return const("on", "isodate", when, description="Calendar date")


def _mass(name: str, kilograms_per_m2: float, description: str, resource: Resource) -> Constant[float]:
    return const(name, "kg/m^2", kilograms_per_m2, description=description, constraint=is_scalar + is_non_negative, resource=resource)


@entity()
class SowingEvent(Entity):
    """Sowing of a crop on one date."""

    on: Constant[date]
    """Calendar date of sowing."""

    density: Constant[float] | None = None
    """Seeding density in seeds/m²."""

    depth: Constant[float] | None = None
    """Sowing depth in m."""


@entity()
class FertiliserEvent(Entity):
    """Fertiliser application on one date."""

    on: Constant[date]
    """Calendar date of application."""

    n: Constant[float]
    """Nitrogen applied in kg/m²."""

    p: Constant[float] = _mass("p", 0.0, "Phosphorus applied", "phosphorus")
    """Phosphorus applied in kg/m²."""

    k: Constant[float] = _mass("k", 0.0, "Potassium applied", "potassium")
    """Potassium applied in kg/m²."""

    product: str = ""
    """Fertiliser product name when published."""


@entity()
class ProtectionEvent(Entity):
    """Crop-protection application, or an explicit standard/none record."""

    on: Constant[date]
    """Calendar date of the application or of the management statement."""

    kind: str
    """herbicide, fungicide, insecticide, pgr, standard, or none."""

    product: str = ""
    """Product name, or excluded class when kind is none."""

    active_ingredient: str = ""
    """Active ingredient when published."""

    rate: float | None = None
    """Application rate in ``rate_unit`` as published (l/ha, kg/ha)."""

    rate_unit: str = ""
    """Rate unit as published (l/ha, kg/ha)."""


@entity()
class IrrigationEvent(Entity):
    """Irrigation on one date."""

    on: Constant[date]
    """Calendar date of irrigation."""

    amount: Constant[float]
    """Irrigation amount in kg/m² (1 mm water ≡ 1 kg/m²)."""


ManagementEvent = SowingEvent | FertiliserEvent | ProtectionEvent | IrrigationEvent


def sowing_event(name: str, on: date, density: float | None = None, depth: float | None = None) -> SowingEvent:
    """Sowing event with the date and rates stored as constants."""
    return SowingEvent(
        name,
        on=_on(on),
        density=None if density is None else const("density", "seeds/m^2", density, description="Seeding density", constraint=is_scalar + is_non_negative),
        depth=None if depth is None else const("depth", "m", depth, description="Sowing depth", constraint=is_scalar + is_non_negative),
    )


def fertiliser_event(name: str, on: date, n: float, p: float = 0.0, k: float = 0.0, product: str = "") -> FertiliserEvent:
    """Fertiliser event with nutrient rates in kg/m²."""
    return FertiliserEvent(
        name,
        on=_on(on),
        n=_mass("n", n, "Nitrogen applied", "nitrogen"),
        p=_mass("p", p, "Phosphorus applied", "phosphorus"),
        k=_mass("k", k, "Potassium applied", "potassium"),
        product=product,
    )


def protection_event(name: str, on: date, kind: str, product: str = "", active_ingredient: str = "", rate: float | None = None, rate_unit: str = "") -> ProtectionEvent:
    """Protection event. Published spray rates stay in their printed unit."""
    return ProtectionEvent(name, on=_on(on), kind=kind, product=product, active_ingredient=active_ingredient, rate=rate, rate_unit=rate_unit)


def irrigation_event(name: str, on: date, amount: float) -> IrrigationEvent:
    """Irrigation event with the amount in kg/m²."""
    return IrrigationEvent(name, on=_on(on), amount=_mass("amount", amount, "Irrigation amount", "water"))
