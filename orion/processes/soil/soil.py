"""Soil profile: shared soil state as a collection of layers."""

from __future__ import annotations

from typing import Any

from orion.core.constant import Constant, const
from orion.core.entity import Entity, entity
from orion.core.quantity import is_non_negative, is_positive, is_scalar
from orion.core.state import State
from orion.core.variable import Variable, var


@entity()
class SoilLayer(Entity):
    """One soil layer: dynamic pools plus SoilGrids constants."""

    top: Constant[float]
    bottom: Constant[float]
    clay: Constant[float]
    sand: Constant[float]
    silt: Constant[float]
    bdod: Constant[float]
    phh2o: Constant[float]
    soc: Constant[float]
    cec: Constant[float]
    organic_nitrogen: Constant[float]
    water: Variable
    nitrogen: Variable
    potassium: Variable
    phosphorus: Variable
    temperature: Variable

    @property
    def thickness(self) -> Constant[float]:
        return const("thickness", "m", self.bottom.value - self.top.value)


def make_soil_layer(
    name: str,
    top: float,
    bottom: float,
    *,
    clay=0.0,
    sand=0.0,
    silt=0.0,
    bdod=1300.0,
    phh2o=7.0,
    soc=0.0,
    cec=0.0,
    organic_nitrogen=0.0,
    water: Any = 0.0,
    nitrogen: Any = 0.0,
    potassium: Any = 0.0,
    phosphorus: Any = 0.0,
    temperature: Any = 15.0,
) -> SoilLayer:
    """Construct a soil layer with explicit quantity values (used by Inputs)."""
    return SoilLayer(
        name=name,
        top=const("top", "m", top, description="Layer top depth", constraint=is_scalar + is_non_negative),
        bottom=const("bottom", "m", bottom, description="Layer bottom depth", constraint=is_scalar + is_non_negative),
        clay=const("clay", "kg/kg", clay, description="Clay mass fraction", constraint=is_scalar + is_non_negative),
        sand=const("sand", "kg/kg", sand, description="Sand mass fraction", constraint=is_scalar + is_non_negative),
        silt=const("silt", "kg/kg", silt, description="Silt mass fraction", constraint=is_scalar + is_non_negative),
        bdod=const("bdod", "kg/m^3", bdod, description="Bulk density", constraint=is_scalar + is_positive),
        phh2o=const("phh2o", "pH", phh2o, description="pH in water", constraint=is_scalar + is_non_negative),
        soc=const("soc", "kg/kg", soc, description="Soil organic carbon mass fraction", constraint=is_scalar + is_non_negative),
        cec=const("cec", "cmolc/kg", cec, description="Cation exchange capacity", constraint=is_scalar + is_non_negative),
        organic_nitrogen=const(
            "organic_nitrogen",
            "kg/kg",
            organic_nitrogen,
            description="Organic nitrogen mass fraction",
            constraint=is_scalar + is_non_negative,
            dimension="nitrogen",
        ),
        water=var(
            "water",
            "kg/m^2",
            water,
            description="Soil water in this layer",
            constraint=is_scalar + is_non_negative,
            dimension="water",
        ),
        nitrogen=var(
            "nitrogen",
            "kg/m^2",
            nitrogen,
            description="Mineral nitrogen in this layer",
            constraint=is_scalar + is_non_negative,
            dimension="nitrogen",
        ),
        potassium=var(
            "potassium",
            "kg/m^2",
            potassium,
            description="Potassium in this layer",
            constraint=is_scalar + is_non_negative,
            dimension="potassium",
        ),
        phosphorus=var(
            "phosphorus",
            "kg/m^2",
            phosphorus,
            description="Phosphorus in this layer",
            constraint=is_scalar + is_non_negative,
            dimension="phosphorus",
        ),
        temperature=var(
            "temperature",
            "°C",
            temperature,
            description="Soil temperature in this layer",
            constraint=is_scalar,
            dimension="heat",
        ),
    )


@entity()
class Soil(State):
    """Shared soil profile: ordered layers with water and nutrient pools."""

    layers: tuple[SoilLayer, ...]

    @property
    def depth(self) -> Constant[float]:
        return const("depth", "m", self.layers[-1].bottom.value)
