"""Canonical SI-oriented unit strings for quantities.

Orion quantity units for the main physical classes:

- surface / area: ``m^2`` (where used)
- mass per area (biomass, nutrients, water): ``kg/m^2``
- solar radiation: ``W/m^2``
- depth / length: ``m``
- mass density (bulk density, root density): ``kg/m^3``
- area density (canopy leaf/stem/ear density): ``m^2/m^3``
- dimensionless coefficients (extinction k): ``1``
- concentrations (mass fractions): ``kg/kg`` (kg thing / kg medium)

Clients and datasets convert published units into these before building states.
"""

from typing import Literal

Unit = Literal[
    "m",
    "m^2",
    "m^2/m^2",
    "kg/m^2",
    "kg/m^3",
    "m^2/m^3",
    "1",
    "kg/kg",
    "cmolc/kg",
    "pH",
    "°C",
    "W/m^2",
    "step",
    "s",
    "isodate",
    "dimensionless",
    "coordinate",
    "seeds/m^2",
    "hours",
    "days",
]
"""Allowed quantity units."""
