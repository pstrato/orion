"""Settings helpers for choosing display unit alternatives."""

from __future__ import annotations

from orion.ui.units import UNIT_ALTERNATIVES

_LABELS: dict[str, str] = {
    "kg/kg": "Concentrations (mass fraction)",
    "kg/m^2": "Mass per area (biomass, nutrients)",
    "water": "Water (precipitation, soil water, irrigation)",
    "kg/m^3": "Mass density",
    "m": "Depth / length",
}


def unit_alternative_options() -> list[dict[str, object]]:
    """One Settings-row descriptor per canonical unit that has alternatives."""
    rows: list[dict[str, object]] = []
    for canonical, alternatives in UNIT_ALTERNATIVES.items():
        rows.append(
            {
                "canonical": canonical,
                "label": _LABELS.get(canonical, canonical),
                "choices": {unit: unit for unit in alternatives},
            }
        )
    return rows
