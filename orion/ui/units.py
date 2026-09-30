"""Display-unit choices. Simulation values stay in canonical SI units."""

from __future__ import annotations

from collections.abc import Mapping

UNIT_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "kg/kg": ("kg/kg", "%", "g/kg"),
    "kg/m^2": ("kg/m^2", "t/ha", "kg/ha"),
    "water": ("mm", "kg/m^2", "kg/ha"),
    "kg/m^3": ("kg/m^3", "g/cm^3"),
    "m": ("m", "cm", "mm"),
}

DEFAULT_UNIT_ALTERNATIVES: dict[str, str] = {
    "kg/kg": "%",
    "kg/m^2": "kg/ha",
    "water": "mm",
    "kg/m^3": "kg/m^3",
    "m": "m",
}

_TO_DISPLAY: dict[tuple[str, str], float] = {
    ("kg/kg", "%"): 100.0,
    ("kg/kg", "g/kg"): 1000.0,
    ("kg/m^2", "t/ha"): 10.0,
    ("kg/m^2", "kg/ha"): 10000.0,
    ("kg/m^2", "mm"): 1.0,
    ("kg/m^3", "g/cm^3"): 0.001,
    ("m", "cm"): 100.0,
    ("m", "mm"): 1000.0,
}


def display_class_for(unit: str, dimension: str | None = None) -> str:
    """Display class for a quantity. Water stored as kg/m² uses the water setting."""
    if dimension == "water" and unit == "kg/m^2":
        return "water"
    return unit


def display_unit_for(canonical: str, preferences: Mapping[str, str]) -> str:
    choices = UNIT_ALTERNATIVES.get(canonical)
    default = DEFAULT_UNIT_ALTERNATIVES.get(canonical, canonical)
    if not choices:
        return canonical
    chosen = preferences.get(canonical, default)
    if chosen not in choices:
        return default
    return chosen


def resolve_unit_alternatives(preferences: Mapping[str, str]) -> dict[str, str]:
    """Keep overrides that differ from the agronomic default."""
    resolved: dict[str, str] = {}
    for key, value in preferences.items():
        choices = UNIT_ALTERNATIVES.get(key)
        if choices is None or value not in choices:
            continue
        if value == DEFAULT_UNIT_ALTERNATIVES.get(key):
            continue
        resolved[key] = value
    return resolved


def convert_for_display(value: float, si_unit: str, display: str) -> tuple[float, str]:
    if display == si_unit:
        return value, si_unit
    factor = _TO_DISPLAY.get((si_unit, display))
    if factor is None:
        return value, si_unit
    return value * factor, display
