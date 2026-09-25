"""Convert published agronomic units into Orion SI quantity units."""

from __future__ import annotations

WHEAT_GRAIN_N_TO_PROTEIN = 5.7
"""Jones factor: wheat grain protein (kg/kg) = grain N (kg/kg) × 5.7."""


def _norm_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "").replace("^", "").replace(".", "")


def to_kg_per_m2(value: float, unit: str) -> float:
    """Convert a mass-per-area (or water depth) value to kg/m².

    Water depths in mm are treated as kg/m² (1 mm over 1 m² ≈ 1 kg of water).
    """
    key = _norm_unit(unit)
    if key in {"kg/m2", "kgm-2"}:
        return value
    if key in {"g/m2", "gm-2"}:
        return value / 1000.0
    if key in {"mm", "mm/m2", "mmm-2"}:
        return value
    if key in {"t/ha", "tha-1", "ton/ha", "tonnes/ha"}:
        return value * 0.1
    if key in {"dt/ha", "dtha-1"}:
        return value * 0.01
    if key in {"kg/ha", "kgha-1"}:
        return value / 10000.0
    raise ValueError(f"Cannot convert {unit!r} to kg/m^2.")


def to_kg_per_m3(value: float, unit: str) -> float:
    """Convert a mass density to kg/m³."""
    key = _norm_unit(unit)
    if key in {"kg/m3", "kgm-3"}:
        return value
    if key in {"g/cm3", "gcm-3", "mg/m3", "mgm-3"}:
        return value * 1000.0
    if key in {"kg/dm3", "kgdm-3", "t/m3", "tm-3"}:
        return value * 1000.0
    if key in {"cg/cm3", "cgcm-3"}:
        return value * 10.0
    raise ValueError(f"Cannot convert {unit!r} to kg/m^3.")


def to_kg_per_kg(value: float, unit: str) -> float:
    """Convert a mass concentration / fraction to kg/kg (kg thing / kg medium)."""
    key = _norm_unit(unit)
    if key in {"kg/kg", "kgkg-1", "1", "fraction", "-"}:
        return value
    if key in {"g/kg", "gkg-1"}:
        return value / 1000.0
    if key in {"mg/kg", "mgkg-1", "ppm"}:
        return value / 1_000_000.0
    if key in {"%", "percent", "pct"}:
        return value / 100.0
    raise ValueError(f"Cannot convert {unit!r} to kg/kg.")


def to_area_index(value: float, unit: str) -> float:
    """Convert a leaf-area index to m²/m²."""
    key = _norm_unit(unit)
    if key in {"", "1", "m2/m2", "dimensionless", "ha/ha"}:
        return value
    raise ValueError(f"Cannot convert {unit!r} to m^2/m^2.")


def grain_protein_from_nitrogen(grain_n: float) -> float:
    """Convert wheat grain nitrogen (kg/kg) to grain protein (kg/kg)."""
    return grain_n * WHEAT_GRAIN_N_TO_PROTEIN
