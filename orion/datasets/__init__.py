"""Wheat site-year datasets: schema, parsers, and cache catalogue."""

from __future__ import annotations

import importlib
from typing import Any

from orion.datasets.convert import WHEAT_GRAIN_N_TO_PROTEIN, grain_protein_from_nitrogen, to_kg_per_kg, to_kg_per_m2, to_kg_per_m3

__all__ = [
    "DATASETS_CACHE",
    "WHEAT_GRAIN_N_TO_PROTEIN",
    "WHEAT_PARSERS",
    "FertiliserEvent",
    "IrrigationEvent",
    "ManagementEvent",
    "Observation",
    "ProtectionEvent",
    "SiteYear",
    "SowingEvent",
    "cache_wheat_archive",
    "grain_protein_from_nitrogen",
    "list_wheat_site_years",
    "parse_agmip_kassie",
    "parse_braunschweig",
    "parse_broadbalk",
    "parse_management_pack",
    "parse_muncheberg",
    "parse_westerfeld",
    "to_kg_per_kg",
    "to_kg_per_m2",
    "to_kg_per_m3",
]

_LAZY_EXPORTS = {
    "FertiliserEvent": ("orion.datasets.events", "FertiliserEvent"),
    "IrrigationEvent": ("orion.datasets.events", "IrrigationEvent"),
    "ManagementEvent": ("orion.datasets.events", "ManagementEvent"),
    "ProtectionEvent": ("orion.datasets.events", "ProtectionEvent"),
    "SowingEvent": ("orion.datasets.events", "SowingEvent"),
    "Observation": ("orion.datasets.observations", "Observation"),
    "SiteYear": ("orion.datasets.site_year", "SiteYear"),
    "DATASETS_CACHE": ("orion.datasets.wheat", "DATASETS_CACHE"),
    "WHEAT_PARSERS": ("orion.datasets.wheat", "WHEAT_PARSERS"),
    "cache_wheat_archive": ("orion.datasets.wheat", "cache_wheat_archive"),
    "list_wheat_site_years": ("orion.datasets.wheat", "list_wheat_site_years"),
    "parse_agmip_kassie": ("orion.datasets.wheat", "parse_agmip_kassie"),
    "parse_braunschweig": ("orion.datasets.wheat", "parse_braunschweig"),
    "parse_broadbalk": ("orion.datasets.wheat", "parse_broadbalk"),
    "parse_management_pack": ("orion.datasets.wheat", "parse_management_pack"),
    "parse_muncheberg": ("orion.datasets.wheat", "parse_muncheberg"),
    "parse_westerfeld": ("orion.datasets.wheat", "parse_westerfeld"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
