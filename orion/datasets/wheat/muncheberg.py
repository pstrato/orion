"""Müncheberg ZALF wheat agro-ecosystem experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from orion.core.input import LocationInput
from orion.datasets.convert import to_kg_per_kg, to_kg_per_m2
from orion.datasets.events import ManagementEvent, ProtectionEvent, fertiliser_event, irrigation_event, protection_event, sowing_event
from orion.datasets.io import as_mapping, cell, event_name, group_rows, parse_date, parse_float, point_location, read_rows
from orion.datasets.observations import Observation, make_observation
from orion.datasets.protection import protection_kind
from orion.datasets.site_year import SiteYear

MUNCHEBERG_LOCATION = point_location("muncheberg", 14.12, 52.52)
SOURCE = "muncheberg"


def parse_muncheberg(directory: Path) -> tuple[SiteYear, ...]:
    """Parse Müncheberg management and crop observation tables."""
    plots = as_mapping(read_rows(directory / "plots.csv"), "plot")
    management = group_rows(read_rows(directory / "management.csv"), "plot")
    crop_rows = group_rows(read_rows(directory / "crop.csv"), "plot")

    years: list[SiteYear] = []
    for (plot_id,), action_rows in management.items():
        plot = plots.get(plot_id, {})
        location = _location(plot)
        cultivar = cell(plot, "cultivar") or cell(action_rows[0], "product")
        treatment = cell(plot, "intensity") or cell(action_rows[0], "intensity") or plot_id
        events: list[ManagementEvent] = []
        sowing_on: date | None = None
        for row in action_rows:
            action = cell(row, "action").lower()
            on = parse_date(cell(row, "date"))
            product = cell(row, "product")
            if action == "sowing":
                sowing_on = on
                events.append(sowing_event(event_name("sowing", on), on))
                if product:
                    cultivar = product
            elif action in {"fertiliser", "fertilizer", "fertilisation", "fertilization"}:
                events.append(
                    fertiliser_event(
                        event_name("fertiliser", on),
                        on,
                        n=to_kg_per_m2(parse_float(cell(row, "n_kg_ha"), 0.0) or 0.0, "kg/ha"),
                        p=to_kg_per_m2(parse_float(cell(row, "p_kg_ha"), 0.0) or 0.0, "kg/ha"),
                        k=to_kg_per_m2(parse_float(cell(row, "k_kg_ha"), 0.0) or 0.0, "kg/ha"),
                        product=product,
                    )
                )
            elif action in {"protection", "pesticide"}:
                events.append(
                    protection_event(
                        event_name("protection", on),
                        on,
                        kind=protection_kind(product=product, measure=action),
                        product=product,
                    )
                )
            elif action == "irrigation":
                amount = parse_float(cell(row, "irrigation_mm"), 0.0) or 0.0
                events.append(irrigation_event(event_name("irrigation", on), on, to_kg_per_m2(amount, "mm")))
        if not any(isinstance(event, ProtectionEvent) for event in events):
            on = sowing_on or parse_date(cell(action_rows[0], "date"))
            kind = "none" if treatment.lower() == "organic" else "standard"
            events.append(protection_event(event_name("protection", on), on, kind=kind))
        season = str((sowing_on or parse_date(cell(action_rows[0], "date"))).year)
        observations = _crop_observations(crop_rows.get((plot_id,), []))
        years.append(
            SiteYear(
                name=f"{SOURCE}:{season}:{treatment}",
                crop="wheat",
                cultivar=cultivar,
                season=season,
                treatment=treatment,
                source=SOURCE,
                location=location,
                events=tuple(events),
                observations=observations,
            )
        )
    return tuple(years)


def _location(plot: dict[str, str]) -> LocationInput:
    lat = parse_float(cell(plot, "latitude"))
    lon = parse_float(cell(plot, "longitude"))
    if lat is None or lon is None:
        return MUNCHEBERG_LOCATION
    return point_location("muncheberg", lon, lat)


def _crop_observations(rows: list[dict[str, str]]) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for row in rows:
        on = parse_date(cell(row, "date"))
        bbch = parse_float(cell(row, "bbch"))
        if bbch is not None:
            observations.append(make_observation(f"phenology-{on.isoformat()}", on, "phenology_stage", "step", bbch))
        biomass = parse_float(cell(row, "aboveground_biomass_dt_ha"))
        if biomass is not None:
            observations.append(make_observation(f"biomass-{on.isoformat()}", on, "biomass", "kg/m^2", to_kg_per_m2(biomass, "dt/ha")))
        grain = parse_float(cell(row, "yield_dt_ha"))
        if grain is not None:
            observations.append(make_observation("yield", on, "yield", "kg/m^2", to_kg_per_m2(grain, "dt/ha")))
        protein = parse_float(cell(row, "grain_protein_percent"))
        if protein is not None:
            observations.append(make_observation("grain_protein", on, "grain_protein", "kg/kg", to_kg_per_kg(protein, "%")))
        grain_n = parse_float(cell(row, "grain_n_percent"))
        if grain_n is not None:
            observations.append(make_observation("grain_n", on, "grain_n", "kg/kg", to_kg_per_kg(grain_n, "%")))
    return tuple(observations)
