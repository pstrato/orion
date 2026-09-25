"""Westerfeld / BonaRes winter-wheat long-term trial."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from orion.core.input import LocationInput
from orion.datasets.convert import to_kg_per_kg, to_kg_per_m2
from orion.datasets.events import FertiliserEvent, ProtectionEvent, SowingEvent, fertiliser_event, protection_event, sowing_event
from orion.datasets.io import as_mapping, cell, event_name, group_rows, is_wheat, parse_date, parse_float, point_location, read_rows
from orion.datasets.observations import Observation, make_observation
from orion.datasets.protection import protection_kind
from orion.datasets.site_year import SiteYear

WESTERFELD_LOCATION = point_location("westerfeld", 11.702, 51.819)
SOURCE = "westerfeld"


def parse_westerfeld(directory: Path) -> tuple[SiteYear, ...]:
    """Parse Westerfeld sowing, fertiliser, plant-protection, and yield tables."""
    plots = as_mapping(read_rows(directory / "PLOT.csv") if (directory / "PLOT.csv").is_file() else (), "plot_id")
    sowing_groups = group_rows(read_rows(directory / "SOWING.csv"), "plot_id", "experimental_year")
    fertiliser_groups = group_rows(_optional(directory, "FERTILIZATION.csv"), "plot_id", "experimental_year")
    protection_groups = group_rows(_optional(directory, "PLANT_PROTECTION.csv"), "plot_id", "experimental_year")
    yield_groups = group_rows(read_rows(directory / "YIELD.csv"), "plot_id", "experimental_year")

    years: list[SiteYear] = []
    for (plot_id, season), sow_rows in sowing_groups.items():
        wheat_rows = [row for row in sow_rows if is_wheat(cell(row, "crop"))]
        if not wheat_rows:
            continue
        plot = plots.get(plot_id, {})
        location = _location(plot)
        treatment = cell(plot, "treatment") or plot_id
        cultivar = cell(wheat_rows[0], "cultivar")
        events: list[SowingEvent | FertiliserEvent | ProtectionEvent] = []
        for row in wheat_rows:
            on = parse_date(cell(row, "date"))
            density = parse_float(cell(row, "seeding_rate"))
            events.append(sowing_event(event_name("sowing", on), on, density=density))
        for row in fertiliser_groups.get((plot_id, season), []):
            on = parse_date(cell(row, "date"))
            events.append(
                fertiliser_event(
                    event_name("fertiliser", on),
                    on,
                    n=to_kg_per_m2(parse_float(cell(row, "n_kg_ha"), 0.0) or 0.0, "kg/ha"),
                    p=to_kg_per_m2(parse_float(cell(row, "p_kg_ha"), 0.0) or 0.0, "kg/ha"),
                    k=to_kg_per_m2(parse_float(cell(row, "k_kg_ha"), 0.0) or 0.0, "kg/ha"),
                    product=cell(row, "product"),
                )
            )
        protect_rows = protection_groups.get((plot_id, season), [])
        if protect_rows:
            for row in protect_rows:
                on = parse_date(cell(row, "date"))
                rate = parse_float(cell(row, "rate_l_ha"))
                rate_unit = "l/ha" if rate is not None else ""
                if rate is None:
                    rate = parse_float(cell(row, "rate_kg_ha"))
                    rate_unit = "kg/ha" if rate is not None else ""
                events.append(
                    protection_event(
                        event_name("protection", on),
                        on,
                        kind=protection_kind(type_name=cell(row, "type"), product=cell(row, "product")),
                        product=cell(row, "product"),
                        rate=rate,
                        rate_unit=rate_unit,
                    )
                )
        else:
            on = events[0].on.value
            events.append(protection_event(event_name("protection", on), on, kind="none"))
        observations = _yield_observations(yield_groups.get((plot_id, season), []), events[0].on.value, season)
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


def _optional(directory: Path, name: str) -> tuple[dict[str, str], ...]:
    path = directory / name
    if not path.is_file():
        return ()
    return read_rows(path)


def _location(plot: dict[str, str]) -> LocationInput:
    lat = parse_float(cell(plot, "latitude"))
    lon = parse_float(cell(plot, "longitude"))
    if lat is None or lon is None:
        return WESTERFELD_LOCATION
    return point_location("westerfeld", lon, lat)


def _yield_observations(rows: list[dict[str, str]], sowing: date, season: str) -> tuple[Observation, ...]:
    if not rows:
        return ()
    row = rows[0]
    on = _harvest_date(sowing, season)
    harvest = cell(row, "harvest_date")
    if harvest:
        on = parse_date(harvest)
    observations: list[Observation] = []
    grain = parse_float(cell(row, "yield_dt_ha"))
    if grain is not None:
        observations.append(make_observation("yield", on, "yield", "kg/m^2", to_kg_per_m2(grain, "dt/ha")))
    protein = parse_float(cell(row, "protein_percent"))
    if protein is not None:
        observations.append(make_observation("grain_protein", on, "grain_protein", "kg/kg", to_kg_per_kg(protein, "%")))
    biomass = parse_float(cell(row, "aboveground_biomass_dt_ha"))
    if biomass is not None:
        observations.append(make_observation("biomass", on, "biomass", "kg/m^2", to_kg_per_m2(biomass, "dt/ha")))
    return tuple(observations)


def _harvest_date(sowing: date, season: str) -> date:
    if sowing.month >= 8:
        return date(sowing.year + 1, 7, 31)
    return date(int(season), 8, 1)
