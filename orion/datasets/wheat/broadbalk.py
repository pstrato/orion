"""Rothamsted Broadbalk wheat experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from orion.datasets.convert import to_kg_per_kg, to_kg_per_m2
from orion.datasets.events import ProtectionEvent, fertiliser_event, protection_event, sowing_event
from orion.datasets.io import cell, event_name, parse_date, parse_float, point_location, read_rows, yes
from orion.datasets.observations import make_observation
from orion.datasets.site_year import SiteYear

BROADBALK_LOCATION = point_location("broadbalk", -0.37, 51.81)
SOURCE = "broadbalk"


def parse_broadbalk(directory: Path) -> tuple[SiteYear, ...]:
    """Parse Broadbalk yield tables with fertiliser and protection contrasts."""
    rows = read_rows(directory / "yields.csv")
    years: list[SiteYear] = []
    for row in rows:
        season = cell(row, "year")
        section = cell(row, "section")
        plot = cell(row, "plot")
        treatment = f"section-{section}-plot-{plot}"
        sowing_on = parse_date(cell(row, "sowing_date"))
        harvest_on = parse_date(cell(row, "harvest_date")) if cell(row, "harvest_date") else sowing_on
        n_rate = parse_float(cell(row, "n_kg_ha"), 0.0) or 0.0
        events = [
            sowing_event(event_name("sowing", sowing_on), sowing_on),
            fertiliser_event(event_name("fertiliser", sowing_on), sowing_on, n=to_kg_per_m2(n_rate, "kg/ha")),
            *_protection_events(row, sowing_on),
        ]
        grain = parse_float(cell(row, "grain_t_ha"), 0.0) or 0.0
        straw = parse_float(cell(row, "straw_t_ha"), 0.0) or 0.0
        observations = [
            make_observation("yield", harvest_on, "yield", "kg/m^2", to_kg_per_m2(grain, "t/ha")),
            make_observation("biomass", harvest_on, "biomass", "kg/m^2", to_kg_per_m2(grain + straw, "t/ha")),
        ]
        grain_n = parse_float(cell(row, "grain_n_percent"))
        if grain_n is not None:
            observations.append(make_observation("grain_n", harvest_on, "grain_n", "kg/kg", to_kg_per_kg(grain_n, "%")))
        years.append(
            SiteYear(
                name=f"{SOURCE}:{season}:{treatment}",
                crop="wheat",
                cultivar=cell(row, "cultivar"),
                season=season,
                treatment=treatment,
                source=SOURCE,
                location=BROADBALK_LOCATION,
                events=tuple(events),
                observations=tuple(observations),
            )
        )
    return tuple(years)


def _protection_events(row: dict[str, str], on: date) -> list[ProtectionEvent]:
    herbicides = cell(row, "herbicides")
    fungicides = cell(row, "fungicides")
    events: list[ProtectionEvent] = []
    if fungicides and not yes(fungicides):
        events.append(protection_event(event_name("protection", on), on, kind="none", product="fungicide"))
    if herbicides and not yes(herbicides):
        events.append(protection_event(f"protection-herbicide-{on.isoformat()}", on, kind="none", product="herbicide"))
    if yes(herbicides) and yes(fungicides):
        events.append(protection_event(event_name("protection", on), on, kind="standard"))
    if not events:
        events.append(protection_event(event_name("protection", on), on, kind="standard"))
    return events
