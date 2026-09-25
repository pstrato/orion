"""Braunschweig FACE winter-wheat experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from orion.datasets.convert import to_kg_per_kg, to_kg_per_m2
from orion.datasets.events import ManagementEvent, ProtectionEvent, fertiliser_event, irrigation_event, protection_event, sowing_event
from orion.datasets.io import cell, event_name, group_rows, parse_date, parse_float, point_location, read_rows
from orion.datasets.observations import Observation, make_observation
from orion.datasets.protection import protection_kind
from orion.datasets.site_year import SiteYear

BRAUNSCHWEIG_LOCATION = point_location("braunschweig", 10.45, 52.3)
SOURCE = "braunschweig"


def parse_braunschweig(directory: Path) -> tuple[SiteYear, ...]:
    """Parse Braunschweig management, biomass, and grain-quality worksheets."""
    management = group_rows(read_rows(directory / "management.csv"), "treatment")
    quality = group_rows(read_rows(directory / "grain_quality.csv"), "treatment")
    biomass = group_rows(read_rows(directory / "biomass.csv"), "treatment")

    years: list[SiteYear] = []
    for (treatment,), action_rows in management.items():
        cultivar = cell(action_rows[0], "cultivar")
        events: list[ManagementEvent] = []
        sowing_on: date | None = None
        for row in action_rows:
            measure = cell(row, "measure").lower()
            on = parse_date(cell(row, "date"))
            product = cell(row, "product")
            if measure == "sowing":
                sowing_on = on
                density = parse_float(cell(row, "rate"))
                events.append(sowing_event(event_name("sowing", on), on, density=density))
                if product:
                    cultivar = product
            elif measure in {"fertilisation", "fertilization", "fertiliser"}:
                n_rate = parse_float(cell(row, "n_kg_ha"), 0.0) or 0.0
                events.append(fertiliser_event(event_name("fertiliser", on), on, n=to_kg_per_m2(n_rate, "kg/ha"), product=product))
            elif measure in {"pesticide", "protection"}:
                rate = parse_float(cell(row, "rate"))
                events.append(
                    protection_event(
                        event_name("protection", on),
                        on,
                        kind=protection_kind(product=product, measure=measure),
                        product=product,
                        rate=rate,
                        rate_unit="l/ha" if rate is not None else "",
                    )
                )
            elif measure == "irrigation":
                amount = parse_float(cell(row, "irrigation_mm"), 0.0) or 0.0
                events.append(irrigation_event(event_name("irrigation", on), on, to_kg_per_m2(amount, "mm")))
        if not any(isinstance(event, ProtectionEvent) for event in events):
            on = sowing_on or parse_date(cell(action_rows[0], "date"))
            events.append(protection_event(event_name("protection", on), on, kind="standard"))
        season = str((sowing_on or parse_date(cell(action_rows[0], "date"))).year)
        observations = _observations(quality.get((treatment,), []), biomass.get((treatment,), []))
        years.append(
            SiteYear(
                name=f"{SOURCE}:{season}:{treatment}",
                crop="wheat",
                cultivar=cultivar,
                season=season,
                treatment=treatment,
                source=SOURCE,
                location=BRAUNSCHWEIG_LOCATION,
                events=tuple(events),
                observations=observations,
            )
        )
    return tuple(years)


def _observations(quality_rows: list[dict[str, str]], biomass_rows: list[dict[str, str]]) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for row in biomass_rows:
        on = parse_date(cell(row, "date"))
        grain = parse_float(cell(row, "yield_t_ha"))
        if grain is not None:
            observations.append(make_observation("yield", on, "yield", "kg/m^2", to_kg_per_m2(grain, "t/ha")))
        mass = parse_float(cell(row, "biomass_t_ha"))
        if mass is not None:
            observations.append(make_observation("biomass", on, "biomass", "kg/m^2", to_kg_per_m2(mass, "t/ha")))
    for row in quality_rows:
        on = parse_date(cell(row, "date"))
        protein = parse_float(cell(row, "protein_percent"))
        if protein is not None:
            observations.append(make_observation("grain_protein", on, "grain_protein", "kg/kg", to_kg_per_kg(protein, "%")))
    return tuple(observations)
