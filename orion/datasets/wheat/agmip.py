"""AgMIP / Kassie–Kimball multi-country wheat modelling pack."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from orion.datasets.convert import to_kg_per_m2
from orion.datasets.events import FertiliserEvent, IrrigationEvent, ProtectionEvent, SowingEvent, fertiliser_event, irrigation_event, protection_event, sowing_event
from orion.datasets.io import cell, event_name, group_rows, location_from_row, parse_date, parse_float, point_location, read_rows
from orion.datasets.observations import Observation, make_observation, observation_from_published
from orion.datasets.protection import protection_kind
from orion.datasets.site_year import SiteYear

SOURCE = "agmip_kassie"
FALLBACK_LOCATION = point_location("agmip", 0.0, 0.0)


def parse_agmip_kassie(directory: Path) -> tuple[SiteYear, ...]:
    """Parse AgMIP management and experimental-data tables.

    Crop protection is recorded as ``standard`` when spray programmes are not listed.
    """
    return parse_management_pack(directory, source=SOURCE)


def parse_management_pack(directory: Path, source: str | None = None) -> tuple[SiteYear, ...]:
    """Parse a site pack: ``management.csv`` plus ``experimental_data.csv``.

    Optional ``fertiliser.csv`` and ``irrigation.csv`` supply dated events and replace the
    seasonal totals. ``harvest_date`` is recorded as BBCH 89 when maturity is not already
    observed. Crop protection uses the optional ``protection`` column, otherwise ``standard``.
    """
    archive = source or directory.name
    management = group_rows(read_rows(directory / "management.csv"), "site", "treatment")
    experimental = group_rows(read_rows(directory / "experimental_data.csv"), "site", "treatment")
    fertiliser_rows = _optional_groups(directory, "fertiliser.csv")
    irrigation_rows = _optional_groups(directory, "irrigation.csv")

    years: list[SiteYear] = []
    for (site_name, treatment), rows in management.items():
        row = rows[0]
        sowing_on = parse_date(cell(row, "sowing_date"))
        cultivar = cell(row, "cultivar")
        density = parse_float(cell(row, "seeding_rate", "density"))
        location = location_from_row(row, point_location(site_name or FALLBACK_LOCATION.name, 0.0, 0.0))
        kind = protection_kind(type_name=cell(row, "protection")) if cell(row, "protection") else "standard"
        events: list[SowingEvent | FertiliserEvent | ProtectionEvent | IrrigationEvent] = [
            sowing_event(event_name("sowing", sowing_on), sowing_on, density=density),
            *_fertiliser_events(fertiliser_rows.get((site_name, treatment), []), row, sowing_on),
            protection_event(event_name("protection", sowing_on), sowing_on, kind=kind),
            *_irrigation_events(irrigation_rows.get((site_name, treatment), []), row, sowing_on),
        ]
        observations = _observations(experimental.get((site_name, treatment), []))
        observations = _with_harvest(observations, cell(row, "harvest_date"))
        season = str(sowing_on.year)
        years.append(
            SiteYear(
                name=f"{archive}:{site_name}:{treatment}",
                crop="wheat",
                cultivar=cultivar,
                season=season,
                treatment=treatment,
                source=archive,
                location=location,
                events=tuple(events),
                observations=observations,
            )
        )
    return tuple(years)


def _optional_groups(directory: Path, filename: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    path = directory / filename
    if not path.is_file():
        return {}
    return group_rows(read_rows(path), "site", "treatment")


def _fertiliser_events(rows: list[dict[str, str]], management: dict[str, str], sowing_on: date) -> tuple[FertiliserEvent, ...]:
    """Dated fertiliser rows replace the management-table nitrogen total."""
    if rows:
        events: list[FertiliserEvent] = []
        for row in rows:
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
        return tuple(events)
    if not cell(management, "n_kg_ha"):
        return ()
    n_rate = parse_float(cell(management, "n_kg_ha"), 0.0) or 0.0
    return (fertiliser_event(event_name("fertiliser", sowing_on), sowing_on, n=to_kg_per_m2(n_rate, "kg/ha")),)


def _irrigation_events(rows: list[dict[str, str]], management: dict[str, str], sowing_on: date) -> tuple[IrrigationEvent, ...]:
    """Dated irrigation rows replace the management-table seasonal total."""
    if rows:
        events: list[IrrigationEvent] = []
        for row in rows:
            amount = parse_float(cell(row, "irrigation_mm"), 0.0) or 0.0
            if not amount:
                continue
            on = parse_date(cell(row, "date"))
            events.append(irrigation_event(event_name("irrigation", on), on, to_kg_per_m2(amount, "mm")))
        return tuple(events)
    amount = parse_float(cell(management, "irrigation_mm"), 0.0) or 0.0
    if not amount:
        return ()
    return (irrigation_event(event_name("irrigation", sowing_on), sowing_on, to_kg_per_m2(amount, "mm")),)


def _observations(rows: list[dict[str, str]]) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for row in rows:
        on = parse_date(cell(row, "date")) if cell(row, "date") else date.min
        value = parse_float(cell(row, "value"))
        if value is None:
            continue
        observed = observation_from_published(on, cell(row, "variable"), value, cell(row, "unit"))
        if observed is not None:
            observations.append(observed)
    return tuple(observations)


def _with_harvest(observations: tuple[Observation, ...], harvest_date: str) -> tuple[Observation, ...]:
    """Record a published harvest date as fully ripe (BBCH 89) when maturity is absent."""
    if not harvest_date:
        return observations
    on = parse_date(harvest_date)
    if any(item.reading.name == "phenology_stage" and item.on.value == on for item in observations):
        return observations
    ripe = make_observation(f"phenology_stage-{on.isoformat()}", on, "phenology_stage", "step", 89.0)
    return (*observations, ripe)
