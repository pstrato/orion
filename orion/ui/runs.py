"""Simulation runs: configurations, optional datasets, overlay colours."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from orion.ui.configuration import DEFAULT_CONFIGURATION_COLORS, Configuration

INPUT_DATASETS = "datasets"
INPUT_CUSTOM = "custom"

DATASET_SUMMARY_COLUMNS: list[dict[str, str]] = [
    {"name": "source", "label": "Source", "field": "source", "align": "left"},
    {"name": "season", "label": "Season", "field": "season", "align": "left"},
    {"name": "treatment", "label": "Treatment", "field": "treatment", "align": "left"},
    {"name": "cultivar", "label": "Cultivar", "field": "cultivar", "align": "left"},
    {"name": "latitude", "label": "Latitude", "field": "latitude", "align": "left"},
    {"name": "longitude", "label": "Longitude", "field": "longitude", "align": "left"},
    {"name": "start", "label": "Start", "field": "start", "align": "left"},
    {"name": "end", "label": "End", "field": "end", "align": "left"},
    {"name": "days", "label": "Days", "field": "days", "align": "left"},
]


class SiteYear(Protocol):
    """Dataset row the Inputs tab can list. Datasets are still being refactored."""

    name: str
    source: str
    season: str
    treatment: str
    cultivar: str
    location: Any

    def horizon(self) -> tuple[date, date]: ...


@dataclass(frozen=True)
class SimulationRun:
    """One Simulate job: a configuration, optionally bound to a site-year dataset."""

    name: str
    color: str
    configuration: Configuration
    site_year: SiteYear | None = None


def dataset_select_options(site_years: tuple[SiteYear, ...]) -> dict[str, str]:
    """NiceGUI select options: site-year name → Inputs-tab label."""
    options: dict[str, str] = {}
    for site in site_years:
        label = f"{site.source} · {site.season} · {site.treatment}"
        if site.cultivar:
            label = f"{label} ({site.cultivar})"
        options[site.name] = label
    return options


def selected_site_years(catalogue: tuple[SiteYear, ...], names: list[str] | tuple[str, ...]) -> tuple[SiteYear, ...]:
    """Resolve Inputs-tab selection, keeping the user's order."""
    by_name = {site.name: site for site in catalogue}
    return tuple(by_name[name] for name in names if name in by_name)


def dataset_leaf_label(site: SiteYear) -> str:
    """Treatment label for a hierarchy leaf, with cultivar when known."""
    if site.cultivar:
        return f"{site.treatment} ({site.cultivar})"
    return site.treatment


def dataset_hierarchy(site_years: tuple[SiteYear, ...]) -> list[dict[str, Any]]:
    """Inputs-tab tree: source → season → treatment leaves."""
    grouped: dict[str, dict[str, list[SiteYear]]] = defaultdict(lambda: defaultdict(list))
    for site in site_years:
        grouped[site.source][site.season].append(site)
    tree: list[dict[str, Any]] = []
    for source in sorted(grouped):
        seasons = grouped[source]
        season_nodes: list[dict[str, Any]] = []
        source_names: list[str] = []
        for season in sorted(seasons):
            leaves = [
                {
                    "id": site.name,
                    "label": dataset_leaf_label(site),
                    "children": [],
                    "site_names": [site.name],
                }
                for site in seasons[season]
            ]
            names = [site.name for site in seasons[season]]
            source_names.extend(names)
            season_nodes.append(
                {
                    "id": f"season:{source}:{season}",
                    "label": season,
                    "children": leaves,
                    "site_names": names,
                }
            )
        tree.append(
            {
                "id": f"source:{source}",
                "label": source,
                "children": season_nodes,
                "site_names": source_names,
            }
        )
    return tree


def ticked_site_year_names(ticked: list[str] | tuple[str, ...], catalogue: tuple[SiteYear, ...]) -> list[str]:
    """Keep ticked tree ids that are real site-year names."""
    valid = {site.name for site in catalogue}
    return [name for name in ticked if name in valid]


def active_site_years(mode: str, catalogue: tuple[SiteYear, ...], names: list[str] | tuple[str, ...]) -> tuple[SiteYear, ...]:
    """Site-years Simulate uses: selection in datasets mode, none for custom location."""
    if mode != INPUT_DATASETS:
        return ()
    return selected_site_years(catalogue, names)


def planned_runs(configurations: tuple[Configuration, ...], site_years: tuple[SiteYear, ...]) -> tuple[SimulationRun, ...]:
    """Runs to simulate together: enabled configurations, or datasets × configurations."""
    enabled = tuple(config for config in configurations if config.enabled)
    if not site_years:
        return tuple(SimulationRun(name=config.name, color=config.color, configuration=config) for config in enabled)
    runs: list[SimulationRun] = []
    index = 0
    many_configs = len(enabled) > 1
    for site in site_years:
        for config in enabled:
            name = f"{site.name} · {config.name}" if many_configs else site.name
            if len(site_years) == 1:
                color = config.color
            else:
                color = DEFAULT_CONFIGURATION_COLORS[index % len(DEFAULT_CONFIGURATION_COLORS)]
            runs.append(SimulationRun(name=name, color=color, configuration=config, site_year=site))
            index += 1
    return tuple(runs)


def dataset_summary_rows(sites: tuple[SiteYear, ...]) -> list[dict[str, Any]]:
    """Table rows for selected site-years: identity, location, and horizon."""
    rows: list[dict[str, Any]] = []
    for site in sites:
        start, end = site.horizon()
        point = _centroid_point(site.location)
        rows.append(
            {
                "source": site.source,
                "season": site.season,
                "treatment": site.treatment,
                "cultivar": site.cultivar,
                "latitude": round(float(point.y), 4),
                "longitude": round(float(point.x), 4),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": (end - start).days,
            }
        )
    return rows


def dataset_map_markers(sites: tuple[SiteYear, ...]) -> tuple[tuple[float, float], ...]:
    """Leaflet (latitude, longitude) for each selected site-year."""
    return tuple((float(point.y), float(point.x)) for site in sites for point in (_centroid_point(site.location),))


def dataset_map_view(sites: tuple[SiteYear, ...]) -> tuple[float, float, int]:
    """Map centre and zoom covering selected site-years."""
    if not sites:
        return 20.0, 10.0, 2
    lats = [float(_centroid_point(site.location).y) for site in sites]
    lons = [float(_centroid_point(site.location).x) for site in sites]
    lat = (min(lats) + max(lats)) / 2
    lon = (min(lons) + max(lons)) / 2
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    if len(sites) == 1 or span < 0.5:
        zoom = 8
    elif span < 5:
        zoom = 5
    elif span < 40:
        zoom = 3
    else:
        zoom = 2
    return lat, lon, zoom


def _centroid_point(location: Any) -> Any:
    """Point for a location whose centroid is either a Point or a Constant[Point]."""
    centroid = location.centroid
    return getattr(centroid, "value", centroid)
