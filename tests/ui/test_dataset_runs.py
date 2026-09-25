"""Behaviour: selected datasets become concurrent simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from shapely import Point

from orion.core.constant import const
from orion.core.input import LocationInput
from orion.ui.configuration import Configuration, default_configuration, next_configuration_color
from orion.ui.runs import (
    INPUT_CUSTOM,
    INPUT_DATASETS,
    active_site_years,
    dataset_hierarchy,
    dataset_map_markers,
    dataset_map_view,
    dataset_select_options,
    dataset_summary_rows,
    planned_runs,
    selected_site_years,
    ticked_site_year_names,
)


@dataclass
class _Event:
    name: str
    on: date


@dataclass
class _Site:
    name: str
    crop: str
    cultivar: str
    season: str
    treatment: str
    source: str
    location: LocationInput
    events: tuple[_Event, ...] = ()
    observations: tuple[object, ...] = ()

    def horizon(self) -> tuple[date, date]:
        dates = [event.on for event in self.events]
        start = min(dates)
        end = max(dates) + timedelta(days=1)
        return start, end


def _location(name: str, longitude: float, latitude: float) -> LocationInput:
    return LocationInput(name, geometry=const("geometry", "coordinate", Point(longitude, latitude), "Trial location"))


def _site(
    name: str,
    source: str = "westerfeld",
    season: str = "2018",
    treatment: str = "intensive",
    cultivar: str = "RGT Reform",
    longitude: float = 0.0,
    latitude: float = 50.0,
    on: date = date(2018, 10, 1),
) -> _Site:
    return _Site(
        name=name,
        crop="wheat",
        cultivar=cultivar,
        season=season,
        treatment=treatment,
        source=source,
        location=_location(source, longitude, latitude),
        events=(_Event("sowing", on),),
    )


def test_planned_runs_without_datasets_are_enabled_configurations():
    defaults = default_configuration()
    extra = Configuration(name="n-trial", enabled=True, color="#C45C26", processes=())
    disabled = Configuration(name="off", enabled=False, color="#2B6CB0", processes=())
    runs = planned_runs((defaults, extra, disabled), ())
    assert [run.name for run in runs] == ["defaults", "n-trial"]
    assert runs[0].site_year is None
    assert runs[0].color == defaults.color
    assert runs[1].configuration.name == "n-trial"


def test_planned_runs_one_per_selected_dataset_when_one_configuration_enabled():
    west = _site("westerfeld:2018:intensive")
    agmip = _site("agmip_kassie:Maricopa:WET-HIGHN", source="agmip_kassie", treatment="WET-HIGHN", cultivar="Yecora Rojo")
    runs = planned_runs((default_configuration(),), (west, agmip))
    assert [run.name for run in runs] == [west.name, agmip.name]
    assert runs[0].site_year is west
    assert runs[1].site_year is agmip
    assert runs[0].color != runs[1].color


def test_planned_runs_are_the_product_of_datasets_and_enabled_configurations():
    west = _site("westerfeld:2018:intensive")
    agmip = _site("agmip_kassie:Lincoln:IRRIGATED", source="agmip_kassie", treatment="IRRIGATED")
    extra = Configuration(name="n-trial", enabled=True, color=next_configuration_color((default_configuration(),)), processes=())
    runs = planned_runs((default_configuration(), extra), (west, agmip))
    assert [run.name for run in runs] == [
        f"{west.name} · defaults",
        f"{west.name} · n-trial",
        f"{agmip.name} · defaults",
        f"{agmip.name} · n-trial",
    ]
    assert {run.configuration.name for run in runs} == {"defaults", "n-trial"}
    assert runs[0].site_year is west
    assert runs[2].site_year is agmip


def test_dataset_select_options_label_source_season_treatment_and_cultivar():
    west = _site("westerfeld:2018:intensive")
    options = dataset_select_options((west,))
    assert options[west.name] == "westerfeld · 2018 · intensive (RGT Reform)"


def test_selected_site_years_keep_input_order_and_drop_unknown_names():
    west = _site("westerfeld:2018:intensive")
    agmip = _site("agmip_kassie:Lincoln:IRRIGATED", source="agmip_kassie", treatment="IRRIGATED")
    selected = selected_site_years((agmip, west), ["missing", west.name, agmip.name])
    assert selected == (west, agmip)


def test_dataset_hierarchy_groups_source_then_season_then_treatment():
    west = _site("westerfeld:2018:intensive")
    maricopa = _site("agmip_kassie:Maricopa:WET-HIGHN", source="agmip_kassie", season="1992", treatment="WET-HIGHN", cultivar="Yecora Rojo")
    lincoln = _site("agmip_kassie:Lincoln:IRRIGATED", source="agmip_kassie", season="1991", treatment="IRRIGATED", cultivar="Rongotea")
    tree = dataset_hierarchy((west, maricopa, lincoln))
    assert [node["label"] for node in tree] == ["agmip_kassie", "westerfeld"]
    agmip = tree[0]
    assert agmip["id"] == "source:agmip_kassie"
    assert [child["label"] for child in agmip["children"]] == ["1991", "1992"]
    lincoln_leaf = agmip["children"][0]["children"][0]
    assert lincoln_leaf["id"] == lincoln.name
    assert lincoln_leaf["label"] == "IRRIGATED (Rongotea)"
    assert lincoln_leaf["children"] == []
    assert set(agmip["site_names"]) == {lincoln.name, maricopa.name}
    west_leaf = tree[1]["children"][0]["children"][0]
    assert west_leaf["id"] == west.name
    assert west_leaf["label"] == "intensive (RGT Reform)"


def test_ticked_hierarchy_ids_keep_only_site_year_names():
    west = _site("westerfeld:2018:intensive")
    lincoln = _site("agmip_kassie:Lincoln:IRRIGATED", source="agmip_kassie", treatment="IRRIGATED")
    ticked = ["source:agmip_kassie", "season:westerfeld:2018", west.name, "missing"]
    assert ticked_site_year_names(ticked, (west, lincoln)) == [west.name]


def test_active_site_years_use_selection_only_in_datasets_mode():
    west = _site("westerfeld:2018:intensive")
    assert active_site_years(INPUT_CUSTOM, (west,), [west.name]) == ()
    assert active_site_years(INPUT_DATASETS, (west,), [west.name]) == (west,)
    assert active_site_years(INPUT_DATASETS, (west,), []) == ()


def test_dataset_summary_rows_include_location_horizon_and_treatment():
    west = _site("westerfeld:2018:intensive", longitude=11.702, latitude=51.819, on=date(2018, 10, 12))
    rows = dataset_summary_rows((west,))
    assert rows == [
        {
            "source": "westerfeld",
            "season": "2018",
            "treatment": "intensive",
            "cultivar": "RGT Reform",
            "latitude": 51.819,
            "longitude": 11.702,
            "start": "2018-10-12",
            "end": "2018-10-13",
            "days": 1,
        }
    ]


def test_dataset_map_markers_are_latitude_longitude_pairs():
    west = _site("westerfeld:2018:intensive", longitude=11.702, latitude=51.819, on=date(2018, 10, 12))
    maricopa = _site(
        "agmip_kassie:Maricopa:WET-HIGHN",
        source="agmip_kassie",
        season="1992",
        treatment="WET-HIGHN",
        cultivar="Yecora Rojo",
        longitude=-111.98,
        latitude=33.06,
        on=date(1992, 1, 1),
    )
    assert dataset_map_markers((west, maricopa)) == ((51.819, 11.702), (33.06, -111.98))
    lat, lon, zoom = dataset_map_view((west, maricopa))
    assert 33.06 < lat < 51.819
    assert -111.98 < lon < 11.702
    assert zoom <= 4


def test_dataset_summary_and_map_are_empty_when_nothing_is_selected():
    assert dataset_summary_rows(()) == []
    assert dataset_map_markers(()) == ()
    lat, lon, zoom = dataset_map_view(())
    assert zoom <= 2
    assert lat == 20.0
    assert lon == 10.0
