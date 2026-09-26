"""Behaviour: SoilGrids builds a soil profile from the location, with resampled layers below 30 cm."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import requests
from shapely import Point

from orion.clients.soilgrids import (
    SOILGRIDS_DEPTH_LABELS,
    SOILGRIDS_PROPERTIES,
    SOILGRIDS_URL,
    SoilGridInput,
    clear_soilgrids_memory_cache,
    fetch_soilgrids,
)
from orion.core.constant import const
from orion.core.input import LocationInput
from orion.core.setting import Settings
from orion.processes.soil.soil import Soil


class _Response:
    def __init__(self, payload: dict[str, Any] | None, *, error: bool = False):
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise requests.HTTPError("SoilGrids request failed")

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload


class _Session(requests.Session):
    def __init__(self, payload: dict[str, Any] | None = None, *, error: bool = False):
        super().__init__()
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def get(self, url, params=None, **kwargs):  # type: ignore[override]
        pairs = [(str(key), str(value)) for key, value in params] if isinstance(params, list) else []
        self.calls.append((str(url), pairs))
        return _Response(self.payload, error=self.error)  # type: ignore[return-value]


def _payload(
    values: Mapping[str, Mapping[str, float | None]] | None = None,
    d_factors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    values = values or {}
    d_factors = d_factors or {}
    layers = []
    for prop in (*SOILGRIDS_PROPERTIES, "unknown"):
        depths = []
        for label in SOILGRIDS_DEPTH_LABELS:
            mean = values.get(prop, {}).get(label, 0.0)
            depths.append({"label": label, "values": {} if mean is None else {"mean": mean}})
        layers.append({"name": prop, "unit_measure": {"d_factor": d_factors.get(prop, 1.0)}, "depths": depths})
    return {"properties": {"layers": layers}}


def _settings(tmp_path) -> Settings:
    return Settings("test", tmp_path, False, False, False)


def _location(lon: float = 0.1, lat: float = 51.5) -> LocationInput:
    return LocationInput("field", const("geometry", "coordinate", Point(lon, lat), description="field"))


def _soil_from(payload: dict[str, Any], tmp_path, monkeypatch) -> Soil:
    monkeypatch.setattr("orion.clients.soilgrids.fetch_soilgrids", lambda lon, lat, **kwargs: payload)
    soil = SoilGridInput("soil").states(_settings(tmp_path), _location())
    assert isinstance(soil, Soil)
    return soil


def test_soilgrids_query_asks_for_mean_properties_at_the_standard_depths():
    clear_soilgrids_memory_cache()
    session = _Session(_payload())
    payload = fetch_soilgrids(0.1, 51.5, session=session)
    assert payload["properties"]["layers"]
    assert len(session.calls) == 1
    url, params = session.calls[0]
    assert url == SOILGRIDS_URL
    assert ("lon", "0.1") in params
    assert ("lat", "51.5") in params
    assert ("value", "mean") in params
    for prop in SOILGRIDS_PROPERTIES:
        assert ("property", prop) in params
    for depth in SOILGRIDS_DEPTH_LABELS:
        assert ("depth", depth) in params


def test_repeated_soilgrids_query_for_the_same_point_is_not_sent_again(tmp_path, monkeypatch):
    clear_soilgrids_memory_cache()
    session = _Session(_payload())
    monkeypatch.setattr("orion.clients.soilgrids.cached_session", lambda path: session)
    fetch_soilgrids(0.1, 51.5, cache_path=str(tmp_path))
    fetch_soilgrids(0.1, 51.5, cache_path=str(tmp_path))
    assert len(session.calls) == 1


def test_soilgrids_http_error_is_raised():
    clear_soilgrids_memory_cache()
    with pytest.raises(requests.HTTPError):
        fetch_soilgrids(0.1, 51.5, session=_Session(error=True))


def test_soil_profile_keeps_intervals_to_30cm_then_uses_15cm_layers_and_a_20cm_base(tmp_path, monkeypatch):
    soil = _soil_from(_payload(), tmp_path, monkeypatch)
    bounds = [(float(layer.top.value), float(layer.bottom.value)) for layer in soil.layers]
    assert bounds[0] == pytest.approx((0.0, 0.05))
    assert bounds[1] == pytest.approx((0.05, 0.15))
    assert bounds[2] == pytest.approx((0.15, 0.30))
    assert bounds[3][1] - bounds[3][0] == pytest.approx(0.15)
    assert bounds[-1] == pytest.approx((1.80, 2.0))


def test_soil_profile_converts_mass_fractions_and_blends_depths_by_thickness(tmp_path, monkeypatch):
    clay: dict[str, float | None] = {label: 0.0 for label in SOILGRIDS_DEPTH_LABELS}
    clay["0-5cm"] = None
    clay["60-100cm"] = 30.0
    clay["100-200cm"] = 60.0
    soil = _soil_from(
        _payload(
            {
                "clay": clay,
                "bdod": {label: 130.0 for label in SOILGRIDS_DEPTH_LABELS},
                "phh2o": {label: 65.0 for label in SOILGRIDS_DEPTH_LABELS},
                "nitrogen": {label: 5000.0 for label in SOILGRIDS_DEPTH_LABELS},
            },
            {"bdod": 100.0, "phh2o": 10.0, "nitrogen": 1000.0},
        ),
        tmp_path,
        monkeypatch,
    )
    assert float(soil.layers[0].clay.value) == pytest.approx(0.0)
    blended = next(layer for layer in soil.layers if float(layer.top.value) == pytest.approx(0.90))
    assert float(blended.clay.value) == pytest.approx(0.04)
    assert float(soil.layers[0].bdod.value) == pytest.approx(1300.0)
    assert soil.layers[0].bdod.unit == "kg/m^3"
    assert float(soil.layers[0].phh2o.value) == pytest.approx(6.5)
    assert float(soil.layers[0].organic_nitrogen.value) == pytest.approx(0.005)
    assert soil.layers[0].clay.unit == "kg/kg"


def test_soil_state_queries_the_location_longitude_and_latitude(tmp_path, monkeypatch):
    seen: dict[str, float] = {}

    def fake_fetch(lon: float, lat: float, **kwargs):
        seen["lon"] = lon
        seen["lat"] = lat
        return _payload()

    monkeypatch.setattr("orion.clients.soilgrids.fetch_soilgrids", fake_fetch)
    SoilGridInput("soil").states(_settings(tmp_path), _location(0.1, 51.5))
    assert seen["lon"] == pytest.approx(0.1)
    assert seen["lat"] == pytest.approx(51.5)
