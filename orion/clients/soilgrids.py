from typing import Any

import requests

from orion.clients.http_cache import cached_session
from orion.core.entity import entity
from orion.core.input import Input, Inputs, LocationInput
from orion.core.process import Process
from orion.datasets.convert import to_kg_per_kg, to_kg_per_m3
from orion.processes.soil.soil import Soil, SoilLayer, make_soil_layer

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

_soilgrids_memory: dict[tuple[object, ...], dict[str, Any]] = {}


def clear_soilgrids_memory_cache() -> None:
    """Drop in-process SoilGrids payloads (tests / explicit invalidation)."""
    _soilgrids_memory.clear()


SOILGRIDS_DEPTHS_M: tuple[tuple[float, float], ...] = (
    (0.0, 0.05),
    (0.05, 0.15),
    (0.15, 0.30),
    (0.30, 0.60),
    (0.60, 1.00),
    (1.00, 2.00),
)

SOILGRIDS_DEPTH_LABELS: tuple[str, ...] = (
    "0-5cm",
    "5-15cm",
    "15-30cm",
    "30-60cm",
    "60-100cm",
    "100-200cm",
)

SOIL_RESAMPLE_FROM_M = 0.30
"""Keep SoilGrids intervals down to this depth; resample below it."""

SOIL_LAYER_THICKNESS_M = 0.15
"""Layer thickness used when splitting SoilGrids intervals below ``SOIL_RESAMPLE_FROM_M``."""

SOIL_LAST_LAYER_THICKNESS_M = 0.20
"""Thickness of the deepest simulation layer."""

# Properties fetched as SoilLayer constants (SoilGrids names → our field names).
SOILGRIDS_PROPERTIES: tuple[str, ...] = (
    "clay",
    "sand",
    "silt",
    "bdod",
    "phh2o",
    "soc",
    "cec",
    "nitrogen",
)


def fetch_soilgrids(
    lon: float,
    lat: float,
    *,
    cache_path: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Query SoilGrids v2 properties/mean for standard depths at a point."""
    owned_session = session is None
    memory_key = (round(lon, 5), round(lat, 5), cache_path)
    if owned_session and memory_key in _soilgrids_memory:
        return _soilgrids_memory[memory_key]

    if session is None:
        if cache_path is None:
            session = requests.Session()
        else:
            session = cached_session(cache_path)
    params: list[tuple[str, str]] = [
        ("lon", str(lon)),
        ("lat", str(lat)),
        ("value", "mean"),
    ]
    for prop in SOILGRIDS_PROPERTIES:
        params.append(("property", prop))
    for depth in SOILGRIDS_DEPTH_LABELS:
        params.append(("depth", depth))
    response = session.get(SOILGRIDS_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if owned_session:
        _soilgrids_memory[memory_key] = payload
    return payload


def _layer_values(payload: dict[str, Any]) -> list[dict[str, float]]:
    """Parse SoilGrids JSON into one property dict per depth interval."""
    layers_out: list[dict[str, float]] = [{} for _ in SOILGRIDS_DEPTH_LABELS]
    label_index = {label: i for i, label in enumerate(SOILGRIDS_DEPTH_LABELS)}
    for prop_layer in payload.get("properties", {}).get("layers", []):
        name = prop_layer.get("name")
        if name not in SOILGRIDS_PROPERTIES:
            continue
        unit = prop_layer.get("unit_measure") or {}
        d_factor = float(unit.get("d_factor") or 1.0)
        for depth in prop_layer.get("depths", []):
            label = depth.get("label")
            if label not in label_index:
                continue
            raw = (depth.get("values") or {}).get("mean")
            if raw is None:
                value = 0.0
            else:
                value = float(raw) / d_factor
            field = "organic_nitrogen" if name == "nitrogen" else name
            layers_out[label_index[label]][field] = value
    return layers_out


def _overlap_m(top_a: float, bottom_a: float, top_b: float, bottom_b: float) -> float:
    return max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))


def _simulation_layer_bounds() -> tuple[tuple[float, float], ...]:
    """Native SoilGrids intervals to 30 cm, then 15 cm layers, last layer 20 cm."""
    native = tuple((top, bottom) for top, bottom in SOILGRIDS_DEPTHS_M if bottom <= SOIL_RESAMPLE_FROM_M + 1e-12)
    profile_bottom = SOILGRIDS_DEPTHS_M[-1][1]
    last_top = profile_bottom - SOIL_LAST_LAYER_THICKNESS_M
    bounds = list(native)
    top = native[-1][1] if native else 0.0
    while top + SOIL_LAYER_THICKNESS_M <= last_top + 1e-12:
        bottom = top + SOIL_LAYER_THICKNESS_M
        bounds.append((top, bottom))
        top = bottom
    if not bounds or abs(bounds[-1][1] - profile_bottom) > 1e-12:
        bounds.append((top, profile_bottom))
    return tuple(bounds)


def _resample_layers(per_depth: list[dict[str, float]]) -> list[tuple[float, float, dict[str, float]]]:
    """Thickness-weight SoilGrids intervals onto the simulation layer bounds."""
    resampled: list[tuple[float, float, dict[str, float]]] = []
    for top, bottom in _simulation_layer_bounds():
        weights: list[tuple[float, dict[str, float]]] = []
        for (source_top, source_bottom), props in zip(SOILGRIDS_DEPTHS_M, per_depth, strict=True):
            overlap = _overlap_m(top, bottom, source_top, source_bottom)
            if overlap > 0.0:
                weights.append((overlap, props))
        total = sum(weight for weight, _ in weights)
        merged: dict[str, float] = {}
        if total > 0.0:
            keys = {key for _, props in weights for key in props}
            merged = {key: sum(weight * props.get(key, 0.0) for weight, props in weights) / total for key in keys}
        resampled.append((top, bottom, merged))
    return resampled


@entity()
class SoilGridInput(Input):
    """Creates a Soil state from SoilGrids at each location sample point."""

    def states(self, inputs: Inputs, location: LocationInput):
        """Build a Soil state from a SoilGrids properties query payload.

        Intervals down to 30 cm are kept; below that, properties are thickness-weighted onto 15 cm layers with a 20 cm base.
        """
        cache = str(inputs.settings.cache_path)
        payload = fetch_soilgrids(float(location.geometry.value.y), float(location.geometry.value.x), cache_path=cache)
        resampled = _resample_layers(_layer_values(payload))
        layers: list[SoilLayer] = []
        for i, (top, bottom, props) in enumerate(resampled):
            layers.append(
                make_soil_layer(
                    f"layer-{i}",
                    top,
                    bottom,
                    clay=to_kg_per_kg(props.get("clay", 0.0), "g/kg"),
                    sand=to_kg_per_kg(props.get("sand", 0.0), "g/kg"),
                    silt=to_kg_per_kg(props.get("silt", 0.0), "g/kg"),
                    bdod=to_kg_per_m3(props.get("bdod", 1.3), "kg/dm^3"),
                    phh2o=props.get("phh2o", 7.0),
                    soc=to_kg_per_kg(props.get("soc", 0.0), "g/kg"),
                    cec=props.get("cec", 0.0),
                    organic_nitrogen=to_kg_per_kg(props.get("organic_nitrogen", 0.0), "g/kg"),
                )
            )
        return Soil("soilgrids", None, layers=tuple(layers))

    def processes(self, inputs: Inputs) -> tuple[Process, ...]:
        return ()
