"""CSV and value helpers for wheat experiment archives."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from shapely import Point

from orion.core.constant import const
from orion.core.input import LocationInput

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y%m%d")


def read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Read a CSV or tab-delimited table into lowercase-keyed rows."""
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    rows: list[dict[str, str]] = []
    for raw in reader:
        row = {str(key).strip().lower(): "" if value is None else str(value).strip() for key, value in raw.items() if key is not None}
        if any(row.values()):
            rows.append(row)
    return tuple(rows)


def cell(row: dict[str, str], *names: str, default: str = "") -> str:
    """Return the first present column value (names are matched lowercase)."""
    for name in names:
        value = row.get(name.lower())
        if value:
            return value
    return default


def parse_date(value: str) -> date:
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date {value!r}.")


def parse_float(value: str, default: float | None = None) -> float | None:
    text = value.strip().replace(",", ".")
    if not text:
        return default
    return float(text)


def is_wheat(crop: str) -> bool:
    return "wheat" in crop.lower()


def yes(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "1"}


def point_location(name: str, longitude: float, latitude: float, altitude: float = 100.0) -> LocationInput:
    """Location whose geometry is a single longitude/latitude point."""
    return LocationInput(
        name,
        const("geometry", "coordinate", Point(longitude, latitude), description=name),
        altitude=const("altitude", "m", altitude, "Altitude above sea level"),
    )


def location_from_row(row: dict[str, str], fallback: LocationInput) -> LocationInput:
    lat = parse_float(cell(row, "latitude", "lat"))
    lon = parse_float(cell(row, "longitude", "lon", "long"))
    altitude = parse_float(cell(row, "altitude", "elevation", "elev"))
    name = cell(row, "site", "name") or fallback.name
    height = float(fallback.altitude.value) if altitude is None else altitude
    if lat is None or lon is None:
        if name == fallback.name and altitude is None:
            return fallback
        return point_location(name, fallback.geometry.value.x, fallback.geometry.value.y, height)
    return point_location(name, lon, lat, 0.0 if altitude is None else altitude)


def group_rows(rows: tuple[dict[str, str], ...], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(cell(row, key) for key in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def event_name(kind: str, on: date) -> str:
    return f"{kind}-{on.isoformat()}"


def as_mapping(rows: tuple[dict[str, str], ...], key: str) -> dict[str, dict[str, str]]:
    return {cell(row, key): row for row in rows if cell(row, key)}
