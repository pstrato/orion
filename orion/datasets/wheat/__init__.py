"""Cache-backed wheat experiment archives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from orion.datasets.site_year import SiteYear
from orion.datasets.wheat.agmip import parse_agmip_kassie, parse_management_pack
from orion.datasets.wheat.braunschweig import parse_braunschweig
from orion.datasets.wheat.broadbalk import parse_broadbalk
from orion.datasets.wheat.muncheberg import parse_muncheberg
from orion.datasets.wheat.westerfeld import parse_westerfeld

__all__ = [
    "DATASETS_CACHE",
    "WHEAT_PARSERS",
    "cache_wheat_archive",
    "list_wheat_site_years",
    "parse_agmip_kassie",
    "parse_braunschweig",
    "parse_broadbalk",
    "parse_management_pack",
    "parse_muncheberg",
    "parse_westerfeld",
]

DATASETS_CACHE = "datasets"

WheatParser = Callable[[Path], tuple[SiteYear, ...]]

_MANAGEMENT_PACKS = (
    "agmip_kassie",
    "wageningen",
    "cunderdin",
    "obregon",
    "luancheng",
    "ludhiana",
    "balcarce",
    "egypt_nile",
    "iwyp_valdivia",
    "german_met",
    "eest",
    "pagv",
    "hot_serial_cereal",
)


def _pack_parser(source: str) -> WheatParser:
    def parse(directory: Path) -> tuple[SiteYear, ...]:
        return parse_management_pack(directory, source=source)

    parse.__name__ = f"parse_{source}"
    parse.__qualname__ = f"parse_{source}"
    return parse


WHEAT_PARSERS: Mapping[str, WheatParser] = {
    "westerfeld": parse_westerfeld,
    "muncheberg": parse_muncheberg,
    "broadbalk": parse_broadbalk,
    "braunschweig": parse_braunschweig,
    **{source: _pack_parser(source) for source in _MANAGEMENT_PACKS},
}


def cache_wheat_archive(cache_path: Path, source: str, files: Mapping[str, str]) -> Path:
    """Write archive tables into ``cache_path/datasets/<source>`` and return that directory."""
    if source not in WHEAT_PARSERS:
        raise KeyError(f"Unknown wheat archive {source!r}.")
    directory = Path(cache_path) / DATASETS_CACHE / source
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


def list_wheat_site_years(cache_path: Path) -> tuple[SiteYear, ...]:
    """Parse every cached wheat archive under ``cache_path/datasets``."""
    root = Path(cache_path) / DATASETS_CACHE
    if not root.is_dir():
        return ()
    years: list[SiteYear] = []
    for source, parser in WHEAT_PARSERS.items():
        directory = root / source
        if directory.is_dir():
            years.extend(parser(directory))
    return tuple(sorted(years, key=lambda site: site.name))
