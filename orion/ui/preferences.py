"""Persist UI preferences in the platform user config directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_config_dir

from orion.ui.units import resolve_unit_alternatives


@dataclass
class UiPreferences:
    """User preferences for the Orion UI."""

    plot_order: list[str] = field(default_factory=list)
    """Ordered plot keys (state.variable); earlier entries appear first."""

    unit_alternatives: dict[str, str] = field(default_factory=dict)
    """Canonical SI unit → chosen display alternative (e.g. ``kg/kg`` → ``%``)."""


def preferences_dir() -> Path:
    """Standard per-user config directory for Orion."""
    return Path(user_config_dir("orion", appauthor=False))


def preferences_path() -> Path:
    return preferences_dir() / "ui_preferences.json"


def load_preferences(path: Path | None = None) -> UiPreferences:
    """Load preferences from disk; missing or invalid files yield defaults."""
    target = path or preferences_path()
    if not target.is_file():
        return UiPreferences()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return UiPreferences()
    order = raw.get("plot_order", [])
    if not isinstance(order, list):
        return UiPreferences()
    alternatives_raw = raw.get("unit_alternatives", {})
    if not isinstance(alternatives_raw, dict):
        alternatives_raw = {}
    return UiPreferences(
        plot_order=[str(item) for item in order],
        unit_alternatives=resolve_unit_alternatives({str(k): str(v) for k, v in alternatives_raw.items()}),
    )


def save_preferences(prefs: UiPreferences, path: Path | None = None) -> Path:
    """Write preferences atomically to the config path; return the path used."""
    target = path or preferences_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(prefs)
    payload["unit_alternatives"] = resolve_unit_alternatives(prefs.unit_alternatives)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def merge_plot_order(preferred: list[str], available: list[str]) -> list[str]:
    """Keep saved order for known keys, append any new keys at the end."""
    seen: set[str] = set()
    ordered: list[str] = []
    available_set = set(available)
    for key in preferred:
        if key in available_set and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in available:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def move_plot_key(order: list[str], old_index: int, new_index: int) -> list[str]:
    """Return a new order after moving the item at ``old_index`` to ``new_index``."""
    if old_index == new_index:
        return list(order)
    if not (0 <= old_index < len(order) and 0 <= new_index < len(order)):
        return list(order)
    items = list(order)
    key = items.pop(old_index)
    items.insert(new_index, key)
    return items
