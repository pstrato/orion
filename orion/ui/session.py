"""Persist UI session choices. Simulation results and runs are not stored."""

from __future__ import annotations

import importlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from orion.core.input import Input
from orion.ui.configuration import Configuration, provider_fields
from orion.ui.preferences import UiPreferences, preferences_dir
from orion.ui.reflect import list_entity_values, set_entity_value
from orion.ui.runs import INPUT_CUSTOM, INPUT_DATASETS
from orion.ui.units import resolve_unit_alternatives

SESSION_NAME = "ui_session.json"


def session_path() -> Path:
    """Per-user file for UI choices."""
    return preferences_dir() / SESSION_NAME


def session_document(state: Any) -> dict[str, Any]:
    """JSON-ready UI choices. Results, runs, and model caches are omitted."""
    return {
        "settings": {item.path: _json_value(item.value) for item in list_entity_values(state.settings)},
        "latitude": float(state.latitude),
        "longitude": float(state.longitude),
        "location_name": str(state.location_name),
        "start": state.start.isoformat(),
        "end": state.end.isoformat(),
        "step_hours": int(state.step_hours),
        "input_mode": state.input_mode if state.input_mode in {INPUT_DATASETS, INPUT_CUSTOM} else INPUT_DATASETS,
        "selected_dataset_names": [str(name) for name in state.selected_dataset_names],
        "configurations": [_configuration_document(config) for config in state.configurations],
        "selected_configuration_index": int(state.selected_configuration_index),
        "selected_input_key": str(state.selected_input_key),
        "prefs": {
            "plot_order": [str(key) for key in state.prefs.plot_order],
            "unit_alternatives": {str(key): str(value) for key, value in state.prefs.unit_alternatives.items()},
        },
    }


def save_session(state: Any, path: Path | None = None) -> Path:
    """Write UI choices atomically. Identical contents are left untouched."""
    target = path or session_path()
    text = json.dumps(session_document(state), indent=2)
    if getattr(state, "_session_text", None) == text and target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    object.__setattr__(state, "_session_text", text)
    return target


def load_session(path: Path | None = None) -> Any:
    """Restore UI choices. A missing or unreadable file yields a fresh session."""
    from orion.ui.app import AppState

    state = AppState()
    target = path or session_path()
    if not target.is_file():
        return state
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return state
    if not isinstance(raw, dict):
        return state
    _apply_session(state, raw)
    object.__setattr__(state, "_session_text", json.dumps(session_document(state), indent=2))
    return state


def _apply_session(state: Any, raw: dict[str, Any]) -> None:
    _apply_settings(state, raw.get("settings"))
    _apply_place(state, raw)
    _apply_choices(state, raw)
    _apply_prefs(state, raw.get("prefs"))


def _apply_settings(state: Any, stored: object) -> None:
    from orion.ui.app import ui_settings

    settings = ui_settings()
    if isinstance(stored, dict):
        for path, value in stored.items():
            try:
                settings = set_entity_value(settings, str(path), value)
            except KeyError, TypeError, ValueError:
                continue
    state.settings = settings


def _apply_place(state: Any, raw: dict[str, Any]) -> None:
    if "latitude" in raw:
        state.latitude = float(raw["latitude"])
    if "longitude" in raw:
        state.longitude = float(raw["longitude"])
    if "location_name" in raw:
        state.location_name = str(raw["location_name"])
    if "start" in raw:
        state.start = date.fromisoformat(str(raw["start"]))
    if "end" in raw:
        state.end = date.fromisoformat(str(raw["end"]))
    if "step_hours" in raw:
        state.step_hours = int(raw["step_hours"])


def _apply_choices(state: Any, raw: dict[str, Any]) -> None:
    if raw.get("input_mode") in {INPUT_DATASETS, INPUT_CUSTOM}:
        state.input_mode = str(raw["input_mode"])
    names = raw.get("selected_dataset_names")
    if isinstance(names, list):
        state.selected_dataset_names = [str(name) for name in names]
    configs = raw.get("configurations")
    if isinstance(configs, list):
        loaded = tuple(_configuration_from(item) for item in configs)
        state.configurations = [item for item in loaded if item is not None] or state.configurations
    if "selected_configuration_index" in raw:
        state.selected_configuration_index = int(raw["selected_configuration_index"])
    if "selected_input_key" in raw:
        state.selected_input_key = str(raw["selected_input_key"])


def _apply_prefs(state: Any, prefs: object) -> None:
    if isinstance(prefs, dict):
        order = prefs.get("plot_order", [])
        alternatives = prefs.get("unit_alternatives", {})
        state.prefs = UiPreferences(
            plot_order=[str(item) for item in order] if isinstance(order, list) else [],
            unit_alternatives=resolve_unit_alternatives({str(key): str(value) for key, value in alternatives.items()}) if isinstance(alternatives, dict) else {},
        )


def _configuration_document(config: Configuration) -> dict[str, Any]:
    document: dict[str, Any] = {
        "name": config.name,
        "enabled": config.enabled,
        "color": config.color,
        "readonly": config.readonly,
        "processes": [_input_document(process) for process in config.processes],
        "parameter_edits": [[role, name, value] for role, name, value in config.parameter_edits],
    }
    for key, _label in provider_fields():
        document[key] = str(getattr(config, key))
    return document


def _configuration_from(raw: object) -> Configuration | None:
    if not isinstance(raw, dict) or "name" not in raw:
        return None
    providers = {key: str(raw[key]) for key, _label in provider_fields() if key in raw}
    processes = tuple(item for item in (_input_from(entry) for entry in raw.get("processes", [])) if item is not None)
    edits: list[tuple[str, str, float]] = []
    for edit in raw.get("parameter_edits", []):
        if isinstance(edit, (list, tuple)) and len(edit) == 3:
            edits.append((str(edit[0]), str(edit[1]), float(edit[2])))
    return Configuration(
        name=str(raw["name"]),
        enabled=bool(raw.get("enabled", True)),
        color=str(raw.get("color", "#3D7FBF")),
        processes=processes,
        readonly=bool(raw.get("readonly", False)),
        parameter_edits=tuple(edits),
        **providers,  # type: ignore[arg-type]
    )


def _input_document(process: Input) -> dict[str, str]:
    return {"module": type(process).__module__, "type": type(process).__qualname__, "name": process.name}


def _input_from(raw: object) -> Input | None:
    if not isinstance(raw, dict):
        return None
    module_name = raw.get("module")
    type_name = raw.get("type")
    if not isinstance(module_name, str) or not isinstance(type_name, str):
        return None
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, type_name.split(".")[-1])
        if not isinstance(cls, type) or not issubclass(cls, Input) or cls is Input:
            return None
        return cls(str(raw.get("name", "")))
    except Exception:  # noqa: BLE001 — a removed input class should not block the rest of the session
        return None


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value
