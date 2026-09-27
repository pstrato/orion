"""Behaviour: UI choices are restored from disk and simulation results are not."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from orion.clients.soilgrids import SoilGridInput
from orion.ui.app import AppState
from orion.ui.configuration import Configuration
from orion.ui.preferences import UiPreferences
from orion.ui.session import load_session, save_session, session_document


def test_session_restores_settings_datasets_and_configurations(tmp_path: Path):
    from orion.ui.app import ui_settings

    state = AppState(
        settings=ui_settings(cache_path=tmp_path / "cache", validate_inputs=True, use_gpu=True),
        latitude=48.2,
        longitude=2.3,
        location_name="plot",
        start=date(2021, 3, 1),
        end=date(2021, 8, 1),
        step_hours=6,
        input_mode="custom",
        selected_dataset_names=["year-a", "year-b"],
        selected_configuration_index=0,
        selected_input_key="soil",
        prefs=UiPreferences(plot_order=["weather.Ts"], unit_alternatives={"kg/kg": "g/kg"}),
    )
    state.configurations = [
        Configuration(
            name="trial",
            enabled=False,
            color="#112233",
            processes=(SoilGridInput("extra-soil"),),
            readonly=False,
            weather="Open-Meteo",
            parameter_edits=(("clock", "delta", 12.0),),
        )
    ]
    state.results = {"finished": (None, None)}  # type: ignore[dict-item]
    state.runs = [object()]  # type: ignore[list-item]

    path = tmp_path / "ui_session.json"
    save_session(state, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "results" not in raw
    assert "runs" not in raw

    loaded = load_session(path)
    assert loaded.results == {}
    assert loaded.runs == []
    assert loaded.latitude == 48.2
    assert loaded.longitude == 2.3
    assert loaded.location_name == "plot"
    assert loaded.start == date(2021, 3, 1)
    assert loaded.end == date(2021, 8, 1)
    assert loaded.step_hours == 6
    assert loaded.input_mode == "custom"
    assert loaded.selected_dataset_names == ["year-a", "year-b"]
    assert loaded.selected_input_key == "soil"
    assert loaded.settings.validate_inputs is True
    assert loaded.settings.use_gpu is True
    assert loaded.settings.cache_path == tmp_path / "cache"
    assert loaded.prefs.plot_order == ["weather.Ts"]
    assert loaded.prefs.unit_alternatives["kg/kg"] == "g/kg"
    assert loaded.configurations[0].name == "trial"
    assert loaded.configurations[0].enabled is False
    assert loaded.configurations[0].parameter_edits == (("clock", "delta", 12.0),)
    assert isinstance(loaded.configurations[0].processes[0], SoilGridInput)
    assert loaded.configurations[0].processes[0].name == "extra-soil"


def test_changing_ui_state_saves_and_ignores_simulation_results(tmp_path: Path, monkeypatch):
    path = tmp_path / "ui_session.json"
    monkeypatch.setattr("orion.ui.session.session_path", lambda: path)
    state = AppState(selected_dataset_names=["kept"])
    object.__setattr__(state, "_autosave", True)

    state.selected_dataset_names = ["kept", "added"]
    state.results = {"finished": (None, None)}  # type: ignore[dict-item]

    loaded = load_session()
    assert loaded.selected_dataset_names == ["kept", "added"]
    assert loaded.results == {}
    assert "results" not in session_document(state)


def test_missing_session_file_loads_a_fresh_ui(tmp_path: Path):
    loaded = load_session(tmp_path / "missing.json")
    assert loaded.selected_dataset_names == []
    assert loaded.results == {}
    assert loaded.configurations[0].name == "defaults"
