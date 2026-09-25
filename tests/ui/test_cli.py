"""Behaviour: orion-ui CLI restarts by freeing the prior pid/port."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion.ui import cli


def test_write_and_read_pid_round_trip(tmp_path: Path):
    path = tmp_path / "orion-ui.pid"
    cli.write_pid(path)
    assert cli.read_pid(path) is not None
    assert cli.read_pid(path) == int(path.read_text(encoding="utf-8"))


def test_stop_existing_removes_pidfile_and_terminates_recorded_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "orion-ui.pid"
    path.write_text("4242", encoding="utf-8")
    terminated: list[int] = []

    monkeypatch.setattr(cli, "_pid_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(cli, "_terminate_pid", lambda pid: terminated.append(pid))
    monkeypatch.setattr(cli, "pids_listening_on_port", lambda port: [])
    monkeypatch.setattr(cli, "port_is_free", lambda host, port: True)

    stopped = cli.stop_existing(port=8080, path=path)
    assert stopped == [4242]
    assert terminated == [4242]
    assert not path.exists()


def test_stop_existing_also_stops_port_listeners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "orion-ui.pid"
    terminated: list[int] = []
    monkeypatch.setattr(cli, "read_pid", lambda path=None: None)
    monkeypatch.setattr(cli, "pids_listening_on_port", lambda port: [111, 222])
    monkeypatch.setattr(cli, "_terminate_pid", lambda pid: terminated.append(pid))
    monkeypatch.setattr(cli, "port_is_free", lambda host, port: True)

    stopped = cli.stop_existing(port=8080, path=path)
    assert stopped == [111, 222]
    assert terminated == [111, 222]


def test_main_restart_then_runs(monkeypatch: pytest.MonkeyPatch):
    stopped: dict[str, object] = {}
    ran: dict[str, object] = {}

    def fake_stop(**kwargs):
        stopped.update(kwargs)
        return [9]

    monkeypatch.setattr(cli, "stop_existing", fake_stop)
    monkeypatch.setattr(cli, "write_pid", lambda: Path("pid"))

    import orion.ui.app as app_mod

    monkeypatch.setattr(app_mod, "run", lambda **kwargs: ran.update(kwargs))
    cli.main(["--host", "127.0.0.1", "--port", "8099"])
    assert stopped["port"] == 8099
    assert ran == {"host": "127.0.0.1", "port": 8099, "reload": False}


def test_main_no_restart_skips_stop(monkeypatch: pytest.MonkeyPatch):
    called = {"stop": False}

    def mark_stop(**kwargs):
        called["stop"] = True
        return []

    monkeypatch.setattr(cli, "stop_existing", mark_stop)
    monkeypatch.setattr(cli, "write_pid", lambda: Path("pid"))
    import orion.ui.app as app_mod

    monkeypatch.setattr(app_mod, "run", lambda **kwargs: None)
    cli.main(["--no-restart"])
    assert called["stop"] is False
