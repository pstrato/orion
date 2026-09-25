"""Behaviour: client HTTP cache is a SQLite file inside the settings cache folder."""

from __future__ import annotations

from pathlib import Path

from orion.clients.http_cache import cached_session, clear_http_session_cache, http_cache_sqlite_path


def test_http_cache_sqlite_path_is_inside_cache_folder(tmp_path: Path):
    assert http_cache_sqlite_path(tmp_path) == tmp_path / "http.sqlite"


def test_cached_session_creates_sqlite_under_cache_folder(tmp_path: Path):
    clear_http_session_cache()
    cached_session(tmp_path)
    assert http_cache_sqlite_path(tmp_path).is_file()


def test_cached_session_reuses_same_sqlite_for_same_folder(tmp_path: Path):
    clear_http_session_cache()
    first = cached_session(tmp_path)
    second = cached_session(tmp_path)
    assert first is second
    assert http_cache_sqlite_path(tmp_path).is_file()
