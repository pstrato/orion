"""Shared requests-cache sessions under the settings cache folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
import requests_cache
from retry_requests import retry

_sessions: dict[str, requests.Session] = {}


def http_cache_sqlite_path(cache_path: Path | str) -> Path:
    """SQLite file for HTTP responses: ``<cache_path>/http.sqlite``."""
    return Path(cache_path) / "http.sqlite"


def clear_http_session_cache() -> None:
    """Drop pooled sessions (tests)."""
    _sessions.clear()


def cached_session(cache_path: Path | str) -> requests.Session:
    """Reuse one CachedSession (+ retry) per on-disk cache folder.

    ``Settings.cache_path`` is a directory (datasets live there too). The HTTP
    cache is the ``http.sqlite`` file inside that directory — not a sibling
    ``cache.sqlite`` next to the folder.
    """
    root = Path(cache_path)
    root.mkdir(parents=True, exist_ok=True)
    db = http_cache_sqlite_path(root)
    key = str(db.with_suffix(""))  # CachedSession appends ``.sqlite``
    session = _sessions.get(key)
    if session is None:
        cache_session = requests_cache.CachedSession(key, expire_after=-1)
        session = retry(cache_session, retries=5, backoff_factor=0.2)
        _sessions[key] = session
    return session


def response_was_cached(response: Any) -> bool:
    """True when requests-cache served ``response`` from disk."""
    return bool(getattr(response, "from_cache", False))
