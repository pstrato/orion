"""Drain client fetch-progress events onto the Simulation status line."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Callable


@dataclass(frozen=True)
class FetchProgress:
    """One client fetch update for the Simulation status line."""

    source: str
    message: str
    current: int
    total: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.current / self.total


@contextmanager
def fetch_progress(callback: Callable[[FetchProgress], None]):
    """Placeholder scope until core reports fetch events into the UI queue."""
    del callback
    yield


def latest_fetch_progress(queue: SimpleQueue[FetchProgress]) -> FetchProgress | None:
    """Return the newest queued event, discarding intermediates."""
    latest: FetchProgress | None = None
    while True:
        try:
            latest = queue.get_nowait()
        except Empty:
            return latest


def discard_fetch_progress(queue: SimpleQueue[FetchProgress]) -> None:
    """Drop queued events without applying them (keeps Done from being overwritten)."""
    while True:
        try:
            queue.get_nowait()
        except Empty:
            return
