"""Behaviour: UI status uses the latest client fetch progress event."""

from __future__ import annotations

from queue import SimpleQueue

from orion.ui.fetch_status import FetchProgress, latest_fetch_progress


def test_latest_fetch_progress_returns_the_most_recent_event():
    queue: SimpleQueue[FetchProgress] = SimpleQueue()
    assert latest_fetch_progress(queue) is None
    queue.put(FetchProgress(source="openmeteo", message="Fetching Open-Meteo 1/2", current=1, total=2))
    queue.put(FetchProgress(source="soilgrids", message="Fetching SoilGrids 2/2", current=2, total=2))
    latest = latest_fetch_progress(queue)
    assert latest is not None
    assert latest.message == "Fetching SoilGrids 2/2"
    assert latest.fraction == 1.0
    assert latest_fetch_progress(queue) is None


def test_discard_fetch_progress_clears_queue_without_returning_events():
    from orion.ui.fetch_status import discard_fetch_progress

    queue: SimpleQueue[FetchProgress] = SimpleQueue()
    queue.put(FetchProgress(source="openmeteo", message="Fetching Open-Meteo", current=1, total=1))
    discard_fetch_progress(queue)
    assert latest_fetch_progress(queue) is None
