"""Keys for warm model reuse in the Simulate path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from orion.core.model import Model
from orion.ui.configuration import Configuration
from orion.ui.reflect import run_steps
from orion.ui.runs import INPUT_DATASETS, SimulationRun


def configuration_compile_key(configuration: Configuration) -> tuple[object, ...]:
    """Identity of a configuration's runnable inputs."""
    return (
        configuration.clock,
        configuration.weather,
        configuration.soil,
        configuration.crop,
        configuration.light_interception,
        configuration.parameter_edits,
        tuple((process.name, type(process).__name__) for process in configuration.processes),
    )


def warm_model_key(state: Any, run: SimulationRun) -> tuple[object, ...]:
    """Identity of everything that forces ``build_model`` to run again."""
    site = run.site_year
    if state.input_mode == INPUT_DATASETS and site is not None:
        start, end = site.horizon()
        site_key: object = (site.name, start.isoformat(), end.isoformat())
    else:
        site_key = (
            float(state.latitude),
            float(state.longitude),
            state.location_name,
            state.start.isoformat(),
            state.end.isoformat(),
        )
    return (
        run.name,
        configuration_compile_key(run.configuration),
        str(state.cache_path),
        int(state.step_hours),
        state.input_mode,
        site_key,
    )


@dataclass
class WarmModelCache:
    """Reuse a built model while its warm key is unchanged."""

    models: dict[tuple[object, ...], Model] = field(default_factory=dict)

    def clear(self) -> None:
        self.models.clear()

    def resolve(self, key: tuple[object, ...], factory: Callable[[], Model]) -> Model:
        cached = self.models.get(key)
        if cached is None:
            cached = factory()
            self.models[key] = cached
        return cached


@dataclass
class ConfigurationCompileCache:
    """Track configuration identity and step models for the UI."""

    fingerprint: tuple[object, ...] | None = None

    def sync(self, configurations: tuple[Configuration, ...]) -> None:
        self.fingerprint = tuple(configuration_compile_key(configuration) + (configuration.enabled, configuration.name) for configuration in configurations)

    def simulate(self, built: Model, steps: int) -> tuple[Model, Model]:
        return run_steps(built, steps)
