"""Discoverable process inputs for the configuration builder.

Providers and optional inputs are classes that currently import. Modules still
mid-refactor are skipped.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeGuard

from orion.core.input import Input
from orion.ui.configuration import CLOCK, CROP_WHEAT, LIGHT_INTERCEPTION_BEER_LAMBERT, SOIL_SOILGRIDS, WEATHER_OPEN_METEO, Configuration
from orion.ui.reflect import apply_quantity_edits

ProcessInputFactory = Callable[[], Input]

_DOMAIN_ORDER = {"crop": 0, "management": 1, "soil": 2}
_HORIZON_INPUTS = frozenset({"ClockInput", "LocationInput"})
_ROLE_CLASS_NAMES: dict[str, dict[str, str]] = {
    "clock": {"ClockInput": CLOCK},
    "weather": {"OpenMeteoWeatherInput": WEATHER_OPEN_METEO},
    "soil": {"SoilGridInput": SOIL_SOILGRIDS},
    "crop": {"WheatInput": CROP_WHEAT},
    "light_interception": {"BeerLambertLightInterceptionInput": LIGHT_INTERCEPTION_BEER_LAMBERT},
}
_PROVIDER_CLASS_NAMES = {name for role in _ROLE_CLASS_NAMES.values() for name in role}
_PROVIDER_PACKAGES = ("orion.clients", "orion.processes")


@dataclass(frozen=True)
class ProcessInputSpec:
    """One process input discovered from ``orion.processes`` or ``orion.clients``."""

    label: str
    domain: str
    implemented: bool
    factory: ProcessInputFactory
    input_type: type[Input]


def _iter_modules(package_name: str) -> list[str]:
    try:
        package = importlib.import_module(package_name)
    except Exception:  # noqa: BLE001 — package itself may be mid-refactor
        return []
    names = [package_name]
    if hasattr(package, "__path__"):
        for module in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
            names.append(module.name)
    return names


def _label_for(cls: type) -> str:
    stem = cls.__name__.removesuffix("Input")
    parts = re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", stem)
    if not parts:
        return stem
    return parts[0] + "".join(f" {part.lower()}" for part in parts[1:])


def _domain_for(cls: type) -> str:
    parts = cls.__module__.split(".")
    if len(parts) > 2 and parts[0] == "orion" and parts[1] == "processes":
        return parts[2]
    if len(parts) > 1:
        return parts[1]
    return "other"


def _is_implemented_process(cls: type) -> bool:
    step = getattr(cls, "step", None)
    if step is None or not callable(step):
        return False
    try:
        source = inspect.getsource(step)
    except OSError, TypeError:
        return True
    return "raise NotImplementedError" not in source


def _process_class_for_input(input_cls: type[Input]) -> type | None:
    module = importlib.import_module(input_cls.__module__)
    name = input_cls.__name__.removesuffix("Input") + "Process"
    obj = getattr(module, name, None)
    if isinstance(obj, type) and obj.__name__.endswith("Process"):
        return obj
    return None


def _clock_input(name: str) -> Input:
    from datetime import date

    from orion.core.constant import const
    from orion.processes.clock import ClockInput

    return ClockInput(
        name=name,
        start=const("start", "isodate", date(2024, 1, 1), "Simulation start date"),
        end=const("end", "isodate", date(2024, 1, 11), "Simulation end date"),
        delta=const("delta", "hours", 3, "Simulation delta step in hours"),
    )


def _input_factory(cls: type[Input], name: str) -> ProcessInputFactory:
    def factory() -> Input:
        if cls.__name__ == "ClockInput":
            return _clock_input(name)
        return cls(name)

    return factory


def _is_input_class(cls: object) -> TypeGuard[type[Input]]:
    return isinstance(cls, type) and issubclass(cls, Input) and cls is not Input


def discover_process_inputs() -> tuple[ProcessInputSpec, ...]:
    """Find optional process Input classes under ``orion.processes``.

    Clock and location are the horizon. Weather, soil, crop, and light interception
    are provider slots. Modules that fail to import are skipped.
    """
    found: dict[str, ProcessInputSpec] = {}
    for module_name in _iter_modules("orion.processes"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 — skip broken optional imports
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module_name or not _is_input_class(obj):
                continue
            if obj.__name__ in _HORIZON_INPUTS or obj.__name__ in _PROVIDER_CLASS_NAMES:
                continue
            if not obj.__module__.startswith("orion.processes."):
                continue
            label = _label_for(obj)
            if label in found:
                continue
            process_cls = _process_class_for_input(obj)
            found[label] = ProcessInputSpec(
                label=label,
                domain=_domain_for(obj),
                implemented=process_cls is not None and _is_implemented_process(process_cls),
                factory=_input_factory(obj, label.lower()),
                input_type=obj,
            )
    return tuple(sorted(found.values(), key=lambda spec: (_DOMAIN_ORDER.get(spec.domain, 9), spec.label.lower())))


def process_input_labels() -> list[str]:
    return [spec.label for spec in discover_process_inputs()]


def catalog_select_options() -> dict[str, str]:
    options: dict[str, str] = {}
    for spec in discover_process_inputs():
        badge = "" if spec.implemented else " (abstract)"
        options[spec.label] = f"{spec.domain} · {spec.label}{badge}"
    return options


def make_process_input(label: str) -> Input:
    for spec in discover_process_inputs():
        if spec.label == label:
            return spec.factory()
    raise KeyError(label)


def spec_for_input(process_input: Input) -> ProcessInputSpec | None:
    for spec in discover_process_inputs():
        if isinstance(process_input, spec.input_type):
            return spec
    return None


def unimplemented_process_labels(processes: Sequence[Input]) -> tuple[str, ...]:
    labels: list[str] = []
    for process_input in processes:
        spec = spec_for_input(process_input)
        if spec is not None and not spec.implemented:
            labels.append(spec.label)
    return tuple(labels)


def discover_provider_inputs(role: str) -> tuple[ProcessInputSpec, ...]:
    """Find concrete implementations for one provider role."""
    names = _ROLE_CLASS_NAMES.get(role, {})
    found: dict[str, ProcessInputSpec] = {}
    for package_name in _PROVIDER_PACKAGES:
        for module_name in _iter_modules(package_name):
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001 — skip broken optional imports
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module_name or not _is_input_class(obj):
                    continue
                label = names.get(obj.__name__)
                if label is None or label in found:
                    continue
                found[label] = ProcessInputSpec(
                    label=label,
                    domain=role,
                    implemented=True,
                    factory=_input_factory(obj, label.lower()),
                    input_type=obj,
                )
    return tuple(sorted(found.values(), key=lambda spec: spec.label.lower()))


def weather_provider_options() -> dict[str, str]:
    return {spec.label: spec.label for spec in discover_provider_inputs("weather")}


def soil_provider_options() -> dict[str, str]:
    return {spec.label: spec.label for spec in discover_provider_inputs("soil")}


def crop_provider_options() -> dict[str, str]:
    return {spec.label: spec.label for spec in discover_provider_inputs("crop")}


def light_interception_provider_options() -> dict[str, str]:
    return {spec.label: spec.label for spec in discover_provider_inputs("light_interception")}


def implementation_options(role: str) -> dict[str, str]:
    """Implementation labels for a provider slot."""
    return {spec.label: spec.label for spec in discover_provider_inputs(role)}


def make_provider(label: str, role: str) -> Input:
    """Instantiate the provider implementation named ``label`` for ``role``."""
    for spec in discover_provider_inputs(role):
        if spec.label == label:
            return spec.factory()
    raise KeyError(label)


def make_clock_input(label: str) -> Input:
    return make_provider(label, "clock")


def make_weather_input(label: str) -> Input:
    return make_provider(label, "weather")


def make_soil_input(label: str) -> Input:
    return make_provider(label, "soil")


def make_crop_input(label: str) -> Input:
    return make_provider(label, "crop")


def make_light_interception_input(label: str) -> Input:
    return make_provider(label, "light_interception")


def configured_clock_input(configuration: Configuration) -> Input:
    """Clock implementation with this configuration's quantity edits applied."""
    return apply_quantity_edits(make_clock_input(configuration.clock), configuration.parameter_edits, "clock")


def configured_weather_input(configuration: Configuration) -> Input:
    """Weather provider with this configuration's quantity edits applied."""
    return apply_quantity_edits(make_weather_input(configuration.weather), configuration.parameter_edits, "weather")


def configured_soil_input(configuration: Configuration) -> Input:
    """Soil provider with this configuration's quantity edits applied."""
    return apply_quantity_edits(make_soil_input(configuration.soil), configuration.parameter_edits, "soil")


def configured_crop_input(configuration: Configuration) -> Input:
    """Crop provider with this configuration's quantity edits applied."""
    return apply_quantity_edits(make_crop_input(configuration.crop), configuration.parameter_edits, "crop")


def configured_light_interception_input(configuration: Configuration) -> Input:
    """Light-interception provider with this configuration's quantity edits applied."""
    return apply_quantity_edits(make_light_interception_input(configuration.light_interception), configuration.parameter_edits, "light_interception")


def configured_process_inputs(configuration: Configuration) -> tuple[Input, ...]:
    """Optional process inputs with this configuration's quantity edits applied."""
    return tuple(apply_quantity_edits(process, configuration.parameter_edits, process.name) for process in configuration.processes)
