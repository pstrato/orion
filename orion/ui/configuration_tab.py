"""Configurations tab: list on the left; process inputs from Inputs in the centre."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nicegui import ui

from orion.core.input import Input
from orion.ui.catalog import catalog_select_options, implementation_options, make_process_input, make_provider, spec_for_input
from orion.ui.configuration import Configuration, default_configuration, next_configuration_color, provider_fields
from orion.ui.entity_editor import render_entity_editor
from orion.ui.reflect import apply_quantity_edits
from orion.ui.theme import setting_row

CONFIG_CREATE_HINT = "Defaults is readonly. Add a configuration to edit process inputs."


@dataclass(frozen=True)
class ProcessInputSlot:
    """One process-input row: a mandatory Inputs field or an optional process."""

    key: str
    label: str
    implementation: str
    base: type[Input] | None = None
    instance: Input | None = None


def clamp_configuration_index(configurations: Sequence[Configuration], index: int) -> int:
    """Keep the selected row inside the current list."""
    if not configurations:
        return 0
    return min(max(index, 0), len(configurations) - 1)


def process_input_slots(config: Configuration) -> tuple[ProcessInputSlot, ...]:
    """List provider roles, then optional configuration processes."""
    slots = [_provider_slot(config, name, label) for name, label in provider_fields()]
    taken = {slot.key for slot in slots}
    for process_input in config.processes:
        spec = spec_for_input(process_input)
        label = spec.label if spec is not None else _label_from_class(type(process_input))
        key = process_input.name if process_input.name not in taken else f"optional:{process_input.name}"
        taken.add(key)
        slots.append(ProcessInputSlot(key=key, label=label, implementation=label, instance=process_input))
    return tuple(slots)


def clamp_input_key(config: Configuration, key: str) -> str:
    """Keep the selected process-input key inside the current configuration."""
    keys = [slot.key for slot in process_input_slots(config)]
    if key in keys:
        return key
    return keys[0] if keys else ""


def add_configuration(configurations: list[Configuration]) -> tuple[list[Configuration], int]:
    """Append an editable configuration and select it."""
    color = next_configuration_color(tuple(configurations))
    name = f"configuration {len(configurations)}"
    baseline = default_configuration()
    created = Configuration(
        name=name,
        enabled=True,
        color=color,
        processes=(),
        readonly=False,
        clock=baseline.clock,
        weather=baseline.weather,
        soil=baseline.soil,
        crop=baseline.crop,
        light_interception=baseline.light_interception,
        day_length=baseline.day_length,
    )
    next_list = [*configurations, created]
    return next_list, len(next_list) - 1


def render_configuration_tab(
    host,
    configurations: list[Configuration],
    selected_index: int,
    on_change: Callable[[list[Configuration], int, str], None],
    selected_input_key: str = "",
) -> None:
    """Left: configuration list and +. Centre: process-input list, then the selected editor."""
    host.clear()
    index = clamp_configuration_index(configurations, selected_index)
    selected = configurations[index] if configurations else None
    key = clamp_input_key(selected, selected_input_key) if selected is not None else selected_input_key
    if configurations and (index != selected_index or (selected is not None and key != selected_input_key)):
        on_change(configurations, index, key)
        return

    def persist(next_configs: list[Configuration], next_index: int, next_key: str = key) -> None:
        on_change(next_configs, clamp_configuration_index(next_configs, next_index), next_key)

    def replace_selected(updated: Configuration) -> None:
        next_configs = list(configurations)
        next_configs[index] = updated
        persist(next_configs, index, key)

    with host:
        with ui.row().classes("w-full items-start gap-3 no-wrap"):
            _render_configuration_list(configurations, index, key, persist)
            with ui.column().classes("flex-1 min-w-0 gap-2"):
                if selected is None:
                    ui.label("Create a configuration to edit process inputs.").classes("text-xs text-gray-500")
                    return
                _render_configuration_detail(selected, index, configurations, key, replace_selected, persist)


def _render_configuration_list(
    configurations: list[Configuration],
    index: int,
    key: str,
    persist: Callable[[list[Configuration], int, str], None],
) -> None:
    with ui.column().classes("orion-config-list w-56 min-w-44 shrink-0 gap-1"):
        with ui.row().classes("w-full items-center gap-1"):
            ui.label("Configurations").classes("text-subtitle2")
            ui.button(icon="add", on_click=lambda: persist(*add_configuration(list(configurations)), key)).props("flat dense round").tooltip(CONFIG_CREATE_HINT)
        for row_index, config in enumerate(configurations):
            selected_cls = " orion-config-list-item--selected" if row_index == index else ""

            def make_select(i: int = row_index):
                def on_select() -> None:
                    persist(list(configurations), i, key)

                return on_select

            with ui.row().classes(f"orion-config-list-item items-center gap-2 no-wrap cursor-pointer w-full{selected_cls}").on("click", make_select()):
                ui.element("div").style(f"width:10px;height:10px;border-radius:50%;background:{config.color};flex-shrink:0")
                ui.label(config.name).classes("text-xs truncate")
                if not config.enabled:
                    ui.label("off").classes("text-xs text-gray-400")


def _provider_slot(config: Configuration, name: str, label: str) -> ProcessInputSlot:
    return ProcessInputSlot(key=name, label=label, implementation=str(getattr(config, name)))


def _label_from_class(cls: type) -> str:
    stem = cls.__name__.removesuffix("Input")
    return stem[:1].upper() + stem[1:] if stem else cls.__name__


def _render_configuration_detail(
    config: Configuration,
    index: int,
    configurations: list[Configuration],
    selected_input_key: str,
    replace_selected: Callable[[Configuration], None],
    persist: Callable[[list[Configuration], int, str], None],
) -> None:
    _render_detail_header(config, replace_selected)
    slots = process_input_slots(config)
    _render_slot_list(slots, selected_input_key, lambda next_key: persist(list(configurations), index, next_key))
    selected_slot = next((slot for slot in slots if slot.key == selected_input_key), None)
    if selected_slot is not None:
        _render_selected_editor(config, selected_slot, replace_selected)
    _render_process_actions(config, index, configurations, replace_selected, persist)


def _render_detail_header(config: Configuration, replace_selected: Callable[[Configuration], None]) -> None:
    with ui.row().classes("w-full items-center gap-2"):
        if config.readonly:
            ui.label(config.name).classes("text-subtitle1 font-medium")
            ui.badge("readonly").props("outline dense")
        else:
            name_input = ui.input(value=config.name).props("dense").classes("w-56")

            def on_name(_) -> None:
                next_name = str(name_input.value or "").strip() or config.name
                if next_name != config.name:
                    replace_selected(config.with_name(next_name))

            name_input.on_value_change(on_name)
        ui.space()
        enabled = ui.switch("Enabled", value=config.enabled).props("dense")

        def on_enabled(_) -> None:
            next_enabled = bool(enabled.value)
            if next_enabled != config.enabled:
                replace_selected(config.with_enabled(next_enabled))

        enabled.on_value_change(on_enabled)


def _render_slot_list(slots: Sequence[ProcessInputSlot], selected_key: str, on_select: Callable[[str], None]) -> None:
    ui.label("Process inputs").classes("text-subtitle2")
    for slot in slots:
        selected_cls = " orion-input-slot--selected" if slot.key == selected_key else ""

        def make_select(key: str = slot.key) -> Callable[[], None]:
            def select() -> None:
                if key != selected_key:
                    on_select(key)

            return select

        with ui.row().classes(f"orion-input-slot items-center gap-2 no-wrap cursor-pointer w-full{selected_cls}").on("click", make_select()).tooltip("Edit this process input"):
            ui.label(slot.label).classes("text-xs w-40 shrink-0")
            ui.label(slot.implementation).classes("text-xs text-gray-600 truncate")


def _render_selected_editor(config: Configuration, slot: ProcessInputSlot, replace_selected: Callable[[Configuration], None]) -> None:
    provider_keys = {name for name, _label in provider_fields()}
    if slot.key in provider_keys:
        options = implementation_options(slot.key)
        if slot.implementation not in options:
            options = {slot.implementation: slot.implementation, **options}
        with setting_row("Implementation"):
            box = ui.select(options, value=slot.implementation).props("dense options-dense")

        def on_implementation(_) -> None:
            value = str(box.value)
            if value != slot.implementation:
                replace_selected(_with_implementation(config, slot.key, value))

        box.on_value_change(on_implementation)
    inp = _slot_input(config, slot)

    def on_quantity(name: str, value: float, role: str = slot.key) -> None:
        replace_selected(config.with_parameter(role, name, value))

    render_entity_editor(inp, on_quantity)


def _with_implementation(config: Configuration, key: str, label: str) -> Configuration:
    if key not in {name for name, _label in provider_fields()}:
        raise KeyError(key)
    return config.with_providers(**{key: label})


def _slot_input(config: Configuration, slot: ProcessInputSlot) -> Input:
    if slot.key in {name for name, _label in provider_fields()}:
        try:
            created = make_provider(slot.implementation, slot.key)
        except KeyError:
            created = Input(slot.implementation)
        return apply_quantity_edits(created, config.parameter_edits, slot.key)
    if slot.instance is None:
        raise KeyError(slot.key)
    return apply_quantity_edits(slot.instance, config.parameter_edits, slot.key)


def _render_process_actions(
    config: Configuration,
    index: int,
    configurations: list[Configuration],
    replace_selected: Callable[[Configuration], None],
    persist: Callable[[list[Configuration], int, str], None],
) -> None:
    if config.readonly:
        return
    add_select = ui.select(catalog_select_options(), label="Add process input").classes("min-w-64").props("dense")

    def on_add() -> None:
        label = add_select.value
        if not label:
            return
        created = make_process_input(str(label))
        next_configs = list(configurations)
        next_configs[index] = config.with_processes((*config.processes, created))
        persist(next_configs, index, created.name)

    def on_clear() -> None:
        replace_selected(config.with_processes(()))

    def on_remove() -> None:
        next_configs = [item for i, item in enumerate(configurations) if i != index]
        persist(next_configs, min(index, max(len(next_configs) - 1, 0)), "")

    with ui.row().classes("gap-2 flex-wrap"):
        ui.button("Add", on_click=on_add).props("dense")
        ui.button("Clear processes", on_click=on_clear).props("flat dense")
        ui.button("Remove", on_click=on_remove).props("flat dense color=negative")
