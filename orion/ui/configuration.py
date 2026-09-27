"""Named process-input bundles compared in multi-run simulation."""

from __future__ import annotations

from orion.core.entity import Entity, entity
from orion.core.input import Input

DEFAULT_CONFIGURATION_COLORS: tuple[str, ...] = (
    "#3D7FBF",
    "#5BB8D4",
    "#2B6CB0",
    "#C45C26",
    "#6B46C1",
    "#2A9D8F",
    "#D65A7A",
    "#E9A825",
)

CLOCK = "Clock"
WEATHER_OPEN_METEO = "Open-Meteo"
SOIL_SOILGRIDS = "SoilGrids"
CROP_WHEAT = "Wheat"
LIGHT_INTERCEPTION_BEER_LAMBERT = "Beer-Lambert"
DAY_LENGTH_ASTRONOMICAL = "Astronomical"
DAY_LENGTH_APPARENT = "Apparent"


PROVIDER_FIELDS: tuple[tuple[str, str], ...] = (
    ("clock", "Clock"),
    ("weather", "Weather"),
    ("soil", "Soil"),
    ("crop", "Crop"),
    ("light_interception", "LightInterception"),
    ("day_length", "DayLength"),
)


def provider_fields() -> tuple[tuple[str, str], ...]:
    """Mandatory provider roles edited on a configuration."""
    return PROVIDER_FIELDS


@entity()
class Configuration(Entity):
    """A named set of clock, weather, soil, crop, light interception, day length, and optional process inputs."""

    enabled: bool = True
    """Whether this configuration participates in Simulate."""

    color: str = "#3D7FBF"
    """Plot colour for this configuration."""

    processes: tuple[Input, ...] = ()
    """Optional process inputs (clock, weather, soil, crop, light interception, and day length are chosen separately)."""

    readonly: bool = False
    """Readonly configurations (e.g. defaults) cannot change process inputs or be removed."""

    clock: str = CLOCK
    """Clock implementation label (e.g. Clock)."""

    weather: str = WEATHER_OPEN_METEO
    """Weather provider label (e.g. Open-Meteo)."""

    soil: str = SOIL_SOILGRIDS
    """Soil provider label (e.g. SoilGrids)."""

    crop: str = CROP_WHEAT
    """Crop / species provider label (e.g. Wheat)."""

    light_interception: str = LIGHT_INTERCEPTION_BEER_LAMBERT
    """Light-interception process provider label (e.g. Beer-Lambert)."""

    day_length: str = DAY_LENGTH_ASTRONOMICAL
    """Day-length process provider label (e.g. Astronomical)."""

    parameter_edits: tuple[tuple[str, str, float], ...] = ()
    """Overrides of numeric Input fields as (role, field, value), e.g. ('crop', 'k_leaves', 0.55)."""

    @property
    def entity_kind(self) -> str:
        return "configuration"

    def _copy(self, **overrides: object) -> Configuration:
        return Configuration(
            name=str(overrides["name"]) if "name" in overrides else self.name,
            enabled=bool(overrides["enabled"]) if "enabled" in overrides else self.enabled,
            color=str(overrides["color"]) if "color" in overrides else self.color,
            processes=overrides["processes"] if "processes" in overrides else self.processes,  # type: ignore[arg-type]
            readonly=bool(overrides["readonly"]) if "readonly" in overrides else self.readonly,
            clock=str(overrides["clock"]) if "clock" in overrides else self.clock,
            weather=str(overrides["weather"]) if "weather" in overrides else self.weather,
            soil=str(overrides["soil"]) if "soil" in overrides else self.soil,
            crop=str(overrides["crop"]) if "crop" in overrides else self.crop,
            light_interception=str(overrides["light_interception"]) if "light_interception" in overrides else self.light_interception,
            day_length=str(overrides["day_length"]) if "day_length" in overrides else self.day_length,
            parameter_edits=overrides["parameter_edits"] if "parameter_edits" in overrides else self.parameter_edits,  # type: ignore[arg-type]
        )

    def with_enabled(self, enabled: bool) -> Configuration:
        return self._copy(enabled=enabled)

    def with_name(self, name: str) -> Configuration:
        if self.readonly:
            raise ValueError(f"Configuration {self.name!r} is readonly.")
        return self._copy(name=name, readonly=False)

    def with_processes(self, processes: tuple[Input, ...]) -> Configuration:
        if self.readonly:
            raise ValueError(f"Configuration {self.name!r} is readonly.")
        return self._copy(processes=processes, readonly=False)

    def with_providers(
        self,
        *,
        clock: str | None = None,
        weather: str | None = None,
        soil: str | None = None,
        crop: str | None = None,
        light_interception: str | None = None,
        day_length: str | None = None,
    ) -> Configuration:
        """Change a role's implementation (allowed on readonly defaults)."""
        chosen = {
            "clock": self.clock if clock is None else clock,
            "weather": self.weather if weather is None else weather,
            "soil": self.soil if soil is None else soil,
            "crop": self.crop if crop is None else crop,
            "light_interception": self.light_interception if light_interception is None else light_interception,
            "day_length": self.day_length if day_length is None else day_length,
        }
        current = {
            "clock": self.clock,
            "weather": self.weather,
            "soil": self.soil,
            "crop": self.crop,
            "light_interception": self.light_interception,
            "day_length": self.day_length,
        }
        dropped = {role for role, label in chosen.items() if label != current[role]}
        edits = tuple(edit for edit in self.parameter_edits if edit[0] not in dropped)
        return self._copy(
            clock=chosen["clock"],
            weather=chosen["weather"],
            soil=chosen["soil"],
            crop=chosen["crop"],
            light_interception=chosen["light_interception"],
            day_length=chosen["day_length"],
            parameter_edits=edits,
        )

    def with_parameter(self, role: str, name: str, value: float) -> Configuration:
        """Set a numeric Input field (allowed on readonly defaults)."""
        kept = tuple(edit for edit in self.parameter_edits if edit[0] != role or edit[1] != name)
        return self._copy(parameter_edits=(*kept, (role, name, value)))


def default_configuration() -> Configuration:
    """Readonly baseline: default clock, weather, soil, crop, light interception, and day length."""
    return Configuration(
        name="defaults",
        enabled=True,
        color=DEFAULT_CONFIGURATION_COLORS[0],
        processes=(),
        readonly=True,
        clock=CLOCK,
        weather=WEATHER_OPEN_METEO,
        soil=SOIL_SOILGRIDS,
        crop=CROP_WHEAT,
        light_interception=LIGHT_INTERCEPTION_BEER_LAMBERT,
        day_length=DAY_LENGTH_ASTRONOMICAL,
    )


def next_configuration_color(existing: tuple[Configuration, ...]) -> str:
    used = {c.color for c in existing}
    for color in DEFAULT_CONFIGURATION_COLORS:
        if color not in used:
            return color
    return DEFAULT_CONFIGURATION_COLORS[len(existing) % len(DEFAULT_CONFIGURATION_COLORS)]
