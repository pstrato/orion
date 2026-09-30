"""Day length from latitude and the clock date.

Astronomical day length is the FAO-56 geometric daylight duration: the sun's
centre on the sea-level horizon. Apparent day length uses the same declination,
then sets the sun's upper limb on a horizon lowered by refraction and by the
site altitude.
"""

from __future__ import annotations

import jax.numpy as jnp

from orion.core.entity import entity
from orion.core.input import Input, LocationInput
from orion.core.process import Process
from orion.core.quantity import Quantity, is_non_negative, is_scalar
from orion.core.state import State
from orion.core.variable import Variable, var
from orion.processes.clock import Clock


@entity()
class DayLength(State):
    """Day length for the current clock date."""

    hours: Variable
    """Day length (h)."""


@entity()
class DayLengthProcess(Process):
    """Day length from the clock and the location."""

    def step(self, clock: Clock, location: LocationInput) -> tuple[DayLength]:
        raise NotImplementedError()


# Refraction at the horizon plus the sun's semi-diameter, in degrees.
_REFRACTION_AND_RADIUS_DEG = 0.833
_EARTH_RADIUS_M = 6_371_000.0


def _declination(day_of_year: Quantity) -> jnp.ndarray:
    """FAO-56 solar declination (radians) for a day of year."""
    return 0.409 * jnp.sin(2.0 * jnp.pi * jnp.asarray(day_of_year.value) / 365.0 - 1.39)


def astronomical_day_length(latitude: Quantity, day_of_year: Quantity) -> Variable:
    """FAO-56 daylight hours for one latitude and day of year."""
    declination = _declination(day_of_year)
    latitude_rad = jnp.deg2rad(jnp.asarray(latitude.value))
    beyond_polar_circle = jnp.abs(latitude_rad) >= (0.5 * jnp.pi - jnp.abs(declination))
    same_hemisphere = latitude_rad * declination > 0
    argument = jnp.clip(-jnp.tan(latitude_rad) * jnp.tan(declination), -1.0, 1.0)
    hours = 24.0 / jnp.pi * jnp.arccos(argument)
    hours = jnp.where(beyond_polar_circle & same_hemisphere, 24.0, hours)
    hours = jnp.where(beyond_polar_circle & ~same_hemisphere, 0.0, hours)
    return var("hours", "hours", hours, "Astronomical day length", is_scalar + is_non_negative)


def apparent_day_length(latitude: Quantity, day_of_year: Quantity, altitude: Quantity) -> Variable:
    """Daylight hours with refraction, the sun's radius, and altitude.

    The horizon is 0.833° below the geometric horizon at sea level. Altitude
    lowers it further by the geometric dip of a smooth earth. Sites below sea
    level use the sea-level horizon. Polar day and polar night are 24 h and 0 h.
    """
    declination = _declination(day_of_year)
    latitude_rad = jnp.deg2rad(jnp.asarray(latitude.value))
    height = jnp.maximum(jnp.asarray(altitude.value), 0.0)
    dip = jnp.arccos(_EARTH_RADIUS_M / (_EARTH_RADIUS_M + height))
    solar_elevation = jnp.deg2rad(-_REFRACTION_AND_RADIUS_DEG) - dip
    numer = jnp.sin(solar_elevation) - jnp.sin(latitude_rad) * jnp.sin(declination)
    denom = jnp.cos(latitude_rad) * jnp.cos(declination)
    polar_night = numer >= jnp.abs(denom)
    polar_day = numer <= -jnp.abs(denom)
    cos_hour = jnp.clip(numer / jnp.where(denom == 0, 1.0, denom), -1.0, 1.0)
    hours = 24.0 / jnp.pi * jnp.arccos(cos_hour)
    hours = jnp.where(polar_day, 24.0, hours)
    hours = jnp.where(polar_night, 0.0, hours)
    return var("hours", "hours", hours, "Apparent day length", is_scalar + is_non_negative)


def _hours(value: jnp.ndarray, description: str) -> Variable:
    return var("hours", "hours", value, description, is_scalar + is_non_negative)


@entity()
class AstronomicalDayLengthProcess(DayLengthProcess):
    """FAO-56 astronomical day length."""

    def step(self, clock: Clock, location: LocationInput) -> tuple[DayLength]:
        latitude = var("latitude", "unitless", jnp.asarray(location.geometry.value.y), "Latitude in degrees north")
        return (DayLength(name="day_length", constraint=None, hours=astronomical_day_length(latitude, clock.doy)),)


@entity()
class AstronomicalDayLengthInput(Input):
    """FAO-56 astronomical day length."""

    def states(self) -> DayLength:
        """Initial day length, before the first step."""
        return DayLength(name="day_length", constraint=None, hours=_hours(jnp.zeros(()), "Astronomical day length"))

    def processes(self) -> AstronomicalDayLengthProcess:
        return AstronomicalDayLengthProcess(self.name)


@entity()
class ApparentDayLengthProcess(DayLengthProcess):
    """Apparent day length: refraction, the sun's radius, and altitude."""

    def step(self, clock: Clock, location: LocationInput) -> tuple[DayLength]:
        latitude = var("latitude", "unitless", jnp.asarray(location.geometry.value.y), "Latitude in degrees north")
        return (DayLength(name="day_length", constraint=None, hours=apparent_day_length(latitude, clock.doy, location.altitude)),)


@entity()
class ApparentDayLengthInput(Input):
    """Apparent day length, including refraction and altitude."""

    def states(self) -> DayLength:
        """Initial day length, before the first step."""
        return DayLength(name="day_length", constraint=None, hours=_hours(jnp.zeros(()), "Apparent day length"))

    def processes(self) -> ApparentDayLengthProcess:
        return ApparentDayLengthProcess(self.name)
