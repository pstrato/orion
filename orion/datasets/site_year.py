"""One wheat site-year treatment: management events plus observations."""

from __future__ import annotations

from datetime import date, timedelta

from orion.core.entity import Entity, entity
from orion.core.input import LocationInput
from orion.datasets.convert import grain_protein_from_nitrogen
from orion.datasets.events import FertiliserEvent, IrrigationEvent, ManagementEvent, ProtectionEvent, SowingEvent
from orion.datasets.observations import Observation


@entity()
class SiteYear(Entity):
    """A single treatment in one site-year of a wheat experiment."""

    crop: str
    """Crop species, normalised to wheat."""

    cultivar: str
    """Cultivar name when published."""

    season: str
    """Experimental year or season label."""

    treatment: str
    """Treatment identifier within the site-year."""

    source: str
    """Archive id (westerfeld, muncheberg, broadbalk, braunschweig, agmip_kassie, calibration packs)."""

    location: LocationInput
    """Geographic location of the trial."""

    events: tuple[ManagementEvent, ...] = ()
    """Dated sowing, fertiliser, protection, and irrigation events."""

    observations: tuple[Observation, ...] = ()
    """Dated yield, protein, biomass, and other measurements."""

    def sowings(self) -> tuple[SowingEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, SowingEvent))

    def fertilisers(self) -> tuple[FertiliserEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, FertiliserEvent))

    def protections(self) -> tuple[ProtectionEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, ProtectionEvent))

    def irrigations(self) -> tuple[IrrigationEvent, ...]:
        return tuple(event for event in self.events if isinstance(event, IrrigationEvent))

    def horizon(self) -> tuple[date, date]:
        """Inclusive start and exclusive end covering dated events and observations."""
        dates: list[date] = [event.on.value for event in self.events]
        dates.extend(item.on.value for item in self.observations)
        if not dates:
            raise ValueError(f"{self.name} has no dated events or observations.")
        start = min(dates)
        end = max(dates) + timedelta(days=1)
        if end <= start:
            end = start + timedelta(days=1)
        return start, end

    def latest(self, quantity: str) -> Observation | None:
        found = tuple(item for item in self.observations if item.reading.name == quantity)
        if not found:
            return None
        return max(found, key=lambda item: item.on.value)

    def grain_protein_fraction(self) -> float | None:
        """Grain protein as kg protein / kg grain (falls back from grain N)."""
        protein = self.latest("grain_protein")
        if protein is not None:
            return float(protein.reading.value)
        nitrogen = self.latest("grain_n")
        if nitrogen is not None:
            return grain_protein_from_nitrogen(float(nitrogen.reading.value))
        return None

    def grain_protein_percent(self) -> float | None:
        """Deprecated alias: returns kg/kg (not percent). Prefer ``grain_protein_fraction``."""
        return self.grain_protein_fraction()
