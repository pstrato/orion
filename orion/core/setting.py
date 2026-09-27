from pathlib import Path

from orion.core.entity import Entity, entity


@entity()
class Settings(Entity):
    """Runtime settings: cache paths and other configuration."""

    cache_path: Path
    """Directory for on-disk caches (datasets, ``http.sqlite`` for client responses)."""

    validate_inputs: bool
    """Validate input on simulation start."""

    validate_initial_states: bool
    """Validate initial states."""

    validate_simulation_states: bool
    """Validate simulation states."""

    use_gpu: bool = False
    """Run simulations on a GPU."""
