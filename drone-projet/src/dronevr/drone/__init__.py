"""Backends drone : simulation et MAVLink."""

from .base import DroneBackend
from .simulated import SimulatedDrone

__all__ = ["DroneBackend", "SimulatedDrone", "create_drone"]


def create_drone(config) -> DroneBackend:
    """Instancie le backend correspondant a ``config.backend``.

    L'import de MAVLink est fait ici, et pas en tete de module, pour que le
    projet tourne sans ``pymavlink`` installe tant qu'on reste en simulation.
    """
    if config.backend == "sim":
        return SimulatedDrone(control=config.control, safety=config.safety)

    if config.backend == "real":
        from .mavlink import MavlinkDrone

        return MavlinkDrone(
            url=config.mavlink_url, control=config.control, safety=config.safety
        )

    raise ValueError(f"backend inconnu : {config.backend!r}")
