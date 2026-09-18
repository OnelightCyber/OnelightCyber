"""Suivi de la tete du pilote : casque simule ou casque reel (WebXR)."""

from .base import HeadTracker
from .simulated import SimulatedTracker
from .webxr import WebXRTracker, quaternion_to_euler

__all__ = [
    "HeadTracker",
    "SimulatedTracker",
    "WebXRTracker",
    "create_tracker",
    "quaternion_to_euler",
]


def create_tracker(config) -> HeadTracker:
    """Instancie le tracker correspondant a ``config.tracker``."""
    if config.tracker == "sim":
        return SimulatedTracker()
    if config.tracker == "webxr":
        return WebXRTracker()
    raise ValueError(f"tracker inconnu : {config.tracker!r}")
