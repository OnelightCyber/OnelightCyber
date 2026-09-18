"""Casque reel : poses recues du navigateur via WebXR.

Le navigateur du casque (Meta Quest, Pico, SteamVR...) lit l'orientation de
la tete avec l'API WebXR, la convertit en angles d'Euler et l'envoie sur le
WebSocket. Ce module est le point d'arrivee cote Python : il stocke la
derniere pose recue et la met a disposition de la boucle de controle.

Le decouplage est volontaire. Le casque emet a 72-90 Hz, la boucle de
controle tourne a 50 Hz, et les deux cadences n'ont aucune raison d'etre
synchronisees. On ne bloque donc jamais l'une en attendant l'autre.
"""

from __future__ import annotations

import math
import threading
import time

from ..types import HeadPose, clamp, wrap_pi
from .base import HeadTracker

__all__ = ["WebXRTracker", "quaternion_to_euler"]


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convertit un quaternion WebXR en ``(yaw, pitch, roll)`` radians.

    WebXR travaille dans un repere main droite Y-up : ``-Z`` pointe devant le
    pilote, ``+Y`` vers le haut. La conversion produit directement les angles
    dans la convention du projet (lacet autour de la verticale, tangage
    positif = regarder vers le haut).

    Le cas ``|sinp| >= 1`` est le blocage de cardan : quand le pilote regarde
    exactement au zenith ou au nadir, le lacet n'est plus defini. On sature a
    +/- 90 deg plutot que de laisser ``asin`` lever une erreur de domaine.
    """
    # Tangage
    sinp = 2.0 * (w * x - y * z)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Lacet
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))

    # Roulis
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x * x + z * z))

    return wrap_pi(yaw), pitch, wrap_pi(roll)


class WebXRTracker(HeadTracker):
    """Recoit les poses du casque et les tient a disposition.

    Sur de vrais casques la cadence depasse rarement 90 Hz, mais rien
    n'empeche un client de spammer : ``max_rate`` limite le traitement pour
    qu'un client mal ecrit ne puisse pas saturer la station au sol.
    """

    def __init__(self, max_rate: float = 200.0) -> None:
        if max_rate <= 0.0:
            raise ValueError("max_rate doit etre > 0")
        self._min_interval = 1.0 / max_rate
        self._lock = threading.Lock()
        self._pose = HeadPose(timestamp=0.0)
        self._last_accept = 0.0
        self._received = 0
        self._dropped = 0

    @property
    def stats(self) -> dict[str, int]:
        """Compteurs de diagnostic, affiches dans l'interface."""
        with self._lock:
            return {"received": self._received, "dropped": self._dropped}

    @property
    def connected(self) -> bool:
        """Vrai si au moins une pose a ete recue."""
        with self._lock:
            return self._pose.timestamp > 0.0

    def submit_quaternion(self, x: float, y: float, z: float, w: float) -> bool:
        """Enregistre une pose recue du casque. Renvoie ``False`` si ignoree.

        Un quaternion doit etre unitaire ; s'il ne l'est pas, la mesure est
        corrompue et l'appliquer ferait partir la camera n'importe ou.
        """
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if not math.isfinite(norm) or norm < 1e-6:
            with self._lock:
                self._dropped += 1
            return False

        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        yaw, pitch, roll = quaternion_to_euler(x, y, z, w)
        return self.submit_euler(yaw, pitch, roll)

    def submit_euler(self, yaw: float, pitch: float, roll: float) -> bool:
        """Enregistre une pose deja convertie en angles d'Euler (radians)."""
        if not all(math.isfinite(v) for v in (yaw, pitch, roll)):
            with self._lock:
                self._dropped += 1
            return False

        now = time.monotonic()
        with self._lock:
            if now - self._last_accept < self._min_interval:
                self._dropped += 1
                return False
            self._last_accept = now
            self._received += 1
            self._pose = HeadPose(
                yaw=wrap_pi(yaw),
                pitch=clamp(pitch, -math.pi / 2, math.pi / 2),
                roll=wrap_pi(roll),
                timestamp=now,
            )
        return True

    def read(self) -> HeadPose:
        with self._lock:
            return self._pose
