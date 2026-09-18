"""Casque simule : permet de tout tester sans materiel VR.

Trois usages :

* ``manual`` : la pose est poussee depuis l'interface web (souris, clavier,
  ecran tactile). C'est ce qui permet de faire la demo complete sans casque.
* ``orbit`` : la tete balaie automatiquement de gauche a droite. Pratique
  pour verifier d'un coup d'oeil que la nacelle suit.
* ``still`` : tete immobile, pour tester le reste de la chaine.
"""

from __future__ import annotations

import math
import threading
import time

from ..types import HeadPose, clamp
from .base import HeadTracker

__all__ = ["SimulatedTracker"]

#: Amplitudes du mode ``orbit``.
ORBIT_YAW_AMPLITUDE = math.radians(60.0)
ORBIT_PITCH_AMPLITUDE = math.radians(20.0)
ORBIT_PERIOD = 8.0


class SimulatedTracker(HeadTracker):
    """Casque virtuel. Sert aussi de reference pour ``WebXRTracker``."""

    def __init__(self, pattern: str = "manual") -> None:
        if pattern not in {"manual", "orbit", "still"}:
            raise ValueError(
                f"motif inconnu : {pattern!r} (attendu manual|orbit|still)"
            )
        self.pattern = pattern
        self._lock = threading.Lock()
        self._pose = HeadPose()
        self._t0 = time.monotonic()

    def set_pose(self, yaw: float, pitch: float, roll: float = 0.0) -> None:
        """Injecte une pose (mode ``manual``), en radians.

        Appelee depuis le serveur web, donc depuis un autre thread que la
        boucle de controle : d'ou le verrou.
        """
        pose = HeadPose(
            yaw=clamp(yaw, -math.pi, math.pi),
            pitch=clamp(pitch, -math.pi / 2, math.pi / 2),
            roll=clamp(roll, -math.pi, math.pi),
        )
        with self._lock:
            self._pose = pose

    def read(self) -> HeadPose:
        if self.pattern == "orbit":
            elapsed = time.monotonic() - self._t0
            phase = 2.0 * math.pi * elapsed / ORBIT_PERIOD
            return HeadPose(
                yaw=ORBIT_YAW_AMPLITUDE * math.sin(phase),
                # Deux fois plus vite en tangage : la trajectoire dessine un
                # 8, ce qui exerce les deux axes en meme temps.
                pitch=ORBIT_PITCH_AMPLITUDE * math.sin(2.0 * phase),
                roll=0.0,
            )

        if self.pattern == "still":
            return HeadPose()

        with self._lock:
            return self._pose
