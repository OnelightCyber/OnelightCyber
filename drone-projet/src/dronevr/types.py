"""Types de base partages par tous les modules.

Conventions de repere utilisees dans tout le projet :

* Position : repere NED local (North-East-Down), en metres, origine = point
  de decollage. ``down`` est positif vers le bas, donc l'altitude vaut
  ``-down``.
* Angles : radians partout en interne. Les degres n'apparaissent que dans la
  configuration YAML et dans l'affichage.
* Yaw : 0 = Nord, croissant vers l'Est (sens horaire vu du dessus).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum

__all__ = [
    "ControlInput",
    "FlightMode",
    "HeadPose",
    "StickInput",
    "Telemetry",
    "Vec3",
    "clamp",
    "wrap_pi",
]


def clamp(value: float, low: float, high: float) -> float:
    """Borne ``value`` dans ``[low, high]``.

    ``low`` doit etre inferieur ou egal a ``high``, sinon le resultat n'a
    pas de sens ; on leve plutot que de renvoyer une valeur silencieusement
    fausse, parce qu'une borne inversee dans la config est un bug que l'on
    veut voir tout de suite.
    """
    if low > high:
        raise ValueError(f"bornes inversees : low={low} > high={high}")
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    """Ramene un angle dans ``]-pi, pi]``.

    Indispensable pour les differences de cap : sans ca, passer de 179 deg a
    -179 deg est vu comme une rotation de 358 deg au lieu de 2 deg.
    """
    return -(math.remainder(-angle, 2.0 * math.pi))


@dataclass(frozen=True, slots=True)
class Vec3:
    """Vecteur 3D immuable (NED lorsqu'il s'agit d'une position)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def horizontal_norm(self) -> float:
        """Distance au sol, en ignorant l'altitude."""
        return math.hypot(self.x, self.y)


class FlightMode(str, Enum):
    """Etats possibles du drone.

    La transition normale est
    ``DISARMED -> ARMED -> FLYING -> LANDING -> DISARMED``.
    ``FAILSAFE`` peut etre atteint depuis n'importe quel etat en vol.
    """

    DISARMED = "disarmed"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    FAILSAFE = "failsafe"


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Orientation de la tete du pilote, lue par le casque VR.

    Les angles sont *absolus* (repere du casque), pas encore recentres :
    c'est ``HeadTrackingMapper`` qui applique le zero de calibration.
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)

    def as_degrees(self) -> tuple[float, float, float]:
        return (
            math.degrees(self.yaw),
            math.degrees(self.pitch),
            math.degrees(self.roll),
        )

    def age(self, now: float | None = None) -> float:
        """Anciennete de la mesure en secondes (utilisee par le watchdog)."""
        return (time.monotonic() if now is None else now) - self.timestamp


@dataclass(frozen=True, slots=True)
class StickInput:
    """Commandes manuelles du pilote, independantes de la tete.

    La tete gere le *regard* (nacelle et lacet) ; les manches gerent le
    *deplacement*. Separer les deux evite qu'un simple coup d'oeil ne fasse
    partir le drone en translation.

    ``forward``/``strafe`` sont dans ``[-1, 1]``, ``throttle`` dans ``[0, 1]``
    ou ``throttle = 0.5`` correspond au maintien d'altitude.
    """

    forward: float = 0.0
    strafe: float = 0.0
    throttle: float = 0.5

    def clamped(self) -> "StickInput":
        """Renvoie une copie bornee : l'UI web n'est pas une source de confiance."""
        return StickInput(
            forward=clamp(self.forward, -1.0, 1.0),
            strafe=clamp(self.strafe, -1.0, 1.0),
            throttle=clamp(self.throttle, 0.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class ControlInput:
    """Consignes envoyees au drone, toutes normalisees.

    ``roll``/``pitch``/``yaw_rate``/``throttle`` sont dans ``[-1, 1]``
    (``throttle`` dans ``[0, 1]``), ce qui rend le mapping independant du
    modele de drone : c'est le backend qui convertit en angles ou en PWM.
    ``gimbal_*`` sont en radians, car la nacelle se commande en position.
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw_rate: float = 0.0
    throttle: float = 0.0
    gimbal_yaw: float = 0.0
    gimbal_pitch: float = 0.0

    def neutral(self) -> "ControlInput":
        """Consigne de vol stationnaire, nacelle conservee.

        On garde l'orientation de la nacelle : couper la video en plein vol
        desoriente le pilote, ce qui est pire que le probleme d'origine.
        """
        return replace(
            self, roll=0.0, pitch=0.0, yaw_rate=0.0, throttle=HOVER_THROTTLE
        )


#: Gaz correspondant au vol stationnaire pour le modele simule.
#: Les backends reels le recalculent a partir de la masse et de la poussee.
HOVER_THROTTLE = 0.5


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Etat complet du drone a un instant donne."""

    position: Vec3 = Vec3()
    velocity: Vec3 = Vec3()
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gimbal_yaw: float = 0.0
    gimbal_pitch: float = 0.0
    battery: float = 1.0
    mode: FlightMode = FlightMode.DISARMED
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def altitude(self) -> float:
        """Altitude en metres au-dessus du point de decollage."""
        return -self.position.z

    @property
    def ground_speed(self) -> float:
        return self.velocity.horizontal_norm()

    @property
    def distance_from_home(self) -> float:
        return self.position.horizontal_norm()

    @property
    def heading_deg(self) -> float:
        """Cap en degres dans ``[0, 360[``, comme sur un compas."""
        return math.degrees(self.yaw) % 360.0

    def to_dict(self) -> dict:
        """Serialisation pour le WebSocket (JSON, degres, arrondi)."""
        return {
            "position": {
                "north": round(self.position.x, 2),
                "east": round(self.position.y, 2),
                "down": round(self.position.z, 2),
            },
            "altitude": round(self.altitude, 2),
            "ground_speed": round(self.ground_speed, 2),
            "distance": round(self.distance_from_home, 2),
            "roll": round(math.degrees(self.roll), 1),
            "pitch": round(math.degrees(self.pitch), 1),
            "yaw": round(self.heading_deg, 1),
            "gimbal_yaw": round(math.degrees(self.gimbal_yaw), 1),
            "gimbal_pitch": round(math.degrees(self.gimbal_pitch), 1),
            "battery": round(self.battery * 100.0, 1),
            "mode": self.mode.value,
        }
