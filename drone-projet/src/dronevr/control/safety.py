"""Filet de securite applique entre le pilote et le drone.

Aucune consigne ne descend vers le drone sans passer par ``SafetyMonitor``.
Il ne *conseille* pas : il reecrit les commandes. Quatre protections :

* **Watchdog casque** : plus de pose de tete -> stationnaire.
* **Geofence** : plafond d'altitude et rayon maximal autour du decollage.
* **Batterie** : alerte puis atterrissage automatique.
* **Arret d'urgence** : atterrissage force, verrouille jusqu'a acquittement.

Toutes ces protections sont *latchees* quand elles sont critiques : une fois
le failsafe batterie declenche, une remontee de tension (frequente quand on
reduit les gaz) ne redonne pas la main au pilote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import SafetyConfig
from ..types import HOVER_THROTTLE, ControlInput, Telemetry, Vec3, clamp

__all__ = ["SafetyMonitor", "SafetyVerdict"]


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """Resultat d'un passage dans le moniteur."""

    control: ControlInput
    warnings: tuple[str, ...] = ()
    force_land: bool = False

    @property
    def ok(self) -> bool:
        return not self.warnings and not self.force_land


class SafetyMonitor:
    """Applique les limites de securite a une consigne."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._battery_latched = False
        self._emergency = False

    @property
    def emergency(self) -> bool:
        return self._emergency

    @property
    def battery_failsafe_active(self) -> bool:
        return self._battery_latched

    def trigger_emergency(self) -> None:
        """Declenche un atterrissage d'urgence.

        On descend, on ne coupe pas les moteurs : couper en vol transforme un
        probleme rattrapable en chute. La coupure moteur reste disponible via
        le backend, pour le cas ou le drone est deja au sol.
        """
        self._emergency = True

    def clear(self) -> None:
        """Acquitte les failsafes verrouilles.

        A n'appeler qu'au sol, apres avoir compris pourquoi ils se sont
        declenches (batterie changee, casque rebranche...).
        """
        self._battery_latched = False
        self._emergency = False

    def check(
        self,
        control: ControlInput,
        telemetry: Telemetry,
        head_age: float,
    ) -> SafetyVerdict:
        """Renvoie la consigne corrigee et la liste des alertes.

        ``head_age`` est l'anciennete de la derniere pose de tete, en secondes.
        """
        cfg = self.config
        warnings: list[str] = []
        force_land = False

        if telemetry.battery <= cfg.battery_failsafe:
            self._battery_latched = True
        if self._battery_latched:
            warnings.append("BATTERIE CRITIQUE : atterrissage automatique")
            force_land = True
        elif telemetry.battery <= cfg.battery_warning:
            warnings.append(
                f"batterie faible ({telemetry.battery * 100:.0f} %)"
            )

        if self._emergency:
            warnings.append("ARRET D'URGENCE : atterrissage force")
            force_land = True

        # Watchdog casque. Sans pose fraiche on ne sait plus ou regarde le
        # pilote : on fige tout deplacement mais on garde la nacelle en place,
        # pour que l'image ne parte pas dans tous les sens.
        if head_age > cfg.head_timeout:
            warnings.append(f"casque muet depuis {head_age:.1f} s : stationnaire")
            control = control.neutral()

        if force_land:
            # Descente controlee : on annule toute translation et on impose un
            # gaz legerement inferieur au stationnaire.
            return SafetyVerdict(
                control=ControlInput(
                    roll=0.0,
                    pitch=0.0,
                    yaw_rate=0.0,
                    throttle=HOVER_THROTTLE * 0.75,
                    gimbal_yaw=control.gimbal_yaw,
                    gimbal_pitch=control.gimbal_pitch,
                ),
                warnings=tuple(warnings),
                force_land=True,
            )

        control, fence_warnings = self._apply_geofence(control, telemetry)
        warnings.extend(fence_warnings)

        return SafetyVerdict(control=control, warnings=tuple(warnings))

    def _apply_geofence(
        self, control: ControlInput, telemetry: Telemetry
    ) -> tuple[ControlInput, list[str]]:
        """Empeche de sortir du volume autorise, sans bloquer le retour.

        Le point important est de ne retirer que la composante *sortante* de
        la commande : un blocage total pieger ait le drone contre la barriere,
        incapable de revenir.
        """
        cfg = self.config
        warnings: list[str] = []
        roll, pitch, throttle = control.roll, control.pitch, control.throttle

        if telemetry.altitude >= cfg.max_altitude and throttle > HOVER_THROTTLE:
            warnings.append(f"plafond atteint ({cfg.max_altitude:.0f} m)")
            throttle = HOVER_THROTTLE

        distance = telemetry.distance_from_home
        if distance >= cfg.max_distance and distance > 1e-6:
            # Les consignes sont dans le repere du drone ; la barriere est dans
            # le repere terrestre. On projette donc dans le repere NED, on
            # retire ce qui s'eloigne, puis on revient au repere du drone.
            yaw = telemetry.yaw
            forward = Vec3(math.cos(yaw), math.sin(yaw), 0.0)
            right = Vec3(-math.sin(yaw), math.cos(yaw), 0.0)

            command_ned = forward * pitch + right * roll
            radial = Vec3(
                telemetry.position.x / distance, telemetry.position.y / distance, 0.0
            )
            outward = command_ned.x * radial.x + command_ned.y * radial.y

            if outward > 0.0:
                warnings.append(f"limite de distance ({cfg.max_distance:.0f} m)")
                corrected = command_ned - radial * outward
                # Retour au repere du drone : la base (forward, right) est
                # orthonormee, donc l'inverse est une simple projection.
                pitch = clamp(
                    corrected.x * forward.x + corrected.y * forward.y, -1.0, 1.0
                )
                roll = clamp(
                    corrected.x * right.x + corrected.y * right.y, -1.0, 1.0
                )

        if (roll, pitch, throttle) == (control.roll, control.pitch, control.throttle):
            return control, warnings

        return (
            ControlInput(
                roll=roll,
                pitch=pitch,
                yaw_rate=control.yaw_rate,
                throttle=throttle,
                gimbal_yaw=control.gimbal_yaw,
                gimbal_pitch=control.gimbal_pitch,
            ),
            warnings,
        )
