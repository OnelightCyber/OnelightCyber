"""Transformation de l'orientation de la tete en consignes de vol.

C'est la piece centrale du projet : elle repond a la question "je tourne la
tete, que fait le drone ?". La chaine de traitement est toujours la meme,
dans cet ordre :

1. **Recentrage** sur le zero de calibration (le pilote regarde droit devant).
2. **Lissage** passe-bas, pour supprimer la trepidation du casque.
3. **Zone morte** avec remise a l'echelle, pour ignorer les micro-mouvements.
4. **Expo**, pour du pilotage fin autour du neutre.
5. **Mapping** selon le mode choisi (nacelle seule, lacet suiveur, total).
6. **Limitation de vitesse** de la nacelle, puis bornage final.

Chaque etape est une fonction pure et testable separement ; seul
``HeadTrackingMapper`` porte un etat (le zero, le filtre, la nacelle).
"""

from __future__ import annotations

import math

from ..config import ControlConfig
from ..types import ControlInput, HeadPose, StickInput, clamp, wrap_pi

__all__ = ["HeadTrackingMapper", "apply_deadzone", "apply_expo", "low_pass"]


def apply_deadzone(value: float, deadzone: float, maximum: float) -> float:
    """Applique une zone morte et normalise le reste dans ``[-1, 1]``.

    La remise a l'echelle est ce qui evite le "saut" classique : sans elle, la
    consigne passerait brutalement de 0 a ``deadzone / maximum`` des que la
    tete franchit le seuil. Ici la sortie vaut exactement 0 au bord de la zone
    morte et exactement 1 a ``maximum``.
    """
    if maximum <= deadzone:
        raise ValueError("maximum doit etre strictement superieur a deadzone")
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (maximum - deadzone)
    return math.copysign(min(scaled, 1.0), value)


def apply_expo(value: float, expo: float) -> float:
    """Courbe exponentielle facon radiocommande.

    ``y = expo * x^3 + (1 - expo) * x``. Les extremes sont preserves
    (``y(1) = 1``), seule la pente au centre est adoucie, ce qui donne de la
    precision quand on bouge peu la tete sans perdre de debattement.
    """
    if not 0.0 <= expo < 1.0:
        raise ValueError("expo doit etre dans [0, 1[")
    return expo * value**3 + (1.0 - expo) * value


def low_pass(previous: float, target: float, tau: float, dt: float) -> float:
    """Filtre passe-bas du premier ordre, correct pour des angles.

    L'ecart est calcule avec ``wrap_pi`` : sinon, un passage de +179 deg a
    -179 deg serait vu comme une rotation de presque un tour et la camera
    partirait dans le mauvais sens.

    ``tau`` est la constante de temps : plus elle est grande, plus c'est doux
    mais plus la latence augmente. ``tau = 0`` desactive le filtre.
    """
    if tau <= 0.0 or dt <= 0.0:
        return target
    alpha = dt / (tau + dt)
    return wrap_pi(previous + alpha * wrap_pi(target - previous))


def _slew(previous: float, target: float, max_rate: float, dt: float) -> float:
    """Limite la vitesse de variation a ``max_rate`` rad/s."""
    if max_rate <= 0.0 or dt <= 0.0:
        return target
    max_step = max_rate * dt
    return previous + clamp(target - previous, -max_step, max_step)


class HeadTrackingMapper:
    """Convertit une suite de ``HeadPose`` en ``ControlInput``.

    L'objet est volontairement stateful : le lissage, la limitation de vitesse
    de la nacelle et le zero de calibration ont besoin de se souvenir de
    l'iteration precedente.
    """

    def __init__(self, config: ControlConfig) -> None:
        self.config = config
        self._zero_yaw = 0.0
        self._zero_pitch = 0.0
        self._filtered_yaw = 0.0
        self._filtered_pitch = 0.0
        self._gimbal_yaw = 0.0
        self._gimbal_pitch = 0.0
        self._calibrated = False

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, pose: HeadPose) -> None:
        """Definit l'orientation courante comme le "droit devant".

        A appeler quand le pilote regarde l'axe du drone. Le filtre et la
        nacelle sont remis a zero pour que le recentrage soit immediat et non
        progressif.
        """
        self._zero_yaw = pose.yaw
        self._zero_pitch = pose.pitch
        self._filtered_yaw = 0.0
        self._filtered_pitch = 0.0
        self._gimbal_yaw = 0.0
        self._gimbal_pitch = 0.0
        self._calibrated = True

    def reset(self) -> None:
        """Remet la nacelle et le filtre a zero sans toucher a la calibration."""
        self._filtered_yaw = 0.0
        self._filtered_pitch = 0.0
        self._gimbal_yaw = 0.0
        self._gimbal_pitch = 0.0

    def update(
        self,
        pose: HeadPose,
        sticks: StickInput | None = None,
        dt: float = 0.02,
    ) -> ControlInput:
        """Calcule les consignes correspondant a ``pose``.

        ``sticks`` porte les commandes de deplacement (gaz, avant/arriere,
        lateral) qui, elles, ne viennent pas de la tete. Sans argument, le
        drone se contente de tenir son altitude.
        """
        cfg = self.config
        sticks = (sticks or StickInput()).clamped()

        # La premiere pose recue sert de reference : sans ca, le drone partirait
        # en butee des la premiere image si le casque demarre de travers.
        if not self._calibrated:
            self.calibrate(pose)

        rel_yaw = wrap_pi(pose.yaw - self._zero_yaw)
        rel_pitch = wrap_pi(pose.pitch - self._zero_pitch)

        self._filtered_yaw = low_pass(
            self._filtered_yaw, rel_yaw, cfg.smoothing_tau, dt
        )
        self._filtered_pitch = low_pass(
            self._filtered_pitch, rel_pitch, cfg.smoothing_tau, dt
        )

        norm_yaw = apply_expo(
            apply_deadzone(self._filtered_yaw, cfg.head_deadzone, cfg.max_head_yaw),
            cfg.expo,
        )
        norm_pitch = apply_expo(
            apply_deadzone(self._filtered_pitch, cfg.head_deadzone, cfg.max_head_pitch),
            cfg.expo,
        )

        target_gimbal_yaw, target_gimbal_pitch, yaw_rate, drone_pitch = self._map_mode(
            norm_yaw, norm_pitch
        )

        # La nacelle se deplace en position : on limite sa vitesse pour ne pas
        # rendre l'image inexploitable ni forcer la mecanique.
        self._gimbal_yaw = _slew(
            self._gimbal_yaw, target_gimbal_yaw, cfg.max_gimbal_rate, dt
        )
        self._gimbal_pitch = _slew(
            self._gimbal_pitch, target_gimbal_pitch, cfg.max_gimbal_rate, dt
        )

        return ControlInput(
            roll=clamp(sticks.strafe, -1.0, 1.0),
            pitch=clamp(sticks.forward + drone_pitch, -1.0, 1.0),
            yaw_rate=clamp(yaw_rate, -1.0, 1.0),
            throttle=sticks.throttle,
            gimbal_yaw=clamp(
                self._gimbal_yaw, -cfg.max_gimbal_yaw, cfg.max_gimbal_yaw
            ),
            gimbal_pitch=clamp(
                self._gimbal_pitch, -cfg.max_gimbal_pitch, cfg.max_gimbal_pitch
            ),
        )

    def _map_mode(
        self, norm_yaw: float, norm_pitch: float
    ) -> tuple[float, float, float, float]:
        """Repartit les consignes normalisees selon le mode de pilotage.

        Renvoie ``(nacelle_lacet, nacelle_tangage, vitesse_lacet, tangage_drone)``.
        Les deux premiers sont en radians, les deux derniers normalises.
        """
        cfg = self.config

        if cfg.mode == "gimbal":
            # Le drone ne bouge pas : seule la camera suit la tete. C'est le
            # mode a utiliser pour les premiers essais et pour la demo.
            return (
                norm_yaw * cfg.max_gimbal_yaw,
                norm_pitch * cfg.max_gimbal_pitch,
                0.0,
                0.0,
            )

        if cfg.mode == "yaw_follow":
            # Le lacet de la tete fait pivoter le drone entier ; la nacelle ne
            # garde que le tangage. C'est le mode "immersif" : le drone tourne
            # vraiment quand on tourne la tete.
            return (0.0, norm_pitch * cfg.max_gimbal_pitch, norm_yaw, 0.0)

        if cfg.mode == "full":
            # La tete pilote aussi l'assiette : baisser la tete fait avancer.
            # Tres immersif, mais il n'existe plus de position "neutre" ou le
            # drone est garanti immobile.
            return (0.0, norm_pitch * cfg.max_gimbal_pitch * 0.5, norm_yaw, norm_pitch)

        raise ValueError(f"mode de pilotage inconnu : {cfg.mode!r}")
