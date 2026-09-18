"""Chargement et validation de la configuration.

Toute la config vit dans un YAML (``config/default.yaml``). Les angles y sont
ecrits en **degres** parce que c'est ce qu'un humain manipule ; ils sont
convertis en radians a la lecture, une bonne fois pour toutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "ControlConfig", "SafetyConfig", "VideoConfig", "load_config"]

#: Config par defaut livree avec le projet.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

#: Champs exprimes en degres dans le YAML et convertis en radians a la lecture.
_DEGREE_FIELDS = frozenset(
    {
        "max_gimbal_yaw",
        "max_gimbal_pitch",
        "head_deadzone",
        "max_head_yaw",
        "max_head_pitch",
        "max_tilt",
        "max_yaw_rate",
        "camera_fov",
    }
)


@dataclass(slots=True)
class ControlConfig:
    """Reglages du mapping tete -> drone."""

    #: ``gimbal`` : la tete ne bouge que la nacelle (mode par defaut, le plus sur).
    #: ``yaw_follow`` : le lacet de la tete fait tourner le drone, le tangage
    #: reste sur la nacelle.
    #: ``full`` : la tete pilote aussi l'assiette. Reserve aux pilotes entraines.
    mode: str = "gimbal"

    #: En dessous de cet angle, la tete est consideree immobile. Sans zone
    #: morte, les micro-mouvements du cou font deriver la camera en permanence.
    head_deadzone: float = math.radians(2.0)

    #: Amplitude de tete correspondant a la consigne maximale.
    max_head_yaw: float = math.radians(75.0)
    max_head_pitch: float = math.radians(50.0)

    #: Butees mecaniques de la nacelle.
    max_gimbal_yaw: float = math.radians(140.0)
    max_gimbal_pitch: float = math.radians(90.0)

    #: Assiette et vitesse de lacet maximales du drone.
    max_tilt: float = math.radians(25.0)
    max_yaw_rate: float = math.radians(90.0)

    #: Courbe exponentielle : 0 = lineaire, 1 = tres progressif au centre.
    #: Un expo eleve donne du pilotage fin autour du neutre sans perdre de
    #: debattement aux extremes.
    expo: float = 0.4

    #: Constante de temps du lissage (s). Filtre la trepidation du casque.
    smoothing_tau: float = 0.12

    #: Vitesse maximale de variation de la nacelle (rad/s). Protege la
    #: mecanique et evite une image inexploitable si le pilote tourne vite.
    max_gimbal_rate: float = math.radians(180.0)


@dataclass(slots=True)
class SafetyConfig:
    """Limites et failsafes. Ce sont elles qui rendent le vol rattrapable."""

    #: Delai sans nouvelle pose de tete avant de repasser en stationnaire.
    head_timeout: float = 0.5

    #: Geofence : le drone ne peut ni monter ni s'eloigner au-dela.
    max_altitude: float = 50.0
    max_distance: float = 100.0

    #: Seuils de batterie (fraction de 1.0).
    battery_warning: float = 0.30
    battery_failsafe: float = 0.15

    #: Vitesse de descente lors d'un atterrissage automatique (m/s).
    landing_speed: float = 0.8


@dataclass(slots=True)
class VideoConfig:
    """Flux video FPV."""

    width: int = 640
    height: int = 360
    fps: int = 30

    #: Champ de vision horizontal de la camera.
    camera_fov: float = math.radians(90.0)

    #: Qualite JPEG du flux. 70 est un bon compromis latence / lisibilite.
    jpeg_quality: int = 70

    #: Incruster l'OSD (horizon artificiel, cap, altitude, batterie).
    osd: bool = True


@dataclass(slots=True)
class Config:
    """Configuration complete de l'application."""

    #: ``sim`` = tout simule (aucun materiel requis), ``real`` = vrai drone.
    backend: str = "sim"
    #: ``sim`` = casque simule (souris/clavier), ``webxr`` = vrai casque.
    tracker: str = "sim"
    host: str = "127.0.0.1"
    port: int = 8000

    #: Frequence de la boucle de controle (Hz).
    loop_hz: int = 50

    #: Connexion MAVLink, utilisee uniquement si ``backend == "real"``.
    mavlink_url: str = "udp:127.0.0.1:14550"
    #: Source video reelle : index de webcam ("0") ou URL RTSP.
    camera_source: str = "0"

    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    def validate(self) -> None:
        """Verifie la coherence des reglages.

        On echoue au demarrage plutot qu'en vol : une config incoherente
        detectee a 30 m du sol est un crash, detectee ici c'est un message.
        """
        if self.backend not in {"sim", "real"}:
            raise ValueError(f"backend inconnu : {self.backend!r} (attendu sim|real)")
        if self.tracker not in {"sim", "webxr"}:
            raise ValueError(f"tracker inconnu : {self.tracker!r} (attendu sim|webxr)")
        if self.control.mode not in {"gimbal", "yaw_follow", "full"}:
            raise ValueError(
                f"control.mode inconnu : {self.control.mode!r} "
                "(attendu gimbal|yaw_follow|full)"
            )
        if not 0.0 <= self.control.expo < 1.0:
            raise ValueError("control.expo doit etre dans [0, 1[")
        if self.control.head_deadzone >= self.control.max_head_yaw:
            raise ValueError("control.head_deadzone doit rester < max_head_yaw")
        if self.control.smoothing_tau < 0.0:
            raise ValueError("control.smoothing_tau doit etre >= 0")
        if self.safety.battery_failsafe >= self.safety.battery_warning:
            raise ValueError(
                "safety.battery_failsafe doit etre < safety.battery_warning"
            )
        if self.safety.max_altitude <= 0 or self.safety.max_distance <= 0:
            raise ValueError("les limites de geofence doivent etre > 0")
        if self.loop_hz <= 0:
            raise ValueError("loop_hz doit etre > 0")
        if not 1 <= self.video.jpeg_quality <= 95:
            raise ValueError("video.jpeg_quality doit etre dans [1, 95]")
        if self.video.width <= 0 or self.video.height <= 0 or self.video.fps <= 0:
            raise ValueError("les dimensions et le fps video doivent etre > 0")

    @property
    def dt(self) -> float:
        """Pas de temps nominal de la boucle de controle."""
        return 1.0 / self.loop_hz


def _apply(target: Any, values: dict[str, Any], path: str = "") -> None:
    """Recopie ``values`` dans la dataclass ``target``, recursivement.

    Une cle inconnue leve : une faute de frappe dans le YAML doit se voir,
    pas etre ignoree en silence (sinon on croit avoir baisse une limite de
    securite alors qu'elle est restee a sa valeur par defaut).
    """
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        full = f"{path}{key}"
        if key not in known:
            raise ValueError(f"cle de configuration inconnue : {full!r}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, path=f"{full}.")
            continue
        if isinstance(value, dict):
            raise ValueError(f"{full!r} attend une valeur simple, pas un bloc")
        if key in _DEGREE_FIELDS:
            value = math.radians(float(value))
        elif isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(target, key, value)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Charge la configuration depuis un YAML, puis applique ``overrides``.

    ``overrides`` sert aux options de ligne de commande, qui priment toujours
    sur le fichier. Les valeurs ``None`` sont ignorees, ce qui permet de
    passer directement les arguments d'argparse sans les filtrer.
    """
    config = Config()

    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is not None and not source.exists():
        raise FileNotFoundError(f"fichier de configuration introuvable : {source}")
    if source.exists():
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{source} doit contenir un dictionnaire YAML")
        _apply(config, raw)

    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean:
        _apply(config, clean)

    config.validate()
    return config
