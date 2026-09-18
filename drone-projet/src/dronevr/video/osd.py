"""Incrustation des informations de vol sur l'image (OSD).

Reprend les codes d'un OSD FPV reel : horizon artificiel au centre, ruban de
cap en haut, altitude et vitesse sur les cotes, batterie et mode en bas. Le
pilote ne doit jamais avoir a quitter l'image des yeux.
"""

from __future__ import annotations

import math

from PIL import ImageDraw

from ..types import FlightMode, Telemetry

__all__ = ["draw_osd"]

#: Vert phosphore des OSD analogiques, lisible sur ciel comme sur sol.
OSD_GREEN = (120, 255, 140)
OSD_WARN = (255, 190, 60)
OSD_ALERT = (255, 80, 80)
OSD_SHADOW = (0, 0, 0)

#: Couleur du mode de vol selon l'etat.
MODE_COLORS = {
    FlightMode.DISARMED: (170, 170, 170),
    FlightMode.ARMED: OSD_WARN,
    FlightMode.FLYING: OSD_GREEN,
    FlightMode.LANDING: OSD_WARN,
    FlightMode.FAILSAFE: OSD_ALERT,
}

#: Points cardinaux affiches sur le ruban de cap.
CARDINALS = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SO", 270: "O", 315: "NO"}


def _text(draw: ImageDraw.ImageDraw, xy, label: str, color, anchor: str = "la") -> None:
    """Ecrit avec une ombre portee.

    Sans ombre, le texte vert disparait des qu'il passe sur de l'herbe : c'est
    exactement le moment ou on a besoin de lire l'altitude.
    """
    x, y = xy
    draw.text((x + 1, y + 1), label, fill=OSD_SHADOW, anchor=anchor)
    draw.text((x, y), label, fill=color, anchor=anchor)


def _draw_horizon_ladder(
    draw: ImageDraw.ImageDraw, width: int, height: int, telemetry: Telemetry
) -> None:
    """Echelle de tangage et reticule central, inclines selon le roulis."""
    cx, cy = width / 2.0, height / 2.0
    roll, pitch = telemetry.roll, telemetry.pitch

    # ~4 px par degre : compromis lisibilite / encombrement.
    px_per_deg = height / 90.0
    cos_r, sin_r = math.cos(roll), math.sin(roll)

    for deg in range(-30, 31, 10):
        offset = (math.degrees(pitch) - deg) * px_per_deg
        half = 34 if deg else 56
        for side in (-1, 1):
            x0 = side * 12
            x1 = side * half
            # Rotation du segment autour du centre pour suivre le roulis.
            ax = cx + x0 * cos_r - offset * sin_r
            ay = cy + x0 * sin_r + offset * cos_r
            bx = cx + x1 * cos_r - offset * sin_r
            by = cy + x1 * sin_r + offset * cos_r
            if -20 < ay < height + 20:
                draw.line([(ax, ay), (bx, by)], fill=OSD_GREEN, width=1)
        if deg and -20 < cy + offset * cos_r < height + 20:
            lx = cx + (half + 12) * cos_r - offset * sin_r
            ly = cy + (half + 12) * sin_r + offset * cos_r
            _text(draw, (lx, ly), f"{deg:+d}", OSD_GREEN, anchor="mm")

    # Reticule fixe : repere l'axe de la camera, pas celui du drone.
    draw.line([(cx - 10, cy), (cx - 3, cy)], fill=OSD_GREEN, width=2)
    draw.line([(cx + 3, cy), (cx + 10, cy)], fill=OSD_GREEN, width=2)
    draw.point((cx, cy), fill=OSD_GREEN)


def _draw_heading_tape(
    draw: ImageDraw.ImageDraw, width: int, telemetry: Telemetry
) -> None:
    """Ruban de cap defilant en haut de l'image."""
    cx = width / 2.0
    heading = telemetry.heading_deg
    px_per_deg = 2.2
    span = int((width / 2.0) / px_per_deg) + 10

    for delta in range(-span, span + 1):
        deg = int(round(heading)) + delta
        if deg % 15:
            continue
        x = cx + delta * px_per_deg
        if not 4 <= x <= width - 4:
            continue
        normalized = deg % 360
        label = CARDINALS.get(normalized)
        if label:
            draw.line([(x, 14), (x, 21)], fill=OSD_GREEN, width=1)
            _text(draw, (x, 26), label, OSD_GREEN, anchor="ma")
        elif normalized % 30 == 0:
            draw.line([(x, 16), (x, 21)], fill=OSD_GREEN, width=1)
            _text(draw, (x, 25), str(normalized), OSD_GREEN, anchor="ma")
        else:
            draw.line([(x, 18), (x, 21)], fill=OSD_GREEN, width=1)

    _text(draw, (cx, 2), f"{heading:03.0f}", OSD_GREEN, anchor="ma")
    draw.polygon([(cx - 5, 12), (cx + 5, 12), (cx, 17)], outline=OSD_GREEN)


def draw_osd(
    image,
    telemetry: Telemetry,
    warnings: tuple[str, ...] = (),
    head_yaw: float = 0.0,
) -> None:
    """Incruste l'OSD complet sur ``image`` (modifiee sur place)."""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    _draw_horizon_ladder(draw, width, height, telemetry)
    _draw_heading_tape(draw, width, telemetry)

    _text(draw, (10, height // 2 - 24), f"{telemetry.altitude:5.1f} m", OSD_GREEN)
    _text(draw, (10, height // 2 - 8), f"{telemetry.ground_speed:5.1f} m/s", OSD_GREEN)
    _text(draw, (10, height // 2 + 8), f"{telemetry.distance_from_home:5.0f} m", OSD_GREEN)

    battery = telemetry.battery
    color = OSD_GREEN if battery > 0.3 else (OSD_WARN if battery > 0.15 else OSD_ALERT)
    _text(draw, (width - 10, height - 18), f"{battery * 100:3.0f} %", color, anchor="ra")

    _text(
        draw,
        (10, height - 18),
        telemetry.mode.value.upper(),
        MODE_COLORS.get(telemetry.mode, OSD_GREEN),
    )

    # Indicateur de nacelle : montre de combien la camera est decalee par
    # rapport a l'axe du drone. Sans lui, le pilote perd le nord des qu'il a
    # la tete tournee.
    if abs(telemetry.gimbal_yaw) > math.radians(1.0):
        _text(
            draw,
            (width - 10, height // 2),
            f"CAM {math.degrees(telemetry.gimbal_yaw):+.0f}",
            OSD_WARN,
            anchor="ra",
        )

    for index, message in enumerate(warnings[:3]):
        _text(
            draw,
            (width / 2.0, height - 46 + index * 14),
            message.upper(),
            OSD_ALERT,
            anchor="ma",
        )
