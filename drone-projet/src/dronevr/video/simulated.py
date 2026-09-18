"""Camera simulee : rendu 3D de ce que verrait le drone.

L'image n'est pas un decor plaque, c'est un vrai rendu par lancer de rayons.
Pour chaque pixel on calcule la direction du rayon dans le repere de la
camera, on l'oriente selon l'attitude du drone *et* de la nacelle, puis on
cherche son intersection avec le sol.

C'est ce qui rend la demo credible sans materiel : quand on tourne la tete,
la perspective, le quadrillage au sol et les balises bougent correctement,
exactement comme le ferait une vraie camera.

Repere NED, donc ``z`` positif vers le bas et le sol est le plan ``z = 0``.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from ..config import VideoConfig
from ..types import Telemetry
from .base import VideoSource
from .osd import draw_osd

__all__ = ["SimulatedCamera"]

#: Pas du quadrillage au sol, en metres. 5 m donne une bonne perception de la
#: vitesse sans saturer l'image.
GRID_SPACING = 5.0

#: Distance au-dela de laquelle le quadrillage s'efface dans la brume. Sans
#: ce fondu, les lignes lointaines creent un moire illisible.
FOG_DISTANCE = 120.0

#: Couleurs du decor.
SKY_TOP = np.array([46, 88, 150], dtype=np.float32)
SKY_HORIZON = np.array([168, 196, 224], dtype=np.float32)
GROUND_NEAR = np.array([58, 92, 52], dtype=np.float32)
GROUND_FAR = np.array([120, 138, 108], dtype=np.float32)
GRID_COLOR = np.array([196, 214, 186], dtype=np.float32)
AXIS_COLOR = np.array([235, 170, 120], dtype=np.float32)

#: Balises au sol (nord, est, hauteur, couleur). Elles donnent au pilote des
#: reperes fixes : sans elles, une rotation sur place est invisible.
BEACONS = (
    (40.0, 0.0, 8.0, (220, 70, 70)),
    (0.0, 40.0, 8.0, (70, 140, 220)),
    (-35.0, -35.0, 6.0, (230, 200, 80)),
    (60.0, 60.0, 10.0, (200, 100, 220)),
)


def _camera_axes(yaw: float, pitch: float, roll: float):
    """Renvoie les axes ``(droite, bas, avant)`` de la camera dans le repere NED.

    Construits dans cet ordre : l'avant vient du cap et du site, la droite est
    forcement horizontale (produit vectoriel avec la verticale), le bas
    complete le triedre. Le roulis fait ensuite tourner droite et bas autour
    de l'avant, ce qui ne change pas la direction de visee.
    """
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    # Tangage positif = regarder vers le haut, donc composante ``z`` negative.
    forward = np.array([cos_p * cos_y, cos_p * sin_y, -sin_p], dtype=np.float32)
    right = np.array([-sin_y, cos_y, 0.0], dtype=np.float32)
    down = np.cross(forward, right)

    if roll:
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        right, down = right * cos_r + down * sin_r, down * cos_r - right * sin_r

    return right, down, forward


class SimulatedCamera(VideoSource):
    """Genere le flux FPV par rendu 3D.

    La grille de rayons dans le repere camera ne depend que de la resolution
    et du champ de vision : elle est calculee une fois a l'ouverture, et
    chaque image ne coute plus qu'une rotation et une intersection.
    """

    def __init__(self, config: VideoConfig | None = None) -> None:
        self.config = config or VideoConfig()
        self._rays = None
        self._sky = None

    def open(self) -> None:
        if self._rays is not None:
            return

        width, height = self.config.width, self.config.height

        # Modele stenope : la focale se deduit du champ de vision horizontal.
        focal = (width / 2.0) / math.tan(self.config.camera_fov / 2.0)

        u = (np.arange(width, dtype=np.float32) - width / 2.0 + 0.5) / focal
        v = (np.arange(height, dtype=np.float32) - height / 2.0 + 0.5) / focal
        grid_u, grid_v = np.meshgrid(u, v)

        # (droite, bas, avant) pour chaque pixel, avant toute rotation.
        rays = np.stack(
            [grid_u, grid_v, np.ones_like(grid_u)], axis=-1
        ).reshape(-1, 3)
        self._rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def close(self) -> None:
        self._rays = None
        self._sky = None

    def frame(self, telemetry: Telemetry) -> bytes:
        """Rend une image et l'encode en JPEG."""
        if self._rays is None:
            self.open()

        image = Image.fromarray(self._render(telemetry), mode="RGB")

        if self.config.osd:
            draw_osd(image, telemetry)

        buffer = io.BytesIO()
        image.save(
            buffer, format="JPEG", quality=self.config.jpeg_quality, optimize=False
        )
        return buffer.getvalue()

    def render_image(self, telemetry: Telemetry) -> Image.Image:
        """Rend une image PIL sans encodage. Pratique pour les tests."""
        if self._rays is None:
            self.open()
        image = Image.fromarray(self._render(telemetry), mode="RGB")
        if self.config.osd:
            draw_osd(image, telemetry)
        return image

    # ----- rendu ----------------------------------------------------------

    def _camera_orientation(self, telemetry: Telemetry) -> tuple[float, float, float]:
        """Orientation reelle de la camera = drone + nacelle.

        La nacelle 3 axes annule le roulis du drone : c'est tout son interet,
        et c'est aussi ce qui evite au pilote d'avoir mal au coeur. Le tangage
        et le lacet, eux, s'ajoutent a ceux du drone.
        """
        return (
            telemetry.yaw + telemetry.gimbal_yaw,
            telemetry.pitch + telemetry.gimbal_pitch,
            0.0,
        )

    def _render(self, telemetry: Telemetry) -> np.ndarray:
        width, height = self.config.width, self.config.height
        yaw, pitch, roll = self._camera_orientation(telemetry)
        right, down, forward = _camera_axes(yaw, pitch, roll)

        # Passage des rayons du repere camera au repere NED.
        basis = np.stack([right, down, forward], axis=0)
        directions = self._rays @ basis

        altitude = max(telemetry.altitude, 0.05)
        pixels = self._render_sky(directions)
        self._render_ground(pixels, directions, telemetry, altitude)
        self._render_beacons(pixels, telemetry, right, down, forward, width, height)

        return pixels.reshape(height, width, 3).astype(np.uint8)

    def _render_sky(self, directions: np.ndarray) -> np.ndarray:
        """Degrade du ciel, du bleu profond au zenith au blanc a l'horizon."""
        # ``z`` negatif = vers le haut : 0 a l'horizon, 1 au zenith.
        elevation = np.clip(-directions[:, 2], 0.0, 1.0)[:, None]
        return SKY_HORIZON + (SKY_TOP - SKY_HORIZON) * np.sqrt(elevation)

    def _render_ground(
        self,
        pixels: np.ndarray,
        directions: np.ndarray,
        telemetry: Telemetry,
        altitude: float,
    ) -> None:
        """Intersecte chaque rayon avec le plan du sol et y dessine la grille."""
        dz = directions[:, 2]

        # Un rayon ne touche le sol que s'il descend. Le seuil evite la
        # division par zero et les distances aberrantes pres de l'horizon.
        hits = dz > 1e-4
        if not hits.any():
            return

        distance = altitude / dz[hits]
        north = telemetry.position.x + directions[hits, 0] * distance
        east = telemetry.position.y + directions[hits, 1] * distance

        fog = np.clip(distance / FOG_DISTANCE, 0.0, 1.0)[:, None]
        ground = GROUND_NEAR + (GROUND_FAR - GROUND_NEAR) * fog

        # Largeur de trait proportionnelle a la distance : la ligne garde une
        # epaisseur constante a l'ecran, ce qui supprime le crenelage.
        half_width = np.clip(distance * 0.012, 0.04, 2.0)

        to_line_n = np.abs((north + GRID_SPACING / 2.0) % GRID_SPACING - GRID_SPACING / 2.0)
        to_line_e = np.abs((east + GRID_SPACING / 2.0) % GRID_SPACING - GRID_SPACING / 2.0)
        on_grid = np.minimum(to_line_n, to_line_e) < half_width

        # Fondu de la grille avec la distance, sinon l'horizon devient un mur.
        strength = (1.0 - fog[:, 0]) * on_grid
        ground = ground + (GRID_COLOR - ground) * strength[:, None]

        # Les axes passant par le point de decollage sont mis en evidence :
        # ce sont les seuls reperes absolus dont dispose le pilote.
        on_axis = (np.abs(north) < half_width * 2.0) | (np.abs(east) < half_width * 2.0)
        axis_strength = ((1.0 - fog[:, 0]) * on_axis)[:, None]
        ground = ground + (AXIS_COLOR - ground) * axis_strength

        pixels[hits] = ground

    def _render_beacons(
        self,
        pixels: np.ndarray,
        telemetry: Telemetry,
        right: np.ndarray,
        down: np.ndarray,
        forward: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        """Projette les balises verticales dans l'image.

        On projette (monde -> ecran) au lieu de lancer des rayons : pour
        quelques objets c'est bien plus rapide, et la geometrie est la meme.
        """
        focal = (width / 2.0) / math.tan(self.config.camera_fov / 2.0)
        origin = np.array(
            [telemetry.position.x, telemetry.position.y, telemetry.position.z],
            dtype=np.float32,
        )
        image = pixels.reshape(height, width, 3)

        for north, east, tall, color in BEACONS:
            base = np.array([north, east, 0.0], dtype=np.float32) - origin
            top = np.array([north, east, -tall], dtype=np.float32) - origin

            depth_base = float(base @ forward)
            depth_top = float(top @ forward)
            # Entierement derriere la camera : rien a dessiner.
            if depth_base <= 0.5 and depth_top <= 0.5:
                continue

            def project(point: np.ndarray, depth: float) -> tuple[float, float]:
                return (
                    width / 2.0 + focal * float(point @ right) / depth,
                    height / 2.0 + focal * float(point @ down) / depth,
                )

            depth_base = max(depth_base, 0.5)
            depth_top = max(depth_top, 0.5)
            x_base, y_base = project(base, depth_base)
            x_top, y_top = project(top, depth_top)

            # Largeur apparente d'un mat de 40 cm, jamais moins d'un pixel.
            half = max(1.0, focal * 0.4 / depth_base / 2.0)
            x0 = int(max(0, min(x_base, x_top) - half))
            x1 = int(min(width, max(x_base, x_top) + half + 1))
            y0 = int(max(0, min(y_top, y_base)))
            y1 = int(min(height, max(y_top, y_base) + 1))

            if x1 <= x0 or y1 <= y0:
                continue

            fade = float(np.clip(1.0 - depth_base / FOG_DISTANCE, 0.15, 1.0))
            tint = np.array(color, dtype=np.float32) * fade
            patch = image[y0:y1, x0:x1]
            image[y0:y1, x0:x1] = patch + (tint - patch) * fade
