"""Interface commune aux sources video."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Telemetry

__all__ = ["VideoSource"]


class VideoSource(ABC):
    """Produit les images affichees dans le casque.

    Deux implementations : ``SimulatedCamera`` (image de synthese, aucun
    materiel) et ``RealCamera`` (webcam ou recepteur FPV en RTSP).

    Le contrat volontairement minimal — "donne-moi un JPEG" — est ce qui
    permettra de remplacer le transport par du WebRTC plus tard sans toucher
    au reste du projet.
    """

    @abstractmethod
    def open(self) -> None:
        """Prepare la source. Idempotente."""

    @abstractmethod
    def close(self) -> None:
        """Libere la source. Idempotente, ne leve jamais."""

    @abstractmethod
    def frame(self, telemetry: Telemetry) -> bytes:
        """Renvoie une image encodee en JPEG.

        ``telemetry`` sert au rendu (orientation de la camera) et a l'OSD.
        """
