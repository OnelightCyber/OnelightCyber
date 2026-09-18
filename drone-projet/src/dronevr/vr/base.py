"""Interface commune aux sources de position de tete."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import HeadPose

__all__ = ["HeadTracker"]


class HeadTracker(ABC):
    """Source d'orientation de la tete du pilote.

    Deux implementations : ``SimulatedTracker`` (aucun materiel) et
    ``WebXRTracker`` (vrai casque, poses recues du navigateur). La boucle de
    controle ne fait pas la difference entre les deux.
    """

    @abstractmethod
    def read(self) -> HeadPose:
        """Renvoie la derniere pose connue, sans bloquer.

        Ne bloque jamais : la boucle de controle tourne a frequence fixe et ne
        doit pas dependre de la cadence du casque. Si aucune pose fraiche n'est
        arrivee, on renvoie la derniere connue et c'est le watchdog de
        ``SafetyMonitor`` qui decide si elle est trop vieille.
        """

    def start(self) -> None:
        """Demarre l'acquisition. Par defaut : rien a faire."""

    def stop(self) -> None:
        """Arrete l'acquisition. Par defaut : rien a faire."""
