"""Interface commune a tous les drones (simule ou reel).

Le reste de l'application ne connait *que* cette interface. C'est ce qui
permet de developper et de demontrer le projet entierement sans materiel,
puis de basculer sur un vrai drone en changeant une ligne de configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ControlInput, Telemetry

__all__ = ["DroneBackend"]


class DroneBackend(ABC):
    """Contrat minimal d'un drone pilotable.

    Regles respectees par toutes les implementations :

    * ``arm`` n'est accepte qu'au sol.
    * ``disarm`` coupe les moteurs : n'est accepte qu'au sol, sauf ``force``.
    * ``step`` doit etre appelee regulierement ; c'est elle qui fait avancer
      le temps (simulation) ou qui emet les trames de commande (drone reel).
    """

    @abstractmethod
    def connect(self) -> None:
        """Etablit la liaison. Doit etre idempotente."""

    @abstractmethod
    def close(self) -> None:
        """Libere la liaison. Doit etre idempotente et ne jamais lever."""

    @abstractmethod
    def arm(self) -> bool:
        """Arme les moteurs. Renvoie ``False`` si l'etat ne le permet pas."""

    @abstractmethod
    def disarm(self, force: bool = False) -> bool:
        """Desarme. ``force=True`` coupe les moteurs meme en vol (dernier recours)."""

    @abstractmethod
    def takeoff(self, altitude: float = 2.0) -> bool:
        """Decolle jusqu'a ``altitude`` metres. Necessite d'etre arme."""

    @abstractmethod
    def land(self) -> bool:
        """Lance un atterrissage automatique."""

    @abstractmethod
    def step(self, control: ControlInput, dt: float) -> Telemetry:
        """Applique ``control`` pendant ``dt`` secondes et renvoie l'etat."""

    @property
    @abstractmethod
    def telemetry(self) -> Telemetry:
        """Dernier etat connu, sans faire avancer le temps."""

    def __enter__(self) -> "DroneBackend":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
