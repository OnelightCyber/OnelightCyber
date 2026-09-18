"""Enregistreur de vol ("boite noire").

Chaque vol est ecrit dans un fichier JSONL : une ligne = un instant. Le
format est volontairement bete a lire, pour pouvoir etre ouvert avec
n'importe quoi, et la conversion CSV permet de tracer les courbes dans un
tableur — ce qui est exactement ce qu'il faut pour le dossier de projet.

L'ecriture est bufferisee : la boucle de controle tourne a 50 Hz et ne doit
jamais attendre le disque.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .types import ControlInput, HeadPose, Telemetry

__all__ = ["FlightRecorder"]

#: Nombre d'echantillons gardes en memoire avant ecriture.
BUFFER_SIZE = 64

#: Colonnes du CSV exporte, dans l'ordre.
CSV_COLUMNS = (
    "t",
    "altitude",
    "north",
    "east",
    "ground_speed",
    "roll",
    "pitch",
    "yaw",
    "gimbal_yaw",
    "gimbal_pitch",
    "battery",
    "mode",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "cmd_roll",
    "cmd_pitch",
    "cmd_yaw_rate",
    "cmd_throttle",
)


class FlightRecorder:
    """Ecrit l'etat complet du systeme a chaque pas de la boucle."""

    def __init__(self, path: str | Path, sample_rate: float = 10.0) -> None:
        self.path = Path(path)
        if sample_rate <= 0.0:
            raise ValueError("sample_rate doit etre > 0")
        # On n'enregistre pas les 50 Hz de la boucle : 10 Hz suffit pour
        # tracer des courbes lisibles et divise la taille du fichier par cinq.
        self._interval = 1.0 / sample_rate
        self._buffer: list[str] = []
        self._handle = None
        self._last_write = 0.0
        self._t0 = 0.0
        self._samples = 0

    @property
    def samples(self) -> int:
        return self._samples

    def open(self) -> None:
        """Ouvre le fichier. Idempotente."""
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._t0 = time.monotonic()
        self._last_write = 0.0

    def close(self) -> None:
        """Vide le tampon et ferme. Idempotente, ne leve jamais."""
        if self._handle is None:
            return
        try:
            self._flush()
            self._handle.close()
        except OSError:
            # Un disque plein ne doit pas masquer la vraie raison de l'arret.
            pass
        finally:
            self._handle = None

    def record(
        self,
        telemetry: Telemetry,
        control: ControlInput,
        head: HeadPose,
        warnings: tuple[str, ...] = (),
    ) -> None:
        """Enregistre un echantillon, si l'intervalle est ecoule."""
        if self._handle is None:
            return

        elapsed = time.monotonic() - self._t0
        if elapsed - self._last_write < self._interval:
            return
        self._last_write = elapsed

        head_yaw, head_pitch, head_roll = head.as_degrees()
        sample = {
            "t": round(elapsed, 3),
            **telemetry.to_dict(),
            "head": {
                "yaw": round(head_yaw, 1),
                "pitch": round(head_pitch, 1),
                "roll": round(head_roll, 1),
            },
            "cmd": {
                "roll": round(control.roll, 3),
                "pitch": round(control.pitch, 3),
                "yaw_rate": round(control.yaw_rate, 3),
                "throttle": round(control.throttle, 3),
            },
        }
        if warnings:
            sample["warnings"] = list(warnings)

        self._buffer.append(json.dumps(sample, ensure_ascii=False))
        self._samples += 1
        if len(self._buffer) >= BUFFER_SIZE:
            self._flush()

    def _flush(self) -> None:
        if self._handle is None or not self._buffer:
            return
        self._handle.write("\n".join(self._buffer) + "\n")
        self._handle.flush()
        self._buffer.clear()

    def __enter__(self) -> "FlightRecorder":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def to_csv(jsonl_path: str | Path, csv_path: str | Path) -> int:
        """Convertit un enregistrement JSONL en CSV. Renvoie le nombre de lignes.

        Le CSV est ce que comprennent LibreOffice, Excel et les tableurs en
        ligne : c'est le format a utiliser pour les graphiques du dossier.
        """
        source, destination = Path(jsonl_path), Path(csv_path)
        rows = 0

        with source.open(encoding="utf-8") as src, destination.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            dst.write(",".join(CSV_COLUMNS) + "\n")
            for line in src:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                position = sample.get("position", {})
                head = sample.get("head", {})
                command = sample.get("cmd", {})
                values = {
                    "t": sample.get("t"),
                    "altitude": sample.get("altitude"),
                    "north": position.get("north"),
                    "east": position.get("east"),
                    "ground_speed": sample.get("ground_speed"),
                    "roll": sample.get("roll"),
                    "pitch": sample.get("pitch"),
                    "yaw": sample.get("yaw"),
                    "gimbal_yaw": sample.get("gimbal_yaw"),
                    "gimbal_pitch": sample.get("gimbal_pitch"),
                    "battery": sample.get("battery"),
                    "mode": sample.get("mode"),
                    "head_yaw": head.get("yaw"),
                    "head_pitch": head.get("pitch"),
                    "head_roll": head.get("roll"),
                    "cmd_roll": command.get("roll"),
                    "cmd_pitch": command.get("pitch"),
                    "cmd_yaw_rate": command.get("yaw_rate"),
                    "cmd_throttle": command.get("throttle"),
                }
                dst.write(
                    ",".join("" if values[c] is None else str(values[c]) for c in CSV_COLUMNS)
                    + "\n"
                )
                rows += 1

        return rows
