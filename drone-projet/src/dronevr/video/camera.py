"""Camera reelle : webcam locale ou recepteur FPV en reseau.

Utilise uniquement quand ``backend: real``. OpenCV n'est importe qu'ici, et
seulement a l'ouverture : le projet doit rester utilisable sans lui.

Deux montages possibles :

* **Camera sur le drone, liaison analogique** : le recepteur video se branche
  sur un convertisseur USB, qui apparait comme une webcam. ``camera_source``
  vaut alors l'index du peripherique (``"0"``, ``"1"``...).
* **Camera IP / numerique** : le drone diffuse en RTSP. ``camera_source``
  vaut l'URL complete, par exemple ``rtsp://192.168.1.50:8554/live``.
"""

from __future__ import annotations

import io
import threading

from PIL import Image

from ..config import VideoConfig
from ..types import Telemetry
from .base import VideoSource
from .osd import draw_osd

__all__ = ["RealCamera"]

#: Image de repli quand aucune trame n'est encore arrivee.
_PLACEHOLDER_COLOR = (18, 22, 28)


class RealCamera(VideoSource):
    """Lit un flux video et le reencode en JPEG avec l'OSD.

    La lecture se fait dans un thread dedie. C'est indispensable : sur un flux
    RTSP, ``read()`` bloque jusqu'a la trame suivante, ce qui figerait la
    boucle de controle a la cadence du reseau. Ici la boucle prend toujours la
    derniere trame disponible et n'attend jamais.
    """

    def __init__(self, source: str = "0", config: VideoConfig | None = None) -> None:
        self.source = source
        self.config = config or VideoConfig()
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._frames = 0
        self._errors = 0

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"frames": self._frames, "errors": self._errors}

    def open(self) -> None:
        if self._capture is not None:
            return

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depend du materiel
            raise RuntimeError(
                "opencv-python est requis pour une camera reelle. "
                "Installer avec : pip install -e '.[real]'"
            ) from exc

        # Un ``camera_source`` entierement numerique designe un peripherique
        # local ; tout le reste est traite comme une URL.
        target = int(self.source) if self.source.isdigit() else self.source
        capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            raise RuntimeError(
                f"impossible d'ouvrir la source video {self.source!r} : "
                "verifier le branchement du recepteur ou l'URL du flux"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Tampon d'une seule trame : on veut l'image la plus recente, pas une
        # file d'images en retard. La latence prime sur la fluidite en FPV.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._reader, name="camera-reader", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # pragma: no cover - nettoyage best effort
                pass

    def _reader(self) -> None:
        """Boucle de lecture, executee dans son propre thread."""
        import cv2

        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok:
                with self._lock:
                    self._errors += 1
                # Le flux peut hoqueter ; on laisse la main plutot que de
                # marteler une source morte.
                self._stop.wait(0.05)
                continue

            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                frame = cv2.resize(
                    frame, (self.config.width, self.config.height),
                    interpolation=cv2.INTER_AREA,
                )

            # OpenCV travaille en BGR, PIL en RGB.
            image = Image.fromarray(frame[:, :, ::-1], mode="RGB")
            with self._lock:
                self._latest = image
                self._frames += 1

    def frame(self, telemetry: Telemetry) -> bytes:
        """Renvoie la derniere trame recue, OSD incruste, en JPEG."""
        with self._lock:
            image = self._latest

        if image is None:
            image = Image.new(
                "RGB", (self.config.width, self.config.height), _PLACEHOLDER_COLOR
            )
        else:
            image = image.copy()

        if self.config.osd:
            draw_osd(image, telemetry)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
        return buffer.getvalue()
