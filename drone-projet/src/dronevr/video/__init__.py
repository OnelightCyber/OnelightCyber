"""Sources video : image de synthese ou camera reelle."""

from .base import VideoSource
from .osd import draw_osd
from .simulated import SimulatedCamera

__all__ = ["SimulatedCamera", "VideoSource", "create_camera", "draw_osd"]


def create_camera(config) -> VideoSource:
    """Instancie la source video correspondant a ``config.backend``.

    OpenCV n'est importe que si l'on demande une vraie camera : le projet
    doit rester installable et testable sans lui.
    """
    if config.backend == "sim":
        return SimulatedCamera(config.video)

    if config.backend == "real":
        from .camera import RealCamera

        return RealCamera(source=config.camera_source, config=config.video)

    raise ValueError(f"backend inconnu : {config.backend!r}")
