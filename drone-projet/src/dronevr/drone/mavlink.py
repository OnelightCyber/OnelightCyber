"""Backend pour un vrai drone, via le protocole MAVLink.

Ce module n'est utilise que lorsque ``backend: real`` est configure. Il n'est
volontairement pas importe au demarrage : le projet doit tourner entierement
sans ``pymavlink`` installe et sans materiel.

Compatible ArduPilot et PX4. Les consignes d'attitude sont envoyees en
``SET_ATTITUDE_TARGET``, la nacelle en ``MAV_CMD_DO_MOUNT_CONTROL``.

ATTENTION : ce code n'a pas pu etre essaye sur un drone reel. Avant tout vol,
suivre la procedure de ``docs/MATERIEL.md`` : essai en SITL, puis helices
DEMONTEES, puis vol en exterieur degage.
"""

from __future__ import annotations

import math
import time

from ..config import ControlConfig, SafetyConfig
from ..types import ControlInput, FlightMode, Telemetry, Vec3, clamp, wrap_pi
from .base import DroneBackend

__all__ = ["MavlinkDrone"]

#: Delai d'attente du premier battement de coeur, en secondes.
HEARTBEAT_TIMEOUT = 30.0

#: Masque de ``SET_ATTITUDE_TARGET`` : on impose le quaternion d'attitude et
#: la vitesse de lacet, on laisse le controleur gerer roulis et tangage.
IGNORE_ROLL_RATE = 1
IGNORE_PITCH_RATE = 2
ATTITUDE_TYPE_MASK = IGNORE_ROLL_RATE | IGNORE_PITCH_RATE


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convertit des angles d'Euler ZYX en quaternion ``[w, x, y, z]``.

    MAVLink ne prend pas d'angles d'Euler pour l'attitude : il faut un
    quaternion, ce qui evite le blocage de cardan a la verticale.
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


class MavlinkDrone(DroneBackend):
    """Pilote un drone reel par MAVLink.

    L'interface est strictement celle de ``SimulatedDrone`` : tout ce qui a
    ete mis au point en simulation fonctionne ici sans modification.
    """

    def __init__(
        self,
        url: str = "udp:127.0.0.1:14550",
        control: ControlConfig | None = None,
        safety: SafetyConfig | None = None,
    ) -> None:
        self.url = url
        self._control = control or ControlConfig()
        self._safety = safety or SafetyConfig()
        self._link = None
        self._telemetry = Telemetry()
        self._home_set = False
        self._home = Vec3()

    # ----- cycle de vie ---------------------------------------------------

    def connect(self) -> None:
        """Ouvre la liaison et attend le premier battement de coeur.

        Sans battement de coeur, on ne connait ni le ``system id`` ni le
        ``component id`` de l'autopilote : impossible d'adresser la moindre
        commande. On echoue donc plutot que de voler a l'aveugle.
        """
        if self._link is not None:
            return

        try:
            from pymavlink import mavutil
        except ImportError as exc:  # pragma: no cover - depend du materiel
            raise RuntimeError(
                "pymavlink est requis pour backend: real. "
                "Installer avec : pip install -e '.[real]'"
            ) from exc

        link = mavutil.mavlink_connection(self.url)
        if link.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT) is None:
            link.close()
            raise TimeoutError(
                f"aucun battement de coeur sur {self.url} apres "
                f"{HEARTBEAT_TIMEOUT:.0f} s : verifier la liaison et le debit"
            )
        self._link = link
        self._request_data_streams()

    def close(self) -> None:
        """Ferme la liaison sans jamais lever : appelee depuis les ``finally``."""
        if self._link is None:
            return
        try:
            self._link.close()
        except Exception:  # pragma: no cover - nettoyage best effort
            pass
        finally:
            self._link = None

    def _require_link(self):
        if self._link is None:
            raise RuntimeError("drone non connecte : appeler connect() d'abord")
        return self._link

    def _request_data_streams(self) -> None:
        """Demande a l'autopilote d'emettre position et attitude a 20 Hz."""
        from pymavlink import mavutil

        link = self._require_link()
        link.mav.request_data_stream_send(
            link.target_system,
            link.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20,
            1,
        )

    # ----- commandes ------------------------------------------------------

    def _send_command(self, command: int, *params: float) -> bool:
        """Envoie un ``COMMAND_LONG`` et attend son accuse de reception."""
        from pymavlink import mavutil

        link = self._require_link()
        values = list(params) + [0.0] * (7 - len(params))
        link.mav.command_long_send(
            link.target_system, link.target_component, command, 0, *values
        )
        ack = link.recv_match(type="COMMAND_ACK", blocking=True, timeout=3.0)
        if ack is None or ack.command != command:
            return False
        return ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED

    def arm(self) -> bool:
        from pymavlink import mavutil

        if self._telemetry.mode is not FlightMode.DISARMED:
            return False
        if self._telemetry.battery <= self._safety.battery_failsafe:
            return False
        return self._send_command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0
        )

    def disarm(self, force: bool = False) -> bool:
        from pymavlink import mavutil

        # 21196 est le code "force" reconnu par ArduPilot : il autorise le
        # desarmement en vol. A ne jamais utiliser autrement qu'au sol.
        magic = 21196.0 if force else 0.0
        return self._send_command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0, magic
        )

    def takeoff(self, altitude: float = 2.0) -> bool:
        from pymavlink import mavutil

        if self._telemetry.mode is not FlightMode.ARMED:
            return False
        target = clamp(altitude, 0.5, self._safety.max_altitude)
        return self._send_command(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, target
        )

    def land(self) -> bool:
        from pymavlink import mavutil

        return self._send_command(mavutil.mavlink.MAV_CMD_NAV_LAND)

    # ----- boucle ---------------------------------------------------------

    def step(self, control: ControlInput, dt: float) -> Telemetry:
        """Emet les consignes puis consomme la telemetrie recue."""
        self._send_attitude(control)
        self._send_gimbal(control)
        self._pump_telemetry()
        return self._telemetry

    def _send_attitude(self, control: ControlInput) -> None:
        """Traduit les consignes normalisees en attitude MAVLink."""
        link = self._require_link()
        cfg = self._control

        roll = clamp(control.roll, -1.0, 1.0) * cfg.max_tilt
        # Comme en simulation : avancer = piquer du nez.
        pitch = -clamp(control.pitch, -1.0, 1.0) * cfg.max_tilt
        yaw_rate = clamp(control.yaw_rate, -1.0, 1.0) * cfg.max_yaw_rate

        link.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            link.target_system,
            link.target_component,
            ATTITUDE_TYPE_MASK,
            _euler_to_quaternion(roll, pitch, self._telemetry.yaw),
            0.0,
            0.0,
            yaw_rate,
            clamp(control.throttle, 0.0, 1.0),
        )

    def _send_gimbal(self, control: ControlInput) -> None:
        """Pointe la nacelle. MAVLink attend des degres ici, pas des radians."""
        from pymavlink import mavutil

        link = self._require_link()
        cfg = self._control
        pitch_deg = math.degrees(
            clamp(control.gimbal_pitch, -cfg.max_gimbal_pitch, cfg.max_gimbal_pitch)
        )
        yaw_deg = math.degrees(
            clamp(control.gimbal_yaw, -cfg.max_gimbal_yaw, cfg.max_gimbal_yaw)
        )
        link.mav.command_long_send(
            link.target_system,
            link.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
            0,
            pitch_deg,
            0.0,
            yaw_deg,
            0.0,
            0.0,
            0.0,
            mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING,
        )

    def _pump_telemetry(self) -> None:
        """Vide la file des messages recus et met a jour l'etat.

        ``blocking=False`` est essentiel : la boucle de controle tourne a
        50 Hz et ne doit jamais attendre le reseau.
        """
        link = self._require_link()
        telemetry = self._telemetry

        while True:
            message = link.recv_match(
                type=[
                    "ATTITUDE",
                    "LOCAL_POSITION_NED",
                    "SYS_STATUS",
                    "HEARTBEAT",
                    "MOUNT_ORIENTATION",
                ],
                blocking=False,
            )
            if message is None:
                break
            telemetry = self._merge(telemetry, message)

        self._telemetry = telemetry

    def _merge(self, telemetry: Telemetry, message) -> Telemetry:
        """Fusionne un message MAVLink dans l'etat courant."""
        from dataclasses import replace

        from pymavlink import mavutil

        kind = message.get_type()

        if kind == "ATTITUDE":
            return replace(
                telemetry,
                roll=message.roll,
                pitch=message.pitch,
                yaw=wrap_pi(message.yaw),
                timestamp=time.monotonic(),
            )

        if kind == "LOCAL_POSITION_NED":
            # L'autopilote donne deja du NED relatif a son origine EKF ; on
            # rebase sur le point de decollage pour que la geofence mesure
            # bien une distance au point de depart.
            raw = Vec3(message.x, message.y, message.z)
            if not self._home_set:
                self._home = raw
                self._home_set = True
            return replace(
                telemetry,
                position=raw - self._home,
                velocity=Vec3(message.vx, message.vy, message.vz),
                timestamp=time.monotonic(),
            )

        if kind == "SYS_STATUS":
            # -1 signifie "non mesure" : on garde alors la valeur precedente
            # plutot que de declencher un failsafe sur une donnee absente.
            remaining = message.battery_remaining
            if remaining < 0:
                return telemetry
            return replace(telemetry, battery=clamp(remaining / 100.0, 0.0, 1.0))

        if kind == "MOUNT_ORIENTATION":
            return replace(
                telemetry,
                gimbal_pitch=math.radians(message.pitch),
                gimbal_yaw=math.radians(message.yaw),
            )

        if kind == "HEARTBEAT":
            armed = bool(
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            if not armed:
                return replace(telemetry, mode=FlightMode.DISARMED)
            if telemetry.mode is FlightMode.DISARMED:
                return replace(telemetry, mode=FlightMode.ARMED)
            # Au-dela de 50 cm on considere le drone reellement en vol.
            if telemetry.mode is FlightMode.ARMED and telemetry.altitude > 0.5:
                return replace(telemetry, mode=FlightMode.FLYING)

        return telemetry

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry
