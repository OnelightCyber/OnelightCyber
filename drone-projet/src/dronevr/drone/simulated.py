"""Drone simule : modele physique simple mais coherent.

Le but n'est pas la fidelite d'un simulateur professionnel, mais d'avoir un
comportement *juste qualitativement* : le drone s'incline pour avancer, il
prend de la vitesse puis se stabilise a cause de la trainee, il tombe si on
coupe les gaz, et la batterie se vide plus vite quand on monte.

Repere NED, angles d'Euler ZYX (lacet, tangage, roulis), convention
aeronautique : tangage positif = nez en l'air, roulis positif = aile droite
en bas.
"""

from __future__ import annotations

import math

from ..config import ControlConfig, SafetyConfig
from ..types import (
    ControlInput,
    FlightMode,
    Telemetry,
    Vec3,
    clamp,
    wrap_pi,
)
from .base import DroneBackend

__all__ = ["SimulatedDrone"]

#: Acceleration de la pesanteur (m/s^2).
GRAVITY = 9.81

#: Constante de temps de la boucle d'attitude. Un vrai quadricoptere rejoint
#: son assiette en une fraction de seconde ; 0.15 s donne un ressenti realiste.
ATTITUDE_TAU = 0.15

#: Trainee lineaire. Fixe la vitesse maximale : a 25 deg d'inclinaison,
#: l'acceleration vaut g*tan(25) ~ 4.6 m/s^2, soit ~15 m/s en palier.
DRAG_HORIZONTAL = 0.30
DRAG_VERTICAL = 0.50

#: Autonomie a plein regime, en secondes (~7 min, realiste pour un FPV).
BATTERY_ENDURANCE = 420.0

#: Vitesse verticale maximale commandee par le manche des gaz (m/s).
MAX_CLIMB_RATE = 3.0

#: Poussee maximale, exprimee en acceleration. Deux fois la pesanteur donne
#: un rapport poids/poussee de 2:1, typique d'un quadricoptere de loisir.
MAX_THRUST = 2.0 * GRAVITY

#: Gains de l'asservissement en cascade altitude -> vitesse verticale -> poussee.
ALTITUDE_P = 1.2
CLIMB_RATE_P = 3.0

#: Constante de temps de la nacelle.
GIMBAL_TAU = 0.08

#: En dessous de cette altitude, le drone est considere pose.
GROUND_EPSILON = 0.05


class SimulatedDrone(DroneBackend):
    """Modele de quadricoptere suffisant pour developper et demontrer.

    Toute la logique d'etat (armement, decollage, atterrissage) est ici, donc
    l'application se comporte exactement pareil en simulation et en reel.
    """

    def __init__(
        self,
        control: ControlConfig | None = None,
        safety: SafetyConfig | None = None,
    ) -> None:
        self._control = control or ControlConfig()
        self._safety = safety or SafetyConfig()

        self._position = Vec3(0.0, 0.0, 0.0)
        self._velocity = Vec3(0.0, 0.0, 0.0)
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._gimbal_yaw = 0.0
        self._gimbal_pitch = 0.0
        self._battery = 1.0
        self._mode = FlightMode.DISARMED
        self._target_altitude = 0.0
        self._connected = False

    # ----- cycle de vie ---------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    # ----- commandes ------------------------------------------------------

    @property
    def on_ground(self) -> bool:
        return -self._position.z <= GROUND_EPSILON

    def arm(self) -> bool:
        """Arme uniquement au sol et avec assez de batterie."""
        if self._mode is not FlightMode.DISARMED or not self.on_ground:
            return False
        if self._battery <= self._safety.battery_failsafe:
            return False
        self._mode = FlightMode.ARMED
        return True

    def disarm(self, force: bool = False) -> bool:
        if not force and not self.on_ground:
            return False
        self._mode = FlightMode.DISARMED
        self._velocity = Vec3(0.0, 0.0, 0.0)
        self._target_altitude = 0.0
        return True

    def takeoff(self, altitude: float = 2.0) -> bool:
        if self._mode is not FlightMode.ARMED:
            return False
        self._target_altitude = clamp(altitude, 0.5, self._safety.max_altitude)
        self._mode = FlightMode.FLYING
        return True

    def land(self) -> bool:
        if self._mode not in {FlightMode.FLYING, FlightMode.FAILSAFE}:
            return False
        self._mode = FlightMode.LANDING
        return True

    # ----- simulation -----------------------------------------------------

    def step(self, control: ControlInput, dt: float) -> Telemetry:
        """Fait avancer la simulation de ``dt`` secondes."""
        if dt <= 0.0:
            return self.telemetry

        self._step_gimbal(control, dt)

        if self._mode is FlightMode.DISARMED:
            # Moteurs coupes : le drone reste pose. On ne simule pas la chute
            # libre, un drone desarme est par definition au sol ici.
            self._velocity = Vec3(0.0, 0.0, 0.0)
            return self.telemetry

        if self._mode is FlightMode.ARMED:
            # Moteurs qui tournent au ralenti : consommation, pas de vol.
            self._drain_battery(0.15, dt)
            return self.telemetry

        self._step_attitude(control, dt)
        thrust = self._compute_thrust(control)
        self._step_translation(thrust, dt)
        self._drain_battery(thrust / MAX_THRUST, dt)
        self._handle_ground_contact()

        return self.telemetry

    def _step_gimbal(self, control: ControlInput, dt: float) -> None:
        """La nacelle rejoint sa consigne avec un leger retard mecanique."""
        alpha = dt / (GIMBAL_TAU + dt)
        cfg = self._control
        target_yaw = clamp(control.gimbal_yaw, -cfg.max_gimbal_yaw, cfg.max_gimbal_yaw)
        target_pitch = clamp(
            control.gimbal_pitch, -cfg.max_gimbal_pitch, cfg.max_gimbal_pitch
        )
        self._gimbal_yaw += alpha * (target_yaw - self._gimbal_yaw)
        self._gimbal_pitch += alpha * (target_pitch - self._gimbal_pitch)

    def _compute_thrust(self, control: ControlInput) -> float:
        """Calcule la poussee par un asservissement en cascade.

        C'est le mode "maintien d'altitude" des vrais controleurs de vol :
        le manche des gaz ne commande pas une puissance moteur mais une
        *vitesse verticale*. Deux boucles imbriquees :

        ``altitude -> vitesse verticale -> poussee``

        L'interet est double. D'abord ``throttle = 0.5`` tient l'altitude
        exactement, meme quand le drone est incline : la division par
        ``cos(roulis) * cos(tangage)`` compense la part de poussee "perdue"
        dans le virage. Ensuite le decollage s'arrete a l'altitude demandee
        au lieu de la depasser sur son elan.
        """
        if self._mode is FlightMode.LANDING:
            # Descente a vitesse constante jusqu'au contact.
            climb_target = self._safety.landing_speed
        elif self._target_altitude > 0.0:
            # Montee automatique : l'erreur d'altitude devient une consigne de
            # vitesse, bornee pour que la montee reste douce.
            error = self._target_altitude - (-self._position.z)
            if abs(error) < 0.2 and abs(self._velocity.z) < 0.3:
                self._target_altitude = 0.0
            climb_target = -clamp(error * ALTITUDE_P, -MAX_CLIMB_RATE, MAX_CLIMB_RATE)
        else:
            # Vol normal : le manche commande la vitesse verticale. En NED,
            # monter c'est aller vers les z negatifs, d'ou le signe.
            stick = clamp(control.throttle, 0.0, 1.0)
            climb_target = (0.5 - stick) * 2.0 * MAX_CLIMB_RATE

        # Boucle interne : acceleration verticale voulue (vers le bas, NED).
        accel_down = (climb_target - self._velocity.z) * CLIMB_RATE_P

        # a_d = -T * cos(roulis) * cos(tangage) + g  ->  on isole T.
        # Le plancher sur le cosinus evite une poussee infinie a la verticale.
        tilt = max(math.cos(self._roll) * math.cos(self._pitch), 0.3)
        thrust = (GRAVITY - accel_down) / tilt
        return clamp(thrust, 0.0, MAX_THRUST)

    def _step_attitude(self, control: ControlInput, dt: float) -> None:
        """Le drone rejoint l'assiette demandee avec un retard du premier ordre."""
        cfg = self._control
        # Pour avancer, un quadricoptere pique du nez : une consigne de tangage
        # positive (aller vers l'avant) donne donc un angle negatif.
        target_roll = clamp(control.roll, -1.0, 1.0) * cfg.max_tilt
        target_pitch = -clamp(control.pitch, -1.0, 1.0) * cfg.max_tilt

        alpha = dt / (ATTITUDE_TAU + dt)
        self._roll += alpha * (target_roll - self._roll)
        self._pitch += alpha * (target_pitch - self._pitch)

        yaw_rate = clamp(control.yaw_rate, -1.0, 1.0) * cfg.max_yaw_rate
        self._yaw = wrap_pi(self._yaw + yaw_rate * dt)

    def _step_translation(self, thrust: float, dt: float) -> None:
        """Integre les equations du mouvement.

        La poussee est dirigee selon l'axe ``-z`` du drone ; c'est son
        inclinaison qui la fait "pencher" et cree le deplacement horizontal.
        C'est pour cela qu'un quadricoptere ne peut pas avancer sans pencher.
        """
        cos_r, sin_r = math.cos(self._roll), math.sin(self._roll)
        cos_p, sin_p = math.cos(self._pitch), math.sin(self._pitch)
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)

        # Troisieme colonne de la matrice de rotation ZYX : l'axe "bas" du drone.
        body_down = Vec3(
            cos_y * sin_p * cos_r + sin_y * sin_r,
            sin_y * sin_p * cos_r - cos_y * sin_r,
            cos_p * cos_r,
        )

        accel = body_down * (-thrust) + Vec3(0.0, 0.0, GRAVITY)
        accel = accel - Vec3(
            self._velocity.x * DRAG_HORIZONTAL,
            self._velocity.y * DRAG_HORIZONTAL,
            self._velocity.z * DRAG_VERTICAL,
        )

        self._velocity = self._velocity + accel * dt
        self._position = self._position + self._velocity * dt

    def _drain_battery(self, throttle: float, dt: float) -> None:
        """Consommation croissante avec les gaz, jamais nulle."""
        load = 0.35 + 0.65 * clamp(throttle, 0.0, 1.0)
        self._battery = max(0.0, self._battery - load * dt / BATTERY_ENDURANCE)

    def _handle_ground_contact(self) -> None:
        """Empeche de passer sous le sol et termine l'atterrissage."""
        if self._position.z < 0.0:
            return

        self._position = Vec3(self._position.x, self._position.y, 0.0)
        self._velocity = Vec3(0.0, 0.0, 0.0)
        self._roll = 0.0
        self._pitch = 0.0

        if self._mode in {FlightMode.LANDING, FlightMode.FAILSAFE}:
            self._mode = FlightMode.DISARMED
            self._target_altitude = 0.0

    # ----- etat -----------------------------------------------------------

    @property
    def telemetry(self) -> Telemetry:
        return Telemetry(
            position=self._position,
            velocity=self._velocity,
            roll=self._roll,
            pitch=self._pitch,
            yaw=self._yaw,
            gimbal_yaw=self._gimbal_yaw,
            gimbal_pitch=self._gimbal_pitch,
            battery=self._battery,
            mode=self._mode,
        )

    def set_battery(self, level: float) -> None:
        """Force le niveau de batterie. Sert aux tests et aux demos de failsafe."""
        self._battery = clamp(level, 0.0, 1.0)
