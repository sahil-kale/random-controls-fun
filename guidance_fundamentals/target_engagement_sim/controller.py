import numpy as np
import math
from guidance_fundamentals.target_engagement_sim.vector3D import Vector3D

class BaseController:
    def __init__(self):
        pass

    def compute_acceleration_command(self, pursuer_state: tuple[Vector3D, Vector3D], target_state: tuple[Vector3D, Vector3D], sim_time: float) -> Vector3D:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class DummyController(BaseController):
    def compute_acceleration_command(self, pursuer_state: tuple[Vector3D, Vector3D], target_state: tuple[Vector3D, Vector3D], sim_time: float) -> Vector3D:
        # Always command zero acceleration
        return Vector3D(0.0, 0.0, 0.0)

class SpiralPathController(BaseController):
    def __init__(
        self,
        spiral_path_velocity_vector: Vector3D,
        spiral_radius_m: float,
        spiral_angular_rate_rad_per_s: float,
        kp: float = 2.0,
        kd: float = 2.0,
        max_accel_mps2: float | None = None,
        phase0_rad: float = 0.0,
        center0_world: Vector3D | None = None,
    ):
        """
        Helix / spiral trajectory tracking.

        The spiral axis is along spiral_path_velocity_vector.
        The centerline moves at constant speed |spiral_path_velocity_vector|.
        The pursuer tracks the moving helix with PD + feedforward.
        """
        super().__init__()
        self.v_axis_vec = spiral_path_velocity_vector
        self.R = float(spiral_radius_m)
        self.w = float(spiral_angular_rate_rad_per_s)

        self.kp = float(kp)
        self.kd = float(kd)
        self.max_accel = max_accel_mps2

        self.phase0 = float(phase0_rad)
        self.center0_world = center0_world  # if None, lock it on first call

        # Feedforward centripetal magnitude (useful for sanity checks/clamping)
        self.centripetal_accel_mag = (self.w ** 2) * self.R

        # Cache a stable transverse basis once we know the axis direction
        self._basis_ready = False
        self._e1 = None
        self._e2 = None
        self._axis_hat = None
        self._v_axis_mag = None

    def _init_basis_if_needed(self):
        if self._basis_ready:
            return

        vmag = self.v_axis_vec.norm()
        if vmag < 1e-9:
            raise ValueError("spiral_path_velocity_vector magnitude must be > 0")

        axis_hat = self.v_axis_vec * (1.0 / vmag)  # direction of helix axis

        # Pick a reference vector not parallel to axis_hat to build an orthonormal basis
        # (this avoids numerical issues when axis is close to +Z)
        ref = Vector3D(0.0, 0.0, 1.0)
        if abs(axis_hat.dot(ref)) > 0.95:
            ref = Vector3D(0.0, 1.0, 0.0)

        e1 = axis_hat.cross(ref).normalize()  # transverse
        e2 = axis_hat.cross(e1).normalize()   # transverse, 90deg from e1

        self._axis_hat = axis_hat
        self._v_axis_mag = vmag
        self._e1 = e1
        self._e2 = e2
        self._basis_ready = True

    def _clamp_accel(self, a: Vector3D) -> Vector3D:
        if self.max_accel is None:
            return a
        mag = a.norm()
        if mag <= self.max_accel or mag < 1e-9:
            return a
        return a * (self.max_accel / mag)

    def compute_acceleration_command(
        self,
        pursuer_state: tuple[Vector3D, Vector3D],
        target_state: tuple[Vector3D, Vector3D],
        sim_time: float
    ) -> Vector3D:
        # Unpack this vehicle's current state (first argument is always "self" state)
        p, v = pursuer_state
        t = float(sim_time)

        self._init_basis_if_needed()

        # Choose the moving helix "centerline" origin
        # If not provided, lock it to the initial position to avoid an initial jump.
        if self.center0_world is None:
            # Put the initial centerline point at the pursuer projected onto axis,
            # so the helix starts "near" the pursuer.
            self.center0_world = p  # simple and usually fine

        # Reference helix definitions:
        # centerline moves along axis at speed |v_axis|
        c_ref = self.center0_world + self._axis_hat * (self._v_axis_mag * t)

        theta = self.phase0 + self.w * t
        ct = math.cos(theta)
        st = math.sin(theta)

        # Position on helix: centerline + radius in transverse plane
        p_ref = c_ref + self._e1 * (self.R * ct) + self._e2 * (self.R * st)

        # Velocity: axis translation + transverse rotation
        # d/dt [R cosθ e1 + R sinθ e2] = R ω [-sinθ e1 + cosθ e2]
        v_ref = (self._axis_hat * self._v_axis_mag) + self._e1 * (self.R * self.w * (-st)) + self._e2 * (self.R * self.w * (ct))

        # Acceleration: transverse centripetal term (axis component is zero if axis speed constant)
        # d/dt of above = -R ω^2 [cosθ e1 + sinθ e2]
        a_ref = self._e1 * (-self.R * (self.w ** 2) * ct) + self._e2 * (-self.R * (self.w ** 2) * st)

        # Tracking errors
        e_p = p_ref - p
        e_v = v_ref - v

        # Canonical accel command: feedforward + PD
        a_cmd = a_ref + (e_p * self.kp) + (e_v * self.kd)

        return self._clamp_accel(a_cmd)
