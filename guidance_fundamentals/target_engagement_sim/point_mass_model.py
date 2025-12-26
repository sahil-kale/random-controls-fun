import numpy as np
from guidance_fundamentals.target_engagement_sim.vector3D import Vector3D

class PointMassModel3D:
    def __init__(self, position: Vector3D, velocity: Vector3D, max_acceleration_m_per_s_s: Vector3D, delay_filt_tau: np.ndarray):
        self.position = position
        self.velocity = velocity
        
        self.max_acceleration_m_per_s_s = max_acceleration_m_per_s_s
        self.delay_filt_tau = delay_filt_tau
        
        # Track filtered acceleration state for first-order lag
        self.filtered_accel = np.array([0.0, 0.0, 0.0])
        
        assert self.max_acceleration_m_per_s_s.magnitude() >= 0.0, f"max_acceleration_m_per_s_s must have non-negative components"
        assert all([self.max_acceleration_m_per_s_s.x >= 0.0, self.max_acceleration_m_per_s_s.y >= 0.0, self.max_acceleration_m_per_s_s.z >= 0.0]), f"max_acceleration_m_per_s_s must be non-negative, got {self.max_acceleration_m_per_s_s.to_array()}"
        
        assert self.delay_filt_tau.shape == (3,), f"delay_filt_tau must be a 3-element array, got {self.delay_filt_tau.shape}"
        assert np.all(self.delay_filt_tau >= 0.0), f"delay_filt_tau must be non-negative, got {self.delay_filt_tau}"
        
    def step(self, commanded_acceleration_m_per_s_s: Vector3D, dt: float):
        assert dt > 0.0, f"dt must be positive, got {dt}"
        
        # Saturate commanded acceleration
        saturated_accel = Vector3D(
            np.clip(commanded_acceleration_m_per_s_s.x, -self.max_acceleration_m_per_s_s.x, self.max_acceleration_m_per_s_s.x),
            np.clip(commanded_acceleration_m_per_s_s.y, -self.max_acceleration_m_per_s_s.y, self.max_acceleration_m_per_s_s.y),
            np.clip(commanded_acceleration_m_per_s_s.z, -self.max_acceleration_m_per_s_s.z, self.max_acceleration_m_per_s_s.z),
        )
        
        # First-order lag filtering: d(accel)/dt = (cmd_accel - accel) / tau
        # Discrete form: accel_new = accel + (cmd_accel - accel) * dt / tau
        for i, tau in enumerate(self.delay_filt_tau):
            cmd_val = [saturated_accel.x, saturated_accel.y, saturated_accel.z][i]
            if tau > 0.0:
                self.filtered_accel[i] = self.filtered_accel[i] + (cmd_val - self.filtered_accel[i]) * dt / tau
            else:
                self.filtered_accel[i] = cmd_val
                
        filtered_accel_vec = Vector3D.from_array(self.filtered_accel)
        
        # Propagate state with explicit Euler
        # note, this led me down into a rabbit hole about symplectic vs explicit euler. 
        # See discretizations/euler_funsies.py for some simulations.
        
        self.position = Vector3D(
            self.position.x + self.velocity.x * dt,
            self.position.y + self.velocity.y * dt,
            self.position.z + self.velocity.z * dt,
        )
        
        
        self.velocity = Vector3D(
            self.velocity.x + filtered_accel_vec.x * dt,
            self.velocity.y + filtered_accel_vec.y * dt,
            self.velocity.z + filtered_accel_vec.z * dt,
        )
        
    def get_state(self) -> tuple[Vector3D, Vector3D]:
        return self.position, self.velocity
    
