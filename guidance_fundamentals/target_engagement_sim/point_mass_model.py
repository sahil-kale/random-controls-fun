import numpy as np
from guidance_fundamentals.target_engagement_sim.vector3D import Vector3D

class PointMassModel3D:
    def __init__(self, position: Vector3D, velocity: Vector3D, max_acceleration_m_per_s_s: Vector3D, delay_filt_tau: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.max_acceleration_m_per_s_s = max_acceleration_m_per_s_s
        self.delay_filt_tau = delay_filt_tau
        self.filtered_accel = np.array([0.0, 0.0, 0.0])
        
    def step(self, commanded_acceleration_m_per_s_s: Vector3D, dt: float):
        saturated_accel = Vector3D(
            np.clip(commanded_acceleration_m_per_s_s.x, -self.max_acceleration_m_per_s_s.x, self.max_acceleration_m_per_s_s.x),
            np.clip(commanded_acceleration_m_per_s_s.y, -self.max_acceleration_m_per_s_s.y, self.max_acceleration_m_per_s_s.y),
            np.clip(commanded_acceleration_m_per_s_s.z, -self.max_acceleration_m_per_s_s.z, self.max_acceleration_m_per_s_s.z),
        )
        
        cmd_array = [saturated_accel.x, saturated_accel.y, saturated_accel.z]
        for i, tau in enumerate(self.delay_filt_tau):
            if tau > 0.0:
                self.filtered_accel[i] += (cmd_array[i] - self.filtered_accel[i]) * dt / tau
            else:
                self.filtered_accel[i] = cmd_array[i]
        
        filtered_accel_vec = Vector3D.from_array(self.filtered_accel)
        
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
    
