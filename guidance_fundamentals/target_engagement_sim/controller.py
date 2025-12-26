import numpy as np
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