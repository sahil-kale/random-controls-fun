import numpy as np
from multi_link_pendulum import MultiLinkPendulum

class IdealMPC:
    def __init__(self, plant_model: MultiLinkPendulum, sim_dt):
        self.plant_model = plant_model
        self.num_control_steps_to_optimize = 10
        self.sim_dt = sim_dt

    def compute_control_inputs(self, state: np.ndarray) -> np.ndarray:
        # state vector: [x, xdot] followed by [theta1, theta1_dot, theta2, theta2_dot, ...]
        num_states = self.plant_model.num_links * 2 + 2
        
