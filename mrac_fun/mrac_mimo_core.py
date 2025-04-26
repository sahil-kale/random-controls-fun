import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from StateSpaceModel import StateSpaceModel

class MIMOMRACController:
    def __init__(self, ref_model, gamma_r, gamma_xp, num_control_inputs):
        self.ref_model = ref_model
        self.gamma_r = gamma_r
        self.gamma_xp = gamma_xp
        num_states = ref_model.A.shape[0]
        self.theta_r = np.zeros((num_states, num_control_inputs)) # assume that ref model has same number of outputs as plant
        self.theta_xp = np.zeros((num_states, num_control_inputs))

        self.num_control_inputs = num_control_inputs
        self.num_states = num_states
        self.assert_matrix_dimensions()

    def assert_matrix_dimensions(self):
        assert self.gamma_r.shape == (self.num_states, self.num_control_inputs), \
            f"gamma_r should be of shape ({self.num_states}, {self.num_control_inputs})"
        assert self.gamma_xp.shape == (self.num_states, self.num_control_inputs), \
            f"gamma_xp should be of shape ({self.num_states}, {self.num_control_inputs})"


    def step(self, r, x_p, dt):
        u = self.theta_r @ r + self.theta_xp @ x_p

        x_m = self.ref_model.output()
        
        assert x_m.shape == x_p.shape, f"Reference model output shape {x_m.shape} does not match plant state shape {x_p.shape}"
        # adaptive update (Lyapunov rule)
        e = x_p - x_m
        d_theta_xp = -self.gamma_xp @ e @ x_p.T
        d_theta_r = -self.gamma_r @ e @ r.T
        
        self.theta_xp += d_theta_xp * dt
        self.theta_r += d_theta_r * dt

        self.ref_model.update(r, dt)
        
        return u
    
    
    def get_theta_r(self):
        return self.theta_r

    def get_theta_xp(self):
        return self.theta_xp
    
    def get_ref_model(self):
        return self.ref_model
    


class MIMOMRACSimulator:
    def __init__(self, plant, controller, noise_std=0.0, noise_scale=0.0):
        self.plant = plant
        self.controller = controller
        self.reset_history()

        self.noise_std = noise_std
        self.noise_scale = noise_scale

    def reset_history(self):
        self.u_history = []
        self.x_p_history = []
        self.x_m_history = []
        self.theta_r_history = []
        self.theta_xp_history = []

    def step(self, r, dt):
        x_p = self.plant.output()

        noise_input = np.random.normal(0, self.noise_std, size=(self.plant.B.shape[1], 1)) * self.noise_scale
        x_p_plus_noise = x_p + noise_input

        u = self.controller.step(r, x_p_plus_noise, dt)
        self.plant.update(u, dt)

        self.u_history.append(u.copy())
        self.x_p_history.append(self.plant.output().copy())
        self.x_m_history.append(self.controller.get_ref_model().output().copy())
        self.theta_r_history.append(self.controller.get_theta_r().copy())
        self.theta_xp_history.append(self.controller.get_theta_xp().copy())
    
    def get_history(self):
        return np.array(self.u_history), np.array(self.x_p_history), np.array(self.x_m_history), \
               np.array(self.theta_r_history), np.array(self.theta_xp_history)
