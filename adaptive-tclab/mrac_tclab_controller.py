import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mrac_fun.mrac_mimo_core import MIMOMRACController
from mrac_fun.StateSpaceModel import StateSpaceModel

import tclab
import numpy as np
import argparse

class MRAC_TCLAB_Controller:
    def __init__(self, target_rise_time_s, ambient_temp, gamma_r, gamma_xp, num_control_inputs):
        bandwidth = target_rise_time_s * 0.35

        # assume first order dynamics for reference model
        A_m = np.array([[-bandwidth, 0]
                        [0, -bandwidth]])
        B_m = np.array([[bandwidth, 0],
                        [0, bandwidth]])
        
        C_m = np.array([[1, 0],
                        [0, 1]])
        D_m = np.array([[0, 0],
                        [0, 0]])
        
        self.ref_model = StateSpaceModel(A_m, B_m, C_m, D_m)

        self.mimo_mrac_controller = MIMOMRACController(self.ref_model, gamma_r, gamma_xp, num_control_inputs)

        self.ambient_temp = ambient_temp

    def step(self, ref_temps, plant_temps, dt):
        r = np.array(ref_temps).reshape(-1, 1)
        x_p = np.array(plant_temps).reshape(-1, 1)

        ambient_temp = np.array([self.ambient_temp]).reshape(-1, 1)

        # subtract ambient temperature from reference and plant temperatures in order to make them relative/"linearized"
        r -= ambient_temp
        x_p -= ambient_temp

        u = self.mimo_mrac_controller.step(r, x_p, dt)
        u = np.clip(u, 0, 100)
        u = u.flatten()

        return u
    
    def get_theta_r(self):
        return self.mimo_mrac_controller.get_theta_r()

    def get_theta_xp(self):
        return self.mimo_mrac_controller.get_theta_xp()
    
    def get_ref_model(self):
        return self.mimo_mrac_controller.get_ref_model()
    

if __name__ == "__main__":
    target_rise_time_s = 10
