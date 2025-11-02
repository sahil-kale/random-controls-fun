import numpy as np

class ADCSController:
    def __init__(self, kp=2.0, kd=0.6):
        self.kp = kp
        self.kd = kd
        pass

    def step(self, q_desired, omega_current, q_current):
        pass
