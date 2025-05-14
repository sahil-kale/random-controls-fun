import tclab
import numpy as np
from tclab import labtime
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from parameterized_thermal_model import TCLabThermalModel

class TCLabMovingHorizonEstimator:
    def __init__(self, Tf, debug):
        self.Tf = Tf # ambient temperature
        self.h1 = 10
        self.h2 = 10
        self.h3 = 10
        self.alpha_1 = 0.001
        self.alpha_2 = 0.005
        self.tau = 20
        self.model = TCLabThermalModel(self.h1, self.h2, self.h3, self.Tf, self.alpha_1, self.alpha_2, self.tau)
        self.debug = debug

    def estimate_and_store_parameters(self, temps, heaters, timestamps):
        assert temps.shape[0] == timestamps.shape[0], "Temperature and time arrays must have the same length."
        assert temps.shape[1] == 2, "Temperature array must have two columns for T1 and T2."
        assert heaters.shape[0] == timestamps.shape[0], "Heater and time arrays must have the same length."
        assert heaters.shape[1] == 2, "Heater array must have two columns for Q1 and Q2."
        temp_1 = temps[:, 0]
        temp_2 = temps[:, 1]
        heater_1 = heaters[:, 0]
        heater_2 = heaters[:, 1]

        def objective(x):
            h1, h2, h3, alpha_1, alpha_2, tau = x
            model = TCLabThermalModel(h1, h2, h3, self.Tf, alpha_1, alpha_2, tau)
            error_sum = 0
            for i in range(len(timestamps) - 1):
                dt = timestamps[i+1] - timestamps[i]
                model.propagate_dynamics(heater_1[i+1], heater_2[i+1], timestamps[i+1], dt)
                states = model.get_temperature()
                error_sum += (temp_1[i+1] - states[0])**2 + (temp_2[i+1] - states[1])**2
            return error_sum
        
        initial_guess = [10, 10, 10, 1/100, 0.75/100, 20]
        bounds = [(1, 100), (1, 100), (1, 1000), (0.001, 0.015), (0.001, 0.015), (1, None)]
        result = minimize(objective, initial_guess, bounds=bounds)
        h1, h2, h3, alpha_1, alpha_2, tau = result.x
        if self.debug:
            print(f"Optimization Result: {result}")
            print(f"Optimized Parameters: h1={h1}, h2={h2}, h3={h3}, alpha_1={alpha_1}, alpha_2={alpha_2}, tau={tau}")
        
        self.model = TCLabThermalModel(h1, h2, h3, self.Tf, alpha_1, alpha_2, tau)
        
        # store parameters to warm start the next MHE optimization
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.tau = tau

