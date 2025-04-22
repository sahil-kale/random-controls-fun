import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import deque
import argparse

C_TO_KELVIN_OFFSET = 273.15

class ThermalModel:
    def __init__(self, h1, h2, h3, Tf, alpha_1, alpha_2, tau):
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.Tf = Tf
        self.tau = tau
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.construct_system_dynamics()
        self.X = np.ones((4, 1)) * Tf  # State vector: [T1, T2, T1_C, T2_C]

    def construct_system_dynamics(self):
        surface_area = 1e-3  # m^2
        surface_area_between_heaters = 2e-4
        mass = 0.004  # kg
        heat_capacity = 500  # J/(kgK)

        r1 = 1 / (self.h1 * surface_area)
        r2 = 1 / (self.h2 * surface_area)
        r3 = 1 / (self.h3 * surface_area_between_heaters)

        A = np.array([
            [(-1/r1 - 1/r3) / (heat_capacity * mass),   (1/r3) / (heat_capacity * mass),         0, 0],
            [(1/r3) / (heat_capacity * mass),           (-1/r2 - 1/r3) / (heat_capacity * mass), 0, 0],
            [1/self.tau, 0, -1/self.tau, 0],
            [0, 1/self.tau, 0, -1/self.tau]
        ])
        self.A = A

        B = np.array([
            [self.alpha_1, 0, 1/(r1)],
            [0, self.alpha_2, 1/(r2)],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.B = B / (heat_capacity * mass)
        self.C = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def calculate_xdot(self, X, U):
        return self.A @ X + self.B @ U

    def propagate_dynamics(self, Q1_pct, Q2_pct, current_time, dt):
        # Buffer the current input
        U_current = np.array([Q1_pct, Q2_pct, self.Tf]).reshape(3, 1)

        # RK4 integration
        NUM_RK_ITERATIONS = 2
        dt_step = dt / NUM_RK_ITERATIONS
        for _ in range(NUM_RK_ITERATIONS):
            k1 = self.calculate_xdot(self.X, U_current)
            k2 = self.calculate_xdot(self.X + dt_step/2 * k1, U_current)
            k3 = self.calculate_xdot(self.X + dt_step/2 * k2, U_current)
            k4 = self.calculate_xdot(self.X + dt_step * k3, U_current)
            self.X = self.X + (dt_step / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return self.X

    def get_temperature(self):
        return self.C @ self.X

def main():
    parser = argparse.ArgumentParser(description="System Identification for Thermal Model")
    parser.add_argument('data_file', type=str, help='Path to the data file')

    args = parser.parse_args()


    df = pd.read_csv(f'{args.data_file}')
    df.columns = df.columns.str.strip()

    timestamps = df['Elapsed Time'].values
    heater_1 = df['Heater 1'].values
    heater_2 = df['Heater 2'].values
    temp_1 = df['Temp 1 (degC)'].values + C_TO_KELVIN_OFFSET
    temp_2 = df['Temp 2 (degC)'].values + C_TO_KELVIN_OFFSET
    Tf = temp_1[0]

    def objective(x):
        h1, h2, h3, alpha_1, alpha_2, tau = x
        model = ThermalModel(h1, h2, h3, Tf, alpha_1, alpha_2, tau)
        error_sum = 0
        for i in range(len(timestamps) - 1):
            dt = timestamps[i+1] - timestamps[i]
            model.propagate_dynamics(heater_1[i+1], heater_2[i+1], timestamps[i+1], dt)
            states = model.get_temperature()
            error_sum += (temp_1[i+1] - states[0])**2 + (temp_2[i+1] - states[1])**2
        return error_sum

    # Initial guess and bounds
    initial_guess = [10, 10, 10, 1/100, 0.75/100, 20]
    bounds = [(1, 100), (1, 100), (1, 1000), (0.001, 0.015), (0.001, 0.015), (1, None)]
    result = minimize(objective, initial_guess, bounds=bounds, method='Nelder-Mead')
    print(result)
    h1, h2, h3, alpha_1, alpha_2, tau = result.x
    print(f"Optimized Parameters: h1={h1}, h2={h2}, h3={h3}, alpha_1={alpha_1}, alpha_2={alpha_2}, tau={tau}")

    # Plotting
    model = ThermalModel(h1, h2, h3, Tf, alpha_1, alpha_2, tau)
    simulated_temperatures = []
    for i in range(len(timestamps) - 1):
        dt = timestamps[i+1] - timestamps[i]
        model.propagate_dynamics(heater_1[i+1], heater_2[i+1], timestamps[i+1], dt)
        states = model.get_temperature()
        simulated_temperatures.append((states[0], states[1]))

    simulated_temperatures = np.array(simulated_temperatures)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Temperature subplot ---
    axs[0].plot(timestamps[1:], temp_1[1:], label='Measured Temp 1', color='red')
    axs[0].plot(timestamps[1:], temp_2[1:], label='Measured Temp 2', color='blue')
    axs[0].plot(timestamps[1:], simulated_temperatures[:, 0], label='Simulated Temp 1', linestyle='--', color='orange')
    axs[0].plot(timestamps[1:], simulated_temperatures[:, 1], label='Simulated Temp 2', linestyle='--', color='green')
    axs[0].set_ylabel('Temperature (K)')
    axs[0].set_title('Measured vs Simulated Temperatures (with Dead-Time)')
    axs[0].legend()
    axs[0].grid(True)

    # --- Control signal subplot ---
    axs[1].plot(timestamps, heater_1, label='Heater 1 %', color='purple')
    axs[1].plot(timestamps, heater_2, label='Heater 2 %', color='teal')
    axs[1].set_xlabel('Elapsed Time (s)')
    axs[1].set_ylabel('Heater Output (%)')
    axs[1].set_title('Heater Control Actions')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
