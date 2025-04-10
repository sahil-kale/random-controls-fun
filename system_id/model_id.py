import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

C_TO_KELVIN_OFFSET = 273.15

class ThermalModel:
    def __init__(self, h1, h2, h3, Tf):
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.Tf = Tf

        self.construct_system_dynamics()
        self.X = np.ones((2, 1)) * Tf

    def construct_system_dynamics(self):
        surface_area = 1*10**-3 # m^2
        surface_area_between_heaters = 2*10**-4
        mass = 0.004 # kg
        heat_capacity = 500 # J/(kgK)

        r1 = 1/(self.h1*surface_area)
        r2 = 1/(self.h2*surface_area)
        r3 = 1/(self.h3*surface_area_between_heaters)

        A = np.array([
            [(-1/r1 - 1/r3), (1/r3)],
            [(1/r3), (-1/r2 - 1/r3)]
        ])
        self.A = A/(heat_capacity*mass)
        B = np.array([
            [1/100, 0, 1/(r1)],
            [0, 1/100, 1/(r2)]
        ])
        self.B = B/(heat_capacity*mass)

    def calculate_xdot(self, X, U): 
        x_dot = self.A @ X + self.B @ U
        return x_dot

    def propogate_dynamics(self, Q1_pct, Q2_pct, dt):
        U = np.array([Q1_pct, Q2_pct, self.Tf]).reshape(3, 1)

        # Runge-Kutta 4th order method for integration (https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html)
        k1 = self.calculate_xdot(self.X, U)
        k2 = self.calculate_xdot(self.X + dt/2*k1, U)
        k3 = self.calculate_xdot(self.X + dt/2*k2, U)
        k4 = self.calculate_xdot(self.X + dt*k3, U)
        self.X = self.X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
        return self.X
    
    def get_temperature(self):
        return self.X.reshape(-1)
    
def main():
    df = pd.read_csv('system_id/data/system_id_data.csv')
    df.columns = df.columns.str.strip()

    timestamps = df['Elapsed Time'].values
    heater_1 = df['Heater 1'].values
    heater_2 = df['Heater 2'].values
    temp_1 = df['Temp 1 (degC)'].values
    temp_2 = df['Temp 2 (degC)'].values

    # add C_TO_KELVIN_OFFSET to the temperatures
    temp_1 += C_TO_KELVIN_OFFSET
    temp_2 += C_TO_KELVIN_OFFSET
    Tf = 29 + C_TO_KELVIN_OFFSET

    def objective(x):
        h1, h2, h3 = x
        model = ThermalModel(h1, h2, h3, Tf)
        simulated_temperatures = []
        num_samples = len(timestamps) - 1
        error_sum = 0

        for i in range(num_samples):
            dt = timestamps[i+1] - timestamps[i]
            q1 = heater_1[i+1]
            q2 = heater_2[i+1]
            model.propogate_dynamics(q1, q2, dt)
            simulated_temperatures.append((model.X[0], model.X[1]))
            error_sum += (temp_1[i+1] - model.X[0])**2 + (temp_2[i+1] - model.X[1])**2
        
        return error_sum # no need to divide by num_samples, as the optimizer will find the minimum anyway
    
    # Initial guess for h1, h2, h3
    initial_guess = [10, 10, 10]
    bounds = [(0.1, None), (0.1, None), (0.1, None)]
    result = minimize(objective, initial_guess, bounds=bounds)
    h1, h2, h3 = result.x
    print(f"Optimized parameters: h1={h1}, h2={h2}, h3={h3}")
    print(result)

    # Plot the results
    model = ThermalModel(h1, h2, h3, Tf)
    simulated_temperatures = []
    num_samples = len(timestamps) - 1
    for i in range(num_samples):
        dt = timestamps[i+1] - timestamps[i]
        q1 = heater_1[i+1]
        q2 = heater_2[i+1]
        model.propogate_dynamics(q1, q2, dt)
        simulated_temperatures.append((model.X[0], model.X[1]))
    simulated_temperatures = np.array(simulated_temperatures)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps[1:], temp_1[1:], label='Measured Temp 1', color='red')
    plt.plot(timestamps[1:], temp_2[1:], label='Measured Temp 2', color='blue')
    plt.plot(timestamps[1:], simulated_temperatures[:, 0], label='Simulated Temp 1', linestyle='--', color='orange')
    plt.plot(timestamps[1:], simulated_temperatures[:, 1], label='Simulated Temp 2', linestyle='--', color='green')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Measured vs Simulated Temperatures')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()