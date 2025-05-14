import numpy as np

class TCLabThermalModel:
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

    def propagate_dynamics(self, Q1_pct, Q2_pct, dt):
        # Buffer the current input
        U_current = np.array([Q1_pct, Q2_pct, self.Tf]).reshape(3, 1)

        # RK4 integration
        NUM_RK_ITERATIONS = 1
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