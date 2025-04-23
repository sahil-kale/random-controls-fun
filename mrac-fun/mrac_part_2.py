import numpy as np
import matplotlib.pyplot as plt

class StateSpaceModel:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.state = np.zeros(A.shape[0])

    def update(self, u, dt):
        x_dot = self.A @ self.state + self.B * u
        self.state = x_dot * dt + self.state # simple euler integration

    def output(self):
        return self.C @ self.state + self.D
    
class MRACSimulator:
    def __init__(self, plant, reference_model, gamma):
        self.plant = plant
        self.reference_model = reference_model
        self.gamma = gamma
        self.theta = np.array(0.0).reshape(-1, 1)

    def calculate_adaptive_control_weight(self, e, r, dt):
        dtheta = (-self.gamma * e * r) # MIT rule
        self.theta += dt * dtheta
        return self.theta
    
    def get_controller_output(self, r):
        return self.theta * r

    def propogate_sim(self, r, u, dt):
        self.reference_model.update(r, dt)
        self.plant.update(u, dt)

    def run_simulation(self, t, r, dt):
        x_p = []
        x_m = []
        u = []
        theta_hist = []

        for i in range(len(t)):
            u_i = self.get_controller_output(r[i])
            u.append(u_i)
            self.propogate_sim(r[i], u_i, dt)
            x_p.append(self.plant.output())
            x_m.append(self.reference_model.output())
            e = x_p[-1] - x_m[-1]
            self.calculate_adaptive_control_weight(e, r[i], dt)
            theta_hist.append(self.theta)
        

        return np.array(x_p), np.array(x_m), np.array(u), np.array(theta_hist)

    def plot_results(self, t, x_p, x_m, u, theta_hist):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, x_m.reshape(-1), label='Reference Model $x_m$')
        plt.plot(t, x_p.reshape(-1), label='Plant Output $x_p$', linestyle='--')
        plt.title('MRAC with MIT Rule (1st-Order Plant)')
        plt.xlabel('Time (s)')
        plt.ylabel('Output')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(t, u.reshape(-1), label='Control Input $u$', color='orange')
        plt.plot(t, theta_hist.reshape(-1), label='Adaptive Parameter $\Theta$', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input / Adaptive Parameter')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    A_p = np.array([-1]).reshape(-1, 1)
    B_p = np.array([1]).reshape(-1, 1)
    C_p = np.array([1]).reshape(1, -1)
    D_p = np.array([0]).reshape(1, 1)
    plant_model = StateSpaceModel(A_p, B_p, C_p, D_p)

    A_m = np.array([-2]).reshape(-1, 1)
    B_m = np.array([2]).reshape(-1, 1)
    C_m = np.array([1]).reshape(1, -1)
    D_m = np.array([0]).reshape(1, 1)
    reference_model = StateSpaceModel(A_m, B_m, C_m, D_m)


    dt = 0.01
    t = np.arange(0, 20, dt)
    
    # step to mag 1 for 10 seconds, then -1 for 10 seconds
    r = np.ones(t.shape) * 1.0
    r[t > 10] = -1.0


    simulator = MRACSimulator(plant_model, reference_model, gamma=5.0)
    x_p, x_m, u, theta_hist = simulator.run_simulation(t, r, dt)
    simulator.plot_results(t, x_p, x_m, u, theta_hist)


