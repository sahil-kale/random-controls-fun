import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from StateSpaceModel import StateSpaceModel


class MRACSimulator:
    def __init__(self, plant, reference_model, gamma):
        self.plant = plant
        self.reference_model = reference_model
        self.gamma = gamma
        self.theta = np.zeros((2,))

    def calculate_adaptive_control_weight(self, e, r, x_p, dt):
        self.theta[0] -= self.gamma * e * r * dt
        self.theta[1] -= self.gamma * e * x_p * dt

    def get_controller_output(self, r, x_p):
        return self.theta[0] * r + self.theta[1] * x_p

    def step(self, r, dt):
        u = self.get_controller_output(r, self.plant.output())
        self.reference_model.update(r, dt)
        self.plant.update(u, dt)
        e = self.plant.output() - self.reference_model.output()
        self.calculate_adaptive_control_weight(e, r, self.plant.output(), dt)
        return self.plant.output().item(), self.reference_model.output().item(), u, self.theta.copy()

if __name__ == "__main__":
    A_p, B_p, C_p, D_p = np.array([[2]]), np.array([[1]]), np.array([[1]]), np.array([[0]])
    A_m, B_m, C_m, D_m = np.array([[-2]]), np.array([[2]]), np.array([[1]]), np.array([[0]])

    plant_model = StateSpaceModel(A_p, B_p, C_p, D_p)
    reference_model = StateSpaceModel(A_m, B_m, C_m, D_m)
    simulator = MRACSimulator(plant_model, reference_model, gamma=5.0)

    # Plotting setup
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    line_ref, = ax1.plot([], [], label='Reference $x_m$', color='blue')
    line_plant, = ax1.plot([], [], label='Plant $x_p$', linestyle='--', color='red')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-6, 6)
    ax1.legend()
    ax1.set_ylabel('Output')

    line_u, = ax2.plot([], [], label='Control $u$', color='orange')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Input')
    ax2.legend()

    line_theta0, = ax3.plot([], [], label=r'$\theta_r$', linestyle='--', color='purple')
    line_theta1, = ax3.plot([], [], label=r'$\theta_{x_p}$', linestyle=':', color='green')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-20, 20)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Adaptive Parameters')
    ax3.legend()

    slider_ax = plt.axes([0.25, 0.05, 0.50, 0.03])
    slider = Slider(slider_ax, 'Reference Input', -5.0, 5.0, valinit=1.0)

    slider_A_p_pole_ax = plt.axes([0.25, 0.01, 0.50, 0.03])
    slider_A_p_pole = Slider(slider_A_p_pole_ax, 'Pole A_p', -5.0, 5.0, valinit=0.0)

    # Simulation buffers
    dt = 0.05
    t_data, x_p_data, x_m_data, u_data, theta_0_data, theta_1_data = [], [], [], [], [], []
    time_counter = 0

    def update(frame):
        global time_counter
        ref = slider.val
        ref = np.array(ref).reshape(-1, 1)
        plant_model.A[0, 0] = slider_A_p_pole.val
        x_p, x_m, u, theta = simulator.step(ref, dt)
        time_counter += dt

        t_data.append(time_counter)
        x_p_data.append(x_p)
        x_m_data.append(x_m)
        u_data.append(u)
        theta_0_data.append(theta[0])
        theta_1_data.append(theta[1])

        t_min = max(0, time_counter - 10)
        ax1.set_xlim(t_min, time_counter)
        ax2.set_xlim(t_min, time_counter)
        ax3.set_xlim(t_min, time_counter)

        line_ref.set_data(t_data, x_m_data)
        line_plant.set_data(t_data, x_p_data)
        line_u.set_data(t_data, u_data)
        line_theta0.set_data(t_data, theta_0_data)
        line_theta1.set_data(t_data, theta_1_data)

        return line_ref, line_plant, line_u, line_theta0, line_theta1

    ani = FuncAnimation(fig, update, interval=dt*1000)
    plt.show()
