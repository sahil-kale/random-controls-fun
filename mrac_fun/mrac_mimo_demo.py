import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from StateSpaceModel import StateSpaceModel
from mrac_mimo_core import MIMOMRACSimulator, MIMOMRACController

if __name__ == "__main__":
    # Plant and Reference Model Setup
    A_p = np.array([[-1, -6],
                    [-10, -2]])
    B_p = np.eye(2)
    C_p = np.eye(2)
    D_p = np.zeros((2, 2))
    plant_model = StateSpaceModel(A_p, B_p, C_p, D_p)

    A_r = np.array([[-3, 0],
                    [0, -5]])
    B_r = np.array([[3, 0],
                    [0, 5]])
    C_r = np.eye(2)
    D_r = np.zeros((2, 2))
    reference_model = StateSpaceModel(A_r, B_r, C_r, D_r)

    gamma_r_const = 5.0
    gamma_xp_const = 5.0

    gamma_r = np.eye(2) * gamma_r_const
    gamma_xp = np.eye(2) * gamma_xp_const

    num_control_inputs = 2
    theta_r = np.zeros((num_control_inputs, 2))
    theta_xp = np.zeros((num_control_inputs, 2))

    controller = MIMOMRACController(reference_model, gamma_r, gamma_xp, num_control_inputs, theta_r, theta_xp)
    simulator = MIMOMRACSimulator(plant_model, controller, noise_std=0.25, noise_scale=0.4)

    # Plotting setup
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    plt.subplots_adjust(bottom=0.3)

    # Line objects
    line_xp1, = ax1.plot([], [], label='Plant State 1', color='red')
    line_xm1, = ax1.plot([], [], '--', label='Reference State 1', color='red', alpha=0.5)
    line_xp2, = ax1.plot([], [], label='Plant State 2', color='blue')
    line_xm2, = ax1.plot([], [], '--', label='Reference State 2', color='blue', alpha=0.5)

    ax1.set_xlim(0, 10)
    ax1.set_ylim(-6, 6)
    ax1.set_ylabel('States')
    ax1.legend()
    ax1.grid(True)

    line_u1, = ax2.plot([], [], label='u1', color='green')
    line_u2, = ax2.plot([], [], label='u2', color='purple')

    ax2.set_xlim(0, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_ylabel('Control Inputs')
    ax2.legend()
    ax2.grid(True)

    line_theta_r_00, = ax3.plot([], [], label=r'$\theta_{r}[0,0]$', linestyle='--', color='orange')
    line_theta_r_11, = ax3.plot([], [], label=r'$\theta_{r}[1,1]$', linestyle='--', color='brown')
    line_theta_xp_00, = ax3.plot([], [], label=r'$\theta_{xp}[0,0]$', linestyle=':', color='green')
    line_theta_xp_11, = ax3.plot([], [], label=r'$\theta_{xp}[1,1]$', linestyle=':', color='blue')

    ax3.set_xlim(0, 10)
    ax3.set_ylim(-20, 20)
    ax3.set_ylabel('Adaptive Parameters')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)

    # Sliders
    ax_slider_r1 = plt.axes([0.25, 0.15, 0.50, 0.03])
    slider_r1 = Slider(ax_slider_r1, 'r1', -5.0, 5.0, valinit=1.0)

    ax_slider_r2 = plt.axes([0.25, 0.10, 0.50, 0.03])
    slider_r2 = Slider(ax_slider_r2, 'r2', -5.0, 5.0, valinit=1.0)

    # Simulation buffers
    dt = 0.02
    t_data, xp1_data, xm1_data, xp2_data, xm2_data = [], [], [], [], []
    u1_data, u2_data = [], []
    theta_r_00_data, theta_r_11_data = [], []
    theta_xp_00_data, theta_xp_11_data = [], []
    time_counter = 0

    def update(frame):
        global time_counter
        r = np.array([[slider_r1.val],
                      [slider_r2.val]])

        simulator.step(r, dt)
        u, x_p, x_m, theta_r, theta_xp = simulator.get_history()

        time_counter += dt

        # Save latest data
        t_data.append(time_counter)
        xp1_data.append(x_p[-1, 0])
        xm1_data.append(x_m[-1, 0])
        xp2_data.append(x_p[-1, 1])
        xm2_data.append(x_m[-1, 1])
        u1_data.append(u[-1, 0])
        u2_data.append(u[-1, 1])
        theta_r_00_data.append(theta_r[-1, 0, 0])
        theta_r_11_data.append(theta_r[-1, 1, 1])
        theta_xp_00_data.append(theta_xp[-1, 0, 0])
        theta_xp_11_data.append(theta_xp[-1, 1, 1])

        # Update plot windows dynamically
        t_min = max(0, time_counter - 10)
        ax1.set_xlim(t_min, time_counter)
        ax2.set_xlim(t_min, time_counter)
        ax3.set_xlim(t_min, time_counter)

        # Update plot data
        line_xp1.set_data(t_data, xp1_data)
        line_xm1.set_data(t_data, xm1_data)
        line_xp2.set_data(t_data, xp2_data)
        line_xm2.set_data(t_data, xm2_data)
        line_u1.set_data(t_data, u1_data)
        line_u2.set_data(t_data, u2_data)
        line_theta_r_00.set_data(t_data, theta_r_00_data)
        line_theta_r_11.set_data(t_data, theta_r_11_data)
        line_theta_xp_00.set_data(t_data, theta_xp_00_data)
        line_theta_xp_11.set_data(t_data, theta_xp_11_data)

        return (line_xp1, line_xm1, line_xp2, line_xm2, line_u1, line_u2,
                line_theta_r_00, line_theta_r_11, line_theta_xp_00, line_theta_xp_11)

    ani = FuncAnimation(fig, update, interval=dt*1000)
    plt.show()
