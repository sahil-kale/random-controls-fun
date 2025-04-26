import numpy as np
import matplotlib.pyplot as plt
from StateSpaceModel import StateSpaceModel
from mrac_mimo_core import MIMOMRACController, MIMOMRACSimulator

def plot_mimo_mrac_results(t, u, x_p, x_m, theta_r, theta_xp):
    fig, axs = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    # 1. Plant vs Reference States
    axs[0].plot(t, x_p[:, 0], label='Plant State 1')
    axs[0].plot(t, x_m[:, 0], '--', label='Ref Model State 1')
    axs[0].plot(t, x_p[:, 1], label='Plant State 2')
    axs[0].plot(t, x_m[:, 1], '--', label='Ref Model State 2')
    axs[0].set_ylabel('States')
    axs[0].set_title('Plant vs Reference States')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Control Inputs
    axs[1].plot(t, u[:, 0], label='u1')
    axs[1].plot(t, u[:, 1], label='u2')
    axs[1].set_ylabel('Control Inputs')
    axs[1].set_title('Control Inputs')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Theta_r Evolution
    for i in range(theta_r.shape[2]):
        axs[2].plot(t, theta_r[:, 0, i], label=f'$\\theta_r[0,{i}]$')
        axs[2].plot(t, theta_r[:, 1, i], label=f'$\\theta_r[1,{i}]$')
    axs[2].set_ylabel('Theta_r entries')
    axs[2].set_title('Evolution of Theta_r')
    axs[2].legend()
    axs[2].grid(True)

    # 4. Theta_xp Evolution
    for i in range(theta_xp.shape[2]):
        axs[3].plot(t, theta_xp[:, 0, i], label=f'$\\theta_{{xp}}[0,{i}]$')
        axs[3].plot(t, theta_xp[:, 1, i], label=f'$\\theta_{{xp}}[1,{i}]$')
    axs[3].set_ylabel('Theta_xp entries')
    axs[3].set_title('Evolution of Theta_xp')
    axs[3].legend()
    axs[3].grid(True)

    # 5. Tracking Error Norm
    error = np.linalg.norm(x_p - x_m, axis=1)  # axis=1 because (time, 2 states)
    axs[4].plot(t, error, label='Tracking Error Norm')
    axs[4].set_ylabel('Error Norm')
    axs[4].set_title('Tracking Error Over Time')
    axs[4].legend()
    axs[4].grid(True)

    axs[4].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
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

    gamma_r_const  = 5.0
    gamma_xp_const = 5.0

    gamma_r = np.eye(2) * gamma_r_const
    gamma_xp = np.eye(2) * gamma_xp_const

    controller = MIMOMRACController(reference_model, gamma_r, gamma_xp, num_control_inputs=2)
    simulator = MIMOMRACSimulator(plant_model, controller)

    dt = 0.001
    T = 60
    t = np.arange(0, T, dt)

    for i in range(len(t)):
        r = 1
        sim_time = t[i]
        if sim_time > 40:
            r = 0
        elif sim_time > 20:
            r = -1

        r = np.ones((2,1)) * r

        simulator.step(r, dt)

    u, x_p, x_m, theta_r, theta_xp = simulator.get_history()

    plot_mimo_mrac_results(t, u, x_p, x_m, theta_r, theta_xp)