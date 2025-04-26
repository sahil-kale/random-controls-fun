import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from StateSpaceModel import StateSpaceModel

class MIMOMRACController:
    def __init__(self, ref_model, gamma_r, gamma_xp, num_control_inputs):
        self.ref_model = ref_model
        self.gamma_r = gamma_r
        self.gamma_xp = gamma_xp
        num_states = ref_model.A.shape[0]
        self.theta_r = np.zeros((num_states, num_control_inputs)) # assume that ref model has same number of outputs as plant
        self.theta_xp = np.zeros((num_states, num_control_inputs))

        self.num_control_inputs = num_control_inputs
        self.num_states = num_states
        self.assert_matrix_dimensions()

    def assert_matrix_dimensions(self):
        assert self.gamma_r.shape == (self.num_states, self.num_control_inputs), \
            f"gamma_r should be of shape ({self.num_states}, {self.num_control_inputs})"
        assert self.gamma_xp.shape == (self.num_states, self.num_control_inputs), \
            f"gamma_xp should be of shape ({self.num_states}, {self.num_control_inputs})"


    def step(self, r, x_p, dt):
        u = self.theta_r @ r + self.theta_xp @ x_p

        x_m = self.ref_model.output()
        
        assert x_m.shape == x_p.shape, f"Reference model output shape {x_m.shape} does not match plant state shape {x_p.shape}"
        # adaptive update (Lyapunov rule)
        e = x_p - x_m
        d_theta_xp = -self.gamma_xp @ e @ x_p.T
        d_theta_r = -self.gamma_r @ e @ r.T
        
        self.theta_xp += d_theta_xp * dt
        self.theta_r += d_theta_r * dt

        self.ref_model.update(r, dt)
        
        return u
    
    
    def get_theta_r(self):
        return self.theta_r

    def get_theta_xp(self):
        return self.theta_xp
    
    def get_ref_model(self):
        return self.ref_model
    


class MIMOMRACSimulator:
    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller
        self.reset_history()

    def reset_history(self):
        self.u_history = []
        self.x_p_history = []
        self.x_m_history = []
        self.theta_r_history = []
        self.theta_xp_history = []

    def step(self, r, dt):
        u = self.controller.step(r, self.plant.output(), dt)
        self.plant.update(u, dt)

        self.u_history.append(u.copy())
        self.x_p_history.append(self.plant.output().copy())
        self.x_m_history.append(self.controller.get_ref_model().output().copy())
        self.theta_r_history.append(self.controller.get_theta_r().copy())
        self.theta_xp_history.append(self.controller.get_theta_xp().copy())
    
    def get_history(self):
        return np.array(self.u_history), np.array(self.x_p_history), np.array(self.x_m_history), \
               np.array(self.theta_r_history), np.array(self.theta_xp_history)
    
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