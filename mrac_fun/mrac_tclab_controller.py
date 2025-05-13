from StateSpaceModel import StateSpaceModel
from mrac_mimo_core import MIMOMRACController

import tclab
import numpy as np
import argparse
from tclab import labtime
import os

class MRAC_TCLAB_Controller:
    def __init__(self, target_rise_time_s, ambient_temp, gamma_r, gamma_xp, num_control_inputs, theta_r, theta_xp, sigma_r=None, sigma_xp=None):
        bandwidth = 0.35/target_rise_time_s 

        # assume first order dynamics for reference model
        A_m = np.array([[-bandwidth, 0],
                        [0, -bandwidth]])
        B_m = np.array([[1, 0],
                        [0, 1]])
        
        C_m = np.array([[bandwidth, 0],
                        [0, bandwidth]])
        D_m = np.array([[0, 0],
                        [0, 0]])
        
        self.ref_model = StateSpaceModel(A_m, B_m, C_m, D_m)

        self.mimo_mrac_controller = MIMOMRACController(self.ref_model, gamma_r, gamma_xp, num_control_inputs, theta_r, theta_xp, sigma_r, sigma_xp)

        self.ambient_temp = ambient_temp

    def step(self, ref_temps, plant_temps, dt):
        r = np.array(ref_temps, dtype=np.float64).reshape(-1, 1)
        x_p = np.array(plant_temps).reshape(-1, 1)

        ambient_temp = np.array([self.ambient_temp]).reshape(-1, 1)

        # subtract ambient temperature from reference and plant temperatures in order to make them relative/"linearized"
        r -= ambient_temp
        x_p -= ambient_temp

        u = self.mimo_mrac_controller.step(r, x_p, dt)
        assert u.shape == (self.mimo_mrac_controller.num_control_inputs, 1), \
            f"Control input shape {u.shape} does not match expected shape ({self.mimo_mrac_controller.num_control_inputs}, 1)"
        u = u.flatten()

        return u
    
    def get_theta_r(self):
        return self.mimo_mrac_controller.get_theta_r()

    def get_theta_xp(self):
        return self.mimo_mrac_controller.get_theta_xp()
    
    def get_ref_model(self):
        return self.mimo_mrac_controller.get_ref_model()
    

def plot_results(T1_history, T2_history, Q1_history, Q2_history, ref_model_history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(T1_history, label='T1 (°C)')
    plt.plot(T2_history, label='T2 (°C)')
    ref_model_history_1 = [x[0] for x in ref_model_history]
    ref_model_history_2 = [x[1] for x in ref_model_history]

    plt.plot(ref_model_history_1, label='Ref Model T1 (°C)', linestyle='--')
    plt.plot(ref_model_history_2, label='Ref Model T2 (°C)', linestyle='--')

    plt.title('Temperature History')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(Q1_history, label='Q1 (%)')
    plt.plot(Q2_history, label='Q2 (%)')
    plt.title('Heater Power History')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_reference_temps(t):
    schedule = [
        (500, [40, 40]),
        (1000, [60, 30]),
        (1500, [40, 40]),
        (2000, [50, 50]),
        (2500, [40, 45]),
        (3000, [40, 40]),
        (3500, [50, 30]),
        (4000, [30, 40]),
        (4500, [40, 50]),
    ]
    for threshold, temps in schedule:
        if t < threshold:
            return temps
    return [50, 30]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRAC TCLab Controller')
    parser.add_argument('--simulation', action='store_true', help='Use simulated TCLab')
    parser.add_argument('--run-time', type=float, default=5000.0, help='Run time in seconds')
    parser.add_argument('--save-weights', action='store_true', help='Save full state feedback weights to file')
    parser.add_argument('--load-weights', action='store_true', help='Load full state feedback weights from file')
    args = parser.parse_args()
    fb_weight_paths = "mrac_fun/fb_weights/"

    if args.simulation:
        lab = tclab.setup(connected=False, speedup=12)()
        print("Running in simulation mode")
    else:
        lab = tclab.TCLab()

    gamma_r  = np.eye(2) * 0.00002
    gamma_xp = np.eye(2) * 0.00002
    num_control_inputs = 2
    target_rise_time_s = 70
    sigma_r  = np.eye(2) * 0.000000001
    sigma_xp = np.eye(2) * 0.000000001

    ambient_temp = lab.T1
    print(f"Ambient temperature: {ambient_temp} °C")

    if args.load_weights:
        print("Loading weights...")
        theta_r = np.load(fb_weight_paths + "theta_r.npy")
        theta_xp = np.load(fb_weight_paths + "theta_xp.npy")
    else:
        theta_r = np.zeros((num_control_inputs, 2))
        theta_xp = np.zeros((num_control_inputs, 2))

    controller = MRAC_TCLAB_Controller(target_rise_time_s, ambient_temp, gamma_r, gamma_xp, num_control_inputs, theta_r, theta_xp, sigma_r, sigma_xp)
    dt = 1.0
    T1_history = []
    T2_history = []
    Q1_history = []
    Q2_history = []
    u_history = []
    ref_model_history = []

    for t in tclab.clock(args.run_time, dt):
        ref_temps = get_reference_temps(t)

        plant_temps = [lab.T1, lab.T2]
        u = controller.step(ref_temps, plant_temps, dt)
        lab.Q1(u[0])
        lab.Q2(u[1])

        print(f"Time: {t:.2f} s, Heater 1: {lab.Q1():.2f} %, Heater 2: {lab.Q2():.2f} %, Temp 1: {lab.T1:.2f} °C, Temp 2: {lab.T2:.2f} °C")
        T1_history.append(lab.T1)
        T2_history.append(lab.T2)
        Q1_history.append(lab.Q1())
        Q2_history.append(lab.Q2())
        u_history.append(u)
        ref_model_output = controller.get_ref_model().output()
        ref_model_history.append(ref_model_output + np.array([ambient_temp, ambient_temp]).reshape(-1, 1))

    lab.Q1(0)
    lab.Q2(0)
    lab.close()
    print("Heaters turned off.")

    if args.save_weights:
        print("Saving weights...")
        # Create directory if it doesn't exist
        if not os.path.exists(fb_weight_paths):
            os.makedirs(fb_weight_paths)
        np.save(fb_weight_paths + "theta_r.npy", controller.get_theta_r())
        np.save(fb_weight_paths + "theta_xp.npy", controller.get_theta_xp())

    plot_results(T1_history, T2_history, Q1_history, Q2_history, ref_model_history)
    print("Experiment completed.")
