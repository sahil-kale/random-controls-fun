import tclab
import numpy as np
import argparse
import os
from tclab import labtime
import time

COOL_OFF_TIME_S = 60 * 5
SAVE_DIR = "system_id/data"

RANDOM_TEMP_CONTROL_DT_S = 45

def main(args):
    np.random.seed(42)
    if args.simulation:
        lab = tclab.setup(connected=False, speedup=10)
        lab = lab()
        print("Running in simulation mode")
    else:
        lab = tclab.TCLab()
    print(f"Starting Experiment")
    lab.LED(0)
    labtime.sleep(1)
    lab.LED(100)
    labtime.sleep(1)

    elapsed_time_data = []
    heater_1_data = []
    heater_2_data = []
    temp_1_data = []
    temp_2_data = []

    lab.Q1(100)
    lab.Q2(0)
    start_time = labtime.time()
    duration = args.duration
    dt = args.dt

    while labtime.time() - start_time < duration:
        current_time = labtime.time()
        elapsed_time = current_time - start_time
        elapsed_time_data.append(elapsed_time)
        heater_1_data.append(lab.Q1())
        heater_2_data.append(lab.Q2())
        temp_1_data.append(lab.T1)
        temp_2_data.append(lab.T2)
        print(f"Elapsed Time: {elapsed_time:.2f} s, Heater 1: {lab.Q1()}, Heater 2: {lab.Q2()}, Temp 1: {lab.T1}, Temp 2: {lab.T2}")
        loop_time = labtime.time() - current_time
        labtime.sleep(max(0, dt - loop_time))

    local_start_time = labtime.time()
    lab.Q1(0)
    lab.Q2(0)
    
    while labtime.time() - local_start_time < COOL_OFF_TIME_S:
        current_time = labtime.time()
        elapsed_time = current_time - start_time
        elapsed_time_data.append(elapsed_time)
        heater_1_data.append(lab.Q1())
        heater_2_data.append(lab.Q2())
        temp_1_data.append(lab.T1)
        temp_2_data.append(lab.T2)
        print(f"Elapsed Time: {elapsed_time:.2f} s, Heater 1: {lab.Q1()}, Heater 2: {lab.Q2()}, Temp 1: {lab.T1}, Temp 2: {lab.T2}")
        loop_time = labtime.time() - current_time
        labtime.sleep(max(0, dt - loop_time))

    lab.Q1(0)
    lab.Q2(100)
    
    local_start_time = labtime.time()
    while labtime.time() - local_start_time < duration:
        current_time = labtime.time()
        elapsed_time = current_time - start_time
        elapsed_time_data.append(elapsed_time)
        heater_1_data.append(lab.Q1())
        heater_2_data.append(lab.Q2())
        temp_1_data.append(lab.T1)
        temp_2_data.append(lab.T2)
        print(f"Elapsed Time: {elapsed_time:.2f} s, Heater 1: {lab.Q1()}, Heater 2: {lab.Q2()}, Temp 1: {lab.T1}, Temp 2: {lab.T2}")
        loop_time = labtime.time() - current_time
        labtime.sleep(max(0, dt - loop_time))

    local_start_time = labtime.time()
    lab.Q1(0)
    lab.Q2(0)
    
    while labtime.time() - local_start_time < COOL_OFF_TIME_S:
        current_time = labtime.time()
        elapsed_time = current_time - start_time
        elapsed_time_data.append(elapsed_time)
        heater_1_data.append(lab.Q1())
        heater_2_data.append(lab.Q2())
        temp_1_data.append(lab.T1)
        temp_2_data.append(lab.T2)
        print(f"Elapsed Time: {elapsed_time:.2f} s, Heater 1: {lab.Q1()}, Heater 2: {lab.Q2()}, Temp 1: {lab.T1}, Temp 2: {lab.T2}")
        loop_time = labtime.time() - current_time
        labtime.sleep(max(0, dt - loop_time))

    local_start_time = labtime.time()
    lab.Q1(0)
    lab.Q2(100)

    last_random_temp_control_time = local_start_time

    while time.time() - local_start_time < duration:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if current_time - last_random_temp_control_time > RANDOM_TEMP_CONTROL_DT_S:
            random_temp_1 = np.random.randint(20, 100)
            random_temp_2 = np.random.randint(20, 100)
            lab.Q1(random_temp_1)
            lab.Q2(random_temp_2)
            last_random_temp_control_time = current_time

        elapsed_time_data.append(elapsed_time)
        heater_1_data.append(lab.Q1())
        heater_2_data.append(lab.Q2())
        temp_1_data.append(lab.T1)
        temp_2_data.append(lab.T2)
        print(f"Elapsed Time: {elapsed_time:.2f} s, Heater 1: {lab.Q1()}, Heater 2: {lab.Q2()}, Temp 1: {lab.T1}, Temp 2: {lab.T2}")
        loop_time = time.time() - current_time
        time.sleep(max(0, dt - loop_time))

    lab.close()

    # Save data
    data = np.array([elapsed_time_data, heater_1_data, heater_2_data, temp_1_data, temp_2_data]).T
    # make a new directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"{SAVE_DIR}/{"simulated_" if args.simulation else ""}system_id_data_single_step{timestamp}.csv"
    
    np.savetxt(csv_filename, data, delimiter=",", header="Elapsed Time, Heater 1, Heater 2, Temp 1 (degC), Temp 2 (degC)", comments='')
    print(f"Data saved to {csv_filename}")
    print("Experiment completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initial System ID')
    parser.add_argument('--duration', type=int, default=100, help='Duration of the experiment in seconds')
    parser.add_argument('--dt', type=float, default=1.0, help='Sampling time in seconds')
    parser.add_argument('--simulation', action='store_true', help='Run simulation instead of real experiment')

    args = parser.parse_args()

    main(args)