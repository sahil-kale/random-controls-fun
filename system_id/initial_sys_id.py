import tclab
import numpy as np
import time
import argparse
import os

RANDOM_TEMP_CONTROL_DT_S = 30
SAVE_DIR = "system_id/data"

def main(args):
    np.random.seed(42)
    lab = tclab.TCLab()
    print(f"Starting Experiment")
    lab.LED(0)
    time.sleep(1)
    lab.LED(100)
    time.sleep(1)

    elapsed_time_data = []
    heater_1_data = []
    heater_2_data = []
    temp_1_data = []
    temp_2_data = []

    lab.Q1(100)
    lab.Q2(100)
    start_time = time.time()
    duration = args.duration
    dt = args.dt
    last_random_temp_control_time = start_time

    while time.time() - start_time < duration:
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

    lab.Q1(0)
    lab.Q2(0)
    lab.close()

    # Save data
    data = np.array([elapsed_time_data, heater_1_data, heater_2_data, temp_1_data, temp_2_data]).T
    # make a new directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"{SAVE_DIR}/system_id_data_{timestamp}.csv"
    
    np.savetxt(csv_filename, data, delimiter=",", header="Elapsed Time, Heater 1, Heater 2, Temp 1 (degC), Temp 2 (degC)", comments='')
    print(f"Data saved to {SAVE_DIR}/system_id_data.csv")
    print("Experiment completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initial System ID')
    parser.add_argument('--duration', type=int, default=100, help='Duration of the experiment in seconds')
    parser.add_argument('--dt', type=float, default=1.0, help='Sampling time in seconds')

    args = parser.parse_args()

    main(args)