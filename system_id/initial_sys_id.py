import tclab
import numpy as np
import argparse
import os
import time
from tclab import labtime

SAVE_DIR = "system_id/data"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_experiment(lab, steps, dt, start_time, label):
    data = []
    print(f"Starting experiment: {label}")
    
    for ref_time, q1, q2 in steps:
        local_start = labtime.time()
        while labtime.time() - local_start < ref_time:
            current_time = labtime.time()
            elapsed_time = current_time - start_time
            lab.Q1(q1)
            lab.Q2(q2)
            data.append([
                elapsed_time, lab.Q1(), lab.Q2(), lab.T1, lab.T2
            ])
            print(f"{label} | t: {elapsed_time:.2f}s | Q1: {lab.Q1()} | Q2: {lab.Q2()} | T1: {lab.T1:.2f} | T2: {lab.T2:.2f}")
            loop_time = labtime.time() - current_time
            labtime.sleep(max(0, dt - loop_time))

    return np.array(data)

def main(args):
    np.random.seed(42)
    ensure_dir(SAVE_DIR)

    if args.simulation:
        lab = tclab.setup(connected=False, speedup=12)()
        print("Running in simulation mode")
    else:
        lab = tclab.TCLab()

    lab.LED(0)
    labtime.sleep(1)
    lab.LED(100)
    labtime.sleep(1)

    dt = args.dt
    start_time = labtime.time()
    all_data = []

    experiments = [
        ("Q1 100% Step", [(args.duration, 100, 0)]),
        ("Cool Off 1", [(args.cool_off_time, 0, 0)]),
        ("Q2 100% Step", [(args.duration, 0, 100)]),
        ("Cool Off 2", [(args.cool_off_time, 0, 0)]),
        ("Random Control", [])
    ]

    # Generate random heater commands
    last_time = 0
    while last_time < args.duration:
        stepsize = min(args.random_temp_control_time, args.duration - last_time)
        q1 = np.random.randint(20, 100)
        q2 = np.random.randint(20, 100)
        experiments[-1][1].append((stepsize, q1, q2))
        last_time += stepsize

    for label, steps in experiments:
        data = run_experiment(lab, steps, dt, start_time, label)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(SAVE_DIR, f"{'simulated_' if args.simulation else ''}{label.replace(' ', '_').lower()}_{timestamp}.csv")
        np.savetxt(filename, data, delimiter=",", header="Elapsed Time, Heater 1, Heater 2, Temp 1 (degC), Temp 2 (degC)", comments='')
        print(f"Saved data for {label} to {filename}")
        all_data.append(data)

    lab.Q1(0)
    lab.Q2(0)
    lab.close()

    # Combine all data
    combined_data = np.vstack(all_data)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    combined_filename = os.path.join(SAVE_DIR, f"{'simulated_' if args.simulation else ''}combined_system_id_{timestamp}.csv")
    np.savetxt(combined_filename, combined_data, delimiter=",", header="Elapsed Time, Heater 1, Heater 2, Temp 1 (degC), Temp 2 (degC)", comments='')
    print(f"All data combined and saved to {combined_filename}")
    print("Experiment completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='System Identification Experiment Framework')
    parser.add_argument('--duration', type=int, default=300, help='Duration of each heating experiment in seconds')
    parser.add_argument('--random_temp_control_time', type=int, default=45, help='Interval for random heater updates')
    parser.add_argument('--cool_off_time', type=int, default=300, help='Cool off period between experiments')
    parser.add_argument('--dt', type=float, default=1.0, help='Sampling time in seconds')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    args = parser.parse_args()
    main(args)
