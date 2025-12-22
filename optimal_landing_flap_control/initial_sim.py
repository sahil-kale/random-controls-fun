#!/usr/bin/env python3
import math
from dataclasses import dataclass

import jsbsim
import matplotlib.pyplot as plt


FT_TO_M = 0.3048


@dataclass
class Config:
    aircraft: str = "c172r"
    dt: float = 0.01
    t_max: float = 40.0

    TAKEOFF_REJECT_AIRSPEED_M_PER_S: float = 28.0  # ~54 kt
    TAKEOFF_FLAPS_NORM: float = 0.0               # 0..1
    TAKEOFF_THROTTLE_NORM: float = 1.0            # 0..1

    REJECT_BRAKES_NORM: float = 1.0               # 0..1
    REJECT_FLAPS_NORM: float = 0.0                # 0..1 (flaps up)
    REJECT_ELEVATOR_AFT_NORM: float = -1.0        # typical JSBSim: -1 is full up (aft stick)


def prop(sim, *candidates: str) -> str:
    for p in candidates:
        try:
            sim.get_property_value(p)
            return p
        except Exception:
            pass
    raise KeyError(f"No valid JSBSim property found from: {candidates}")


def make_sim(cfg: Config) -> jsbsim.FGFDMExec:
    sim = jsbsim.FGFDMExec(None)
    sim.load_model(cfg.aircraft)

    # ground spawn, runway-ish heading, sea level
    sim.set_property_value("ic/lat-gc-deg", 37.6188056)   # SFO-ish (anywhere is fine)
    sim.set_property_value("ic/long-gc-deg", -122.3754167)
    sim.set_property_value("ic/h-sl-ft", 0.0)
    sim.set_property_value("ic/psi-true-deg", 0.0)
    sim.set_property_value("ic/vc-kts", 0.0)
    sim.set_property_value("ic/u-fps", 0.0)
    sim.set_property_value("ic/v-fps", 0.0)
    sim.set_property_value("ic/w-fps", 0.0)
    sim.set_property_value("ic/roc-fpm", 0.0)

    sim.run_ic()
    sim.set_dt(cfg.dt)
    return sim


def set_takeoff_controls(sim: jsbsim.FGFDMExec, cfg: Config) -> None:
    sim.set_property_value("fcs/throttle-cmd-norm", float(cfg.TAKEOFF_THROTTLE_NORM))
    sim.set_property_value("fcs/flap-cmd-norm", float(cfg.TAKEOFF_FLAPS_NORM))
    sim.set_property_value("fcs/elevator-cmd-norm", 0.0)
    sim.set_property_value("fcs/left-brake-cmd-norm", 0.0)
    sim.set_property_value("fcs/right-brake-cmd-norm", 0.0)


def rejected_takeoff_procedure(sim: jsbsim.FGFDMExec, cfg: Config) -> None:
    sim.set_property_value("fcs/throttle-cmd-norm", 0.0)
    sim.set_property_value("fcs/left-brake-cmd-norm", float(cfg.REJECT_BRAKES_NORM))
    sim.set_property_value("fcs/right-brake-cmd-norm", float(cfg.REJECT_BRAKES_NORM))
    sim.set_property_value("fcs/flap-cmd-norm", float(cfg.REJECT_FLAPS_NORM))
    sim.set_property_value("fcs/elevator-cmd-norm", float(cfg.REJECT_ELEVATOR_AFT_NORM))


def run_rto(sim: jsbsim.FGFDMExec, cfg: Config):
    p_vtrue = prop(sim, "velocities/vtrue-fps", "velocities/vc-fps", "velocities/vt-fps")
    p_vgnd = prop(sim, "velocities/vg-fps", "velocities/vground-fps", "velocities/vgnd-fps")

    p_thr = prop(sim, "fcs/throttle-pos-norm", "fcs/throttle-cmd-norm")
    p_flp = prop(sim, "fcs/flap-pos-norm", "fcs/flap-cmd-norm")
    p_ele = prop(sim, "fcs/elevator-pos-norm", "fcs/elevator-cmd-norm")

    t = 0.0
    rejected = False
    reject_time = math.nan

    s_m = 0.0

    log = {
        "t": [],
        "vtrue_mps": [],
        "vgnd_mps": [],
        "dist_m": [],
        "thr": [],
        "flp": [],
        "ele": [],
    }

    while t <= cfg.t_max:
        sim.run()

        vtrue_mps = sim.get_property_value(p_vtrue) * FT_TO_M
        vgnd_mps = sim.get_property_value(p_vgnd) * FT_TO_M
        s_m += vgnd_mps * cfg.dt

        if (not rejected) and (vtrue_mps >= cfg.TAKEOFF_REJECT_AIRSPEED_M_PER_S):
            rejected = True
            reject_time = t
            rejected_takeoff_procedure(sim, cfg)

        log["t"].append(t)
        log["vtrue_mps"].append(vtrue_mps)
        log["vgnd_mps"].append(vgnd_mps)
        log["dist_m"].append(s_m)
        log["thr"].append(sim.get_property_value(p_thr))
        log["flp"].append(sim.get_property_value(p_flp))
        log["ele"].append(sim.get_property_value(p_ele))

        t += cfg.dt

    return log, reject_time


def plot(log, reject_time: float):
    t = log["t"]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 8))

    # --- speeds ---
    axs[0].plot(t, log["vtrue_mps"], label="airspeed (m/s)")
    axs[0].plot(t, log["vgnd_mps"], label="groundspeed (m/s)")
    if math.isfinite(reject_time):
        axs[0].axvline(reject_time, linestyle="--")
    axs[0].set_ylabel("speed (m/s)")
    axs[0].legend()
    axs[0].grid(True)

    # --- distance ---
    axs[1].plot(t, log["dist_m"])
    if math.isfinite(reject_time):
        axs[1].axvline(reject_time, linestyle="--")
    axs[1].set_ylabel("distance (m)")
    axs[1].grid(True)

    # --- controls ---
    axs[2].plot(t, log["thr"], label="throttle")
    axs[2].plot(t, log["flp"], label="flaps")
    axs[2].plot(t, log["ele"], label="elevator")
    if math.isfinite(reject_time):
        axs[2].axvline(reject_time, linestyle="--", label="reject")
    axs[2].set_ylabel("control (norm)")
    axs[2].set_xlabel("time (s)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()



def main():
    cfg = Config(
        TAKEOFF_REJECT_AIRSPEED_M_PER_S=28.0,
        TAKEOFF_FLAPS_NORM=0.0,
        REJECT_ELEVATOR_AFT_NORM=-1.0,
    )

    sim = make_sim(cfg)
    set_takeoff_controls(sim, cfg)
    log, reject_time = run_rto(sim, cfg)
    plot(log, reject_time)


if __name__ == "__main__":
    main()
