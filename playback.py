"""
Playback combined trajectory (wrist + fingers) using qpos directly.
Matches A* planning exactly — no actuator dynamics.

Usage:
    python playback.py
    python playback.py --trajectory output/combined_trajectory.npy
    python playback.py --speed 0.5
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import config

FINGER_QPOS_SLICES = {
    "first":  slice(7, 11),
    "middle": slice(11, 15),
    "ring":   slice(15, 19),
    "thumb":  slice(19, 23),
}

FINGER_NAMES = ["first", "middle", "ring", "thumb"]


def run_playback(traj_path, speed=1.0):
    traj = np.load(traj_path, allow_pickle=True).item()

    W = traj.get("wrist")
    if W is None:
        raise ValueError("No wrist trajectory found")

    active_fingers = {name: traj[name] for name in FINGER_NAMES
                      if name in traj and traj[name] is not None}

    if not active_fingers:
        print("No finger paths found in trajectory.")
        return

    max_steps = max(len(p) for p in active_fingers.values())
    print(f"Playback: {len(active_fingers)} fingers, {max_steps} steps, "
          f"wrist {len(W)} timesteps, speed={speed}x")

    model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0, 0, 0]

    current_step = 0
    step_interval = 0.5 / speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_step_time = time.time()

        while viewer.is_running():
            current_time = time.time()

            if current_step < max_steps:
                # Set arm qpos directly
                w_idx = min(current_step, len(W) - 1)
                data.qpos[0:7] = W[w_idx]

                # Set inactive fingers to travel pose
                data.qpos[7:23] = config.FINGER_TRAVEL_POSE

                # Set active fingers from planned paths
                if current_time - last_step_time >= step_interval:
                    for finger_name, finger_path in active_fingers.items():
                        if current_step < len(finger_path):
                            _, q_deg = finger_path[current_step]
                            data.qpos[FINGER_QPOS_SLICES[finger_name]] = np.deg2rad(q_deg)
                        else:
                            # Hold final pose
                            _, q_deg = finger_path[-1]
                            data.qpos[FINGER_QPOS_SLICES[finger_name]] = np.deg2rad(q_deg)

                    current_step += 1
                    last_step_time = current_time
                    print(f"  Step {current_step}/{max_steps}", end="\r")

                # Kinematic update only — matches A* exactly
                mujoco.mj_fwdPosition(model, data)

            else:
                time.sleep(1.0)
                current_step = 0
                mujoco.mj_resetData(model, data)
                print("\n--- Restarting playback ---")

            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playback combined trajectory")
    parser.add_argument("--trajectory", type=str,
                        default="output/combined_trajectory.npy")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    args = parser.parse_args()
    run_playback(args.trajectory, args.speed)