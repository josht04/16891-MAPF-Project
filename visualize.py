"""
Visualization utilities for wrist RRT* paths.

Usage:
    # Playback in MuJoCo viewer (requires display):
    python visualize.py --trajectory output/wrist_trajectory.npz

    # Plot joint angles vs time:
    python visualize.py --trajectory output/wrist_trajectory.npz --plot
"""
import numpy as np
import argparse
import time
import config


def playback_path(path: list, env=None, fps: float = 10.0):
    """Step through a waypoint path in the MuJoCo viewer.

    Args:
        path: List of arm configs (7D arrays)
        env: MuJoCoEnv instance (if None, creates one)
        fps: Playback speed in frames per second
    """
    if env is None:
        from mujoco_utils import MuJoCoEnv
        env = MuJoCoEnv()

    import mujoco.viewer

    dt = 1.0 / fps
    path_idx = [0]

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            idx = path_idx[0]
            if idx < len(path):
                env.set_arm_config(path[idx])
                path_idx[0] += 1
            time.sleep(dt)
            viewer.sync()


def playback_trajectory(t_uniform: np.ndarray, W: np.ndarray, env=None):
    """Play back a time-parameterized trajectory at real-time speed.

    Args:
        t_uniform: 1D array of timestamps
        W: 2D array (T x 7) of arm configurations
        env: MuJoCoEnv instance
    """
    if env is None:
        from mujoco_utils import MuJoCoEnv
        env = MuJoCoEnv()

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        t_start_real = time.time()
        idx = 0

        while viewer.is_running() and idx < len(t_uniform):
            elapsed_real = time.time() - t_start_real
            # Advance to the right timestep
            while idx < len(t_uniform) - 1 and t_uniform[idx] < elapsed_real:
                idx += 1
            env.set_arm_config(W[idx])
            viewer.sync()
            time.sleep(0.001)  # don't burn CPU

        # Hold final pose
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def plot_trajectory(t_uniform: np.ndarray, W: np.ndarray):
    """Plot joint angles and velocities vs time."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    joint_names = [f"joint_{i+1}" for i in range(7)]

    # Joint angles
    ax = axes[0]
    for j in range(7):
        ax.plot(t_uniform, np.degrees(W[:, j]), label=joint_names[j])
    ax.set_ylabel("Joint angle (deg)")
    ax.set_title("Wrist Trajectory W(t)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Joint velocities
    ax = axes[1]
    if len(t_uniform) > 1:
        dt = np.diff(t_uniform)
        dq = np.diff(W, axis=0)
        vel = dq / dt[:, None]
        t_vel = t_uniform[:-1] + dt / 2  # midpoints

        for j in range(7):
            ax.plot(t_vel, np.degrees(vel[:, j]), label=joint_names[j])
        ax.axhline(y=np.degrees(config.MAX_JOINT_VEL), color="r", linestyle="--",
                    alpha=0.5, label=f"vel limit ({np.degrees(config.MAX_JOINT_VEL):.0f}°/s)")
        ax.axhline(y=-np.degrees(config.MAX_JOINT_VEL), color="r", linestyle="--", alpha=0.5)

    ax.set_ylabel("Joint velocity (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/trajectory_plot.png", dpi=150)
    plt.show()
    print("Saved plot to output/trajectory_plot.png")


def save_trajectory(t_uniform: np.ndarray, W: np.ndarray, path: list,
                    filepath: str = "output/wrist_trajectory.npz"):
    """Save trajectory data for later use (SIPP, visualization, etc)."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.savez(
        filepath,
        t=t_uniform,
        W=W,
        raw_path=np.array(path),
    )
    print(f"Saved trajectory to {filepath}")
    print(f"  {len(t_uniform)} timesteps, total time = {t_uniform[-1]:.2f}s")


def load_trajectory(filepath: str = "output/wrist_trajectory.npz") -> tuple:
    """Load a saved trajectory."""
    data = np.load(filepath)
    return data["t"], data["W"], data["raw_path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize wrist RRT* trajectory")
    parser.add_argument("--trajectory", type=str, default="output/wrist_trajectory.npz",
                        help="Path to saved trajectory .npz file")
    parser.add_argument("--plot", action="store_true", help="Plot joint angles/velocities")
    parser.add_argument("--viewer", action="store_true", help="Playback in MuJoCo viewer")
    args = parser.parse_args()

    t, W, raw_path = load_trajectory(args.trajectory)
    print(f"Loaded trajectory: {len(t)} timesteps, {W.shape[1]} joints, "
          f"total time = {t[-1]:.2f}s")

    if args.plot:
        plot_trajectory(t, W)

    if args.viewer:
        playback_trajectory(t, W)

    if not args.plot and not args.viewer:
        print("Use --plot and/or --viewer to visualize.")