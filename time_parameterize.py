"""
Time parameterization: convert a waypoint path into W(t) sampled at uniform dt.
"""
import numpy as np
from scipy.interpolate import interp1d
import config


def assign_timestamps(path: list, max_joint_vel: float = config.MAX_JOINT_VEL) -> np.ndarray:
    """Assign timestamps based on max joint velocity.

    Each segment takes time = max(|delta_j|) / max_joint_vel,
    i.e., the slowest joint dictates the segment duration.

    Returns:
        timestamps: 1D array of cumulative times, same length as path.
    """
    timestamps = [0.0]
    for i in range(1, len(path)):
        delta = np.abs(path[i] - path[i - 1])
        # Time for this segment: limited by the joint that moves most
        dt_segment = np.max(delta) / max_joint_vel
        dt_segment = max(dt_segment, 1e-4)  # avoid zero-length segments
        timestamps.append(timestamps[-1] + dt_segment)
    return np.array(timestamps)


def resample_path(
    path: list,
    timestamps: np.ndarray,
    dt: float = config.DT,
) -> tuple:
    """Resample path at uniform time intervals using cubic interpolation.

    Args:
        path: List of np.ndarray (N waypoints, each 7D)
        timestamps: 1D array of times for each waypoint
        dt: Desired uniform timestep

    Returns:
        t_uniform: 1D array of uniform timestamps
        wrist_trajectory: 2D array (T x 7) — the W(t) for SIPP
    """
    path_array = np.array(path)  # (N, 7)

    # Uniform time samples
    t_uniform = np.arange(timestamps[0], timestamps[-1], dt)
    # Always include the final time
    if t_uniform[-1] < timestamps[-1] - 1e-6:
        t_uniform = np.append(t_uniform, timestamps[-1])

    # Cubic interpolation per joint
    # Use linear if too few waypoints for cubic
    kind = "cubic" if len(path) >= 4 else "linear"
    interpolator = interp1d(timestamps, path_array, axis=0, kind=kind)
    wrist_trajectory = interpolator(t_uniform)

    return t_uniform, wrist_trajectory


def time_parameterize(
    path: list,
    max_joint_vel: float = config.MAX_JOINT_VEL,
    dt: float = config.DT,
    verbose: bool = True,
) -> tuple:
    """Full pipeline: assign timestamps, resample.

    Returns:
        t_uniform: 1D array of timestamps
        W: 2D array (T x 7) — wrist trajectory W(t)
    """
    timestamps = assign_timestamps(path, max_joint_vel)
    t_uniform, W = resample_path(path, timestamps, dt)

    if verbose:
        total_time = t_uniform[-1] - t_uniform[0]
        print(f"Time parameterization: {len(path)} waypoints -> {len(t_uniform)} timesteps, "
              f"total time = {total_time:.2f}s, dt = {dt}s")

        # Check velocity profile
        if len(t_uniform) > 1:
            dq = np.diff(W, axis=0)
            dt_arr = np.diff(t_uniform)
            velocities = dq / dt_arr[:, None]
            max_vel = np.max(np.abs(velocities))
            print(f"  Max joint velocity in resampled trajectory: {max_vel:.3f} rad/s "
                  f"(limit: {max_joint_vel:.3f})")

    return t_uniform, W