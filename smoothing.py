"""
Post-processing: shortcut smoothing for RRT* paths.
"""
import numpy as np
import config
from mujoco_utils import MuJoCoEnv
from collision import edge_collision_free


def shortcut_smoothing(
    env: MuJoCoEnv,
    path: list,
    max_attempts: int = config.SMOOTHING_ATTEMPTS,
    verbose: bool = True,
) -> list:
    """Random shortcut smoothing: repeatedly try to connect non-adjacent waypoints.

    Args:
        env: MuJoCoEnv instance
        path: List of np.ndarray configs
        max_attempts: Number of random shortcut tries
        verbose: Print progress

    Returns:
        Smoothed path (list of np.ndarray)
    """
    path = [q.copy() for q in path]
    initial_len = len(path)
    initial_cost = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))

    shortcuts_made = 0

    for attempt in range(max_attempts):
        if len(path) < 3:
            break

        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))

        if edge_collision_free(env, path[i], path[j]):
            # Remove intermediate waypoints
            path = path[: i + 1] + path[j:]
            shortcuts_made += 1

    final_cost = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path) - 1))

    if verbose:
        print(f"Smoothing: {initial_len} -> {len(path)} waypoints, "
              f"cost {initial_cost:.4f} -> {final_cost:.4f}, "
              f"{shortcuts_made} shortcuts made")

    return path