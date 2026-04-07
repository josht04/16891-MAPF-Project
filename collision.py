"""
Collision checking for wrist RRT* planning.

All checks use the robot (arm + locked fingers in travel pose)
against static obstacles and the floor.
"""
import numpy as np
import config
from mujoco_utils import MuJoCoEnv


def is_collision_free(env: MuJoCoEnv, arm_q: np.ndarray) -> bool:
    """Check if a single arm configuration is collision-free.

    Sets arm joints, locks fingers at travel pose, runs fwdPosition,
    and scans contacts for robot-vs-obstacle collisions.
    """
    env.set_arm_config(arm_q)

    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if env.is_robot_obstacle_contact(contact.geom1, contact.geom2):
            return False
    return True


def is_within_limits(arm_q: np.ndarray) -> bool:
    """Check if arm configuration is within joint limits."""
    return bool(
        np.all(arm_q >= config.ARM_JOINT_MIN) and
        np.all(arm_q <= config.ARM_JOINT_MAX)
    )


def edge_collision_free(
    env: MuJoCoEnv,
    q1: np.ndarray,
    q2: np.ndarray,
    step_size: float = config.COLLISION_STEP,
) -> bool:
    """Check collision along a straight-line edge in joint space.

    Interpolates between q1 and q2 at the given resolution and
    checks each intermediate configuration.
    """
    diff = q2 - q1
    dist = np.linalg.norm(diff)
    if dist < 1e-8:
        return is_collision_free(env, q1)

    n_steps = max(int(np.ceil(dist / step_size)), 1)

    for i in range(n_steps + 1):
        alpha = i / n_steps
        q_interp = q1 + alpha * diff
        if not is_collision_free(env, q_interp):
            return False
    return True