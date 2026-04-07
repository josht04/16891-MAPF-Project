"""
Sanity checks for collision detection.
Run this before trusting the planner.
"""
import numpy as np
from mujoco_utils import MuJoCoEnv
from collision import is_collision_free, is_within_limits, edge_collision_free
import config


def test_known_free():
    """Home pose should be collision-free."""
    env = MuJoCoEnv()
    assert is_collision_free(env, config.ARM_START), "FAIL: home pose should be free"
    print("  PASS: home pose is collision-free")


def test_known_colliding():
    """A config that clearly penetrates an obstacle."""
    env = MuJoCoEnv()
    # Rack middle shelf is at x≈0.55, z≈0.55 (world).
    # Extend arm straight forward and low to hit it.
    # Try a range of joint_2 / joint_4 values to find one that collides.
    found_collision = False
    for j2 in np.linspace(-1.0, 2.0, 30):
        for j4 in np.linspace(-2.5, 2.5, 30):
            q = np.array([0.0, j2, 3.14, j4, 0.0, 0.5, 1.57])
            if is_within_limits(q) and not is_collision_free(env, q):
                found_collision = True
                # Report what it hit
                env.set_arm_config(q)
                for i in range(env.data.ncon):
                    c = env.data.contact[i]
                    if env.is_robot_obstacle_contact(c.geom1, c.geom2):
                        g1 = env.model.geom(c.geom1).name or f"geom_{c.geom1}"
                        g2 = env.model.geom(c.geom2).name or f"geom_{c.geom2}"
                        print(f"  PASS: found collision at j2={j2:.2f}, j4={j4:.2f}: {g1} <-> {g2}")
                        return
    if not found_collision:
        print("  WARN: could not find a colliding config by grid search. "
              "Obstacles may be too far or joint ranges too narrow.")


def test_edge_through_obstacle():
    """Find two free configs where the straight line between them passes through an obstacle."""
    env = MuJoCoEnv()
    # Strategy: find a free config on each side of an obstacle
    q_a = config.ARM_START.copy()
    
    # Try configs that reach past an obstacle
    found = False
    for j2 in np.linspace(0.3, 1.5, 20):
        for j4 in np.linspace(-2.0, -0.5, 20):
            q_b = np.array([0.0, j2, 3.14, j4, 0.0, 0.5, 1.57])
            if not is_within_limits(q_b):
                continue
            if not is_collision_free(env, q_b):
                continue
            # Both q_a and q_b are free. Check if edge between them is blocked.
            if not edge_collision_free(env, q_a, q_b):
                print(f"  PASS: edge from start to j2={j2:.2f},j4={j4:.2f} is blocked "
                      f"(both endpoints free, intermediate collision)")
                found = True
                break
        if found:
            break
    if not found:
        print("  INFO: no blocked edge found between start and tested configs. "
              "This is OK if obstacles don't lie between them.")


def test_limits():
    """Joint limit checks."""
    env = MuJoCoEnv()
    assert is_within_limits(config.ARM_START), "FAIL: start outside limits"
    assert is_within_limits(config.ARM_GOAL), "FAIL: goal outside limits"

    # Over-limit should fail
    q_bad = config.ARM_JOINT_MAX + 0.1
    assert not is_within_limits(q_bad), "FAIL: over-limit config should fail"

    q_bad = config.ARM_JOINT_MIN - 0.1
    assert not is_within_limits(q_bad), "FAIL: under-limit config should fail"
    print("  PASS: joint limit checks")


def test_floor_collision():
    """Arm reaching below the floor should be flagged."""
    env = MuJoCoEnv()
    # Base is at z=0.5. Pointing straight down might hit the floor.
    q_down = np.array([0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if is_within_limits(q_down):
        free = is_collision_free(env, q_down)
        if not free:
            print("  PASS: downward-pointing arm hits floor")
        else:
            print("  INFO: downward config doesn't hit floor (arm may not reach it)")
    else:
        print("  SKIP: downward config outside joint limits")


if __name__ == "__main__":
    print("Running collision sanity checks...\n")

    print("1. Known free config (home):")
    test_known_free()

    print("\n2. Known colliding config (grid search):")
    test_known_colliding()

    print("\n3. Edge through obstacle:")
    test_edge_through_obstacle()

    print("\n4. Joint limits:")
    test_limits()

    print("\n5. Floor collision:")
    test_floor_collision()

    print("\nAll checks complete.")