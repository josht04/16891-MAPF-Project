import argparse
import numpy as np
import os
import config
from mujoco_utils import MuJoCoEnv
from collision import is_collision_free
from rrt_star import rrt_star
from smoothing import shortcut_smoothing
from time_parameterize import time_parameterize
from visualize import save_trajectory, playback_trajectory, plot_trajectory, load_trajectory
from allegro_astar import AllegroDynamicAStar


def run_wrist_planning(args):
    """RRT* -> smooth -> time parameterize. Returns (t, W, path) or None."""
    env = MuJoCoEnv()
    q_start = config.ARM_START
    q_goal = config.ARM_GOAL

    print(f"Start: {np.array2string(q_start, precision=3)}")
    print(f"Goal:  {np.array2string(q_goal, precision=3)}")
    print(f"L2 distance: {np.linalg.norm(q_goal - q_start):.4f} rad")

    if not is_collision_free(env, q_start):
        print("ERROR: start in collision")
        return None
    if not is_collision_free(env, q_goal):
        print("ERROR: goal in collision")
        return None
    print("Start and goal are collision-free.\n")

    print("Running RRT*...")
    raw_path = rrt_star(env, q_start, q_goal, max_iter=args.max_iter, eta=args.eta)
    if raw_path is None:
        print("No wrist path found.")
        return None

    print("\nSmoothing...")
    smoothed = shortcut_smoothing(env, raw_path)

    print("\nTime parameterization...")
    t_uniform, W = time_parameterize(smoothed)

    save_trajectory(t_uniform, W, smoothed)
    return t_uniform, W, smoothed


def compute_finger_goal_xyz(W_final):
    """FK the goal finger angles at the final wrist pose to get fingertip XYZ."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
    data = mujoco.MjData(model)

    data.qpos[0:7] = W_final
    data.qpos[7:23] = config.FINGER_TRAVEL_POSE

    slices = {"first": slice(7,11), "middle": slice(11,15),
              "ring": slice(15,19), "thumb": slice(19,23)}
    sites = {"first": "ff_grasp", "middle": "mf_grasp",
             "ring": "rf_grasp", "thumb": "th_grasp"}

    for finger, angles in config.FINGER_GOALS_RAD.items():
        data.qpos[slices[finger]] = angles

    mujoco.mj_fwdPosition(model, data)

    goals = {}
    for finger in config.ACTIVE_FINGERS:
        pos = data.site_xpos[model.site(sites[finger]).id].copy()
        goals[finger] = pos
        print(f"  {finger}: goal XYZ = [{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}]")
    return goals


def run_finger_planning(W):
    """Run A* for each active finger against W(t). Returns dict of paths."""
    print("\nComputing goal XYZ from goal joint angles...")
    goal_xyz = compute_finger_goal_xyz(W[-1])

    results = {}

    for finger in config.ACTIVE_FINGERS:
        start_q = config.FINGER_STARTS_DEG[finger]
        g_xyz = goal_xyz[finger]

        print(f"\n{'='*60}")
        print(f"Planning {finger} finger...")
        print(f"  start (deg): {start_q}")
        print(f"  goal (xyz):  {g_xyz}")

        planner = AllegroDynamicAStar(finger)
        path = planner.plan(start_q, g_xyz, W)

        if path is not None:
            results[finger] = path
            print(f"  {finger}: {len(path)} steps")
        else:
            print(f"  {finger}: FAILED")

    return results


def save_combined(t_uniform, W, finger_paths, filepath="output/combined_trajectory.npy"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    out = {"wrist": W, "t": t_uniform}
    for finger, path in finger_paths.items():
        out[finger] = path
    np.save(filepath, out)
    print(f"\nSaved combined trajectory to {filepath}")
    print(f"  Wrist: {W.shape[0]} timesteps")
    for finger, path in finger_paths.items():
        print(f"  {finger}: {len(path)} steps")


def main():
    parser = argparse.ArgumentParser(description="Wrist RRT* + Finger A* Planner")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--max-iter", type=int, default=config.MAX_ITER)
    parser.add_argument("--eta", type=float, default=config.ETA)
    parser.add_argument("--skip-wrist", action="store_true",
                        help="Skip wrist planning, load saved trajectory")
    args = parser.parse_args()

    print("=" * 60)
    print("Wrist RRT* + Finger A* Planner")
    print("=" * 60)

    # --- Wrist ---
    if args.skip_wrist:
        print("\nLoading saved wrist trajectory...")
        t_uniform, W, _ = load_trajectory()
        print(f"  {len(t_uniform)} timesteps, {t_uniform[-1]:.2f}s")
    else:
        result = run_wrist_planning(args)
        if result is None:
            return
        t_uniform, W, _ = result

    # --- Fingers ---
    print(f"\n{'='*60}")
    print("Finger Planning")
    print(f"{'='*60}")
    finger_paths = run_finger_planning(W)

    if not finger_paths:
        print("\nNo finger paths found.")
        return

    # --- Save ---
    save_combined(t_uniform, W, finger_paths)

    # --- Visualize ---
    if args.plot:
        plot_trajectory(t_uniform, W)

    if args.viewer:
        playback_trajectory(t_uniform, W)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()