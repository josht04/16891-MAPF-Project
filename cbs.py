"""
Conflict-Based Search for multi-finger planning.
Takes independent finger paths, detects inter-finger collisions,
and replans with constraints until conflict-free.
"""
import mujoco
import numpy as np
import heapq
from copy import deepcopy

import config
from allegro_astar import AllegroDynamicAStar, FINGER_QPOS_SLICES, FINGER_BODY_NAMES, SITE_NAMES

MAX_CBS_ITERATIONS = 50


class CBSNode:
    def __init__(self, paths, constraints, cost):
        self.paths = paths          # {finger_name: [(t, joint_tuple), ...]}
        self.constraints = constraints  # [{"finger":..., "timestep":..., "joints":...}]
        self.cost = cost            # sum of path lengths

    def __lt__(self, other):
        return self.cost < other.cost


def build_finger_geom_sets(model):
    """Map each finger to its geom IDs for pairwise collision checking."""
    geom_sets = {}
    for finger, body_names in FINGER_BODY_NAMES.items():
        body_ids = {model.body(b).id for b in body_names}
        gids = set()
        for gi in range(model.ngeom):
            if model.geom(gi).bodyid[0] in body_ids and model.geom(gi).contype[0] > 0:
                gids.add(gi)
        geom_sets[finger] = gids
    return geom_sets


def find_first_conflict(paths, W, active_fingers):
    """Find the first timestep where two fingers collide.

    Sets all fingers simultaneously and checks for inter-finger contacts.
    Returns (finger_a, finger_b, timestep, joints_a, joints_b) or None.
    """
    model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
    data = mujoco.MjData(model)
    geom_sets = build_finger_geom_sets(model)

    max_t = max(len(paths[f]) for f in active_fingers)

    for t in range(max_t):
        w_idx = min(t, len(W) - 1)
        data.qpos[0:7] = W[w_idx]
        data.qpos[7:23] = config.FINGER_TRAVEL_POSE

        for finger in active_fingers:
            if t < len(paths[finger]):
                _, q = paths[finger][t]
            else:
                _, q = paths[finger][-1]
            data.qpos[FINGER_QPOS_SLICES[finger]] = np.deg2rad(q)

        mujoco.mj_fwdPosition(model, data)

        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            for fi, fa in enumerate(active_fingers):
                for fb in active_fingers[fi+1:]:
                    a_hit = (g1 in geom_sets[fa] or g2 in geom_sets[fa])
                    b_hit = (g1 in geom_sets[fb] or g2 in geom_sets[fb])
                    if a_hit and b_hit:
                        qa = paths[fa][min(t, len(paths[fa])-1)][1]
                        qb = paths[fb][min(t, len(paths[fb])-1)][1]
                        return (fa, fb, t, qa, qb)
    return None


def total_cost(paths, active_fingers):
    return sum(len(paths[f]) for f in active_fingers)


def cbs(finger_paths, W, goal_xyz, verbose=True):
    """Run CBS over pre-computed finger paths.

    Args:
        finger_paths: {finger_name: [(t, joint_tuple), ...]} from initial A* runs
        W: (T, 7) wrist trajectory
        goal_xyz: {finger_name: np.array([x,y,z])}

    Returns:
        Conflict-free finger_paths dict, or None if CBS fails.
    """
    active = [f for f in config.ACTIVE_FINGERS if f in finger_paths]

    root = CBSNode(
        paths=deepcopy(finger_paths),
        constraints=[],
        cost=total_cost(finger_paths, active)
    )

    pq = [(root.cost, 0, root)]
    node_id = 1

    for iteration in range(MAX_CBS_ITERATIONS):
        if not pq:
            print("CBS: open list empty, no solution.")
            return None

        _, _, node = heapq.heappop(pq)

        if verbose:
            print(f"\nCBS iter {iteration}: cost={node.cost}, "
                  f"constraints={len(node.constraints)}")

        conflict = find_first_conflict(node.paths, W, active)

        if conflict is None:
            if verbose:
                print("CBS: no conflicts, done!")
            return node.paths

        fa, fb, t, qa, qb = conflict
        if verbose:
            print(f"  Conflict: {fa} vs {fb} at t={t}")

        # Branch: constrain each finger in turn
        for constrained, blocked_q in [(fa, qa), (fb, qb)]:
            new_constraints = node.constraints + [
                {"finger": constrained, "timestep": t, "joints": blocked_q}
            ]

            if verbose:
                print(f"  Replanning {constrained} with {len(new_constraints)} constraints...")

            planner = AllegroDynamicAStar(
                constrained,
                constraints=new_constraints
            )

            start_q = config.FINGER_STARTS_DEG[constrained]
            g_xyz = goal_xyz[constrained]
            new_path = planner.plan(start_q, g_xyz, W)

            if new_path is None:
                if verbose:
                    print(f"  {constrained} replan failed, pruning branch.")
                continue

            new_paths = deepcopy(node.paths)
            new_paths[constrained] = new_path

            child = CBSNode(
                paths=new_paths,
                constraints=new_constraints,
                cost=total_cost(new_paths, active)
            )

            heapq.heappush(pq, (child.cost, node_id, child))
            node_id += 1

    print(f"CBS: hit iteration limit ({MAX_CBS_ITERATIONS}).")
    return None