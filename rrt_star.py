"""
RRT* planner in arm joint space.

Returns a raw path (list of np.ndarray configurations) from start to goal,
or None if no path is found within max_iter.
"""
import numpy as np
from scipy.spatial import KDTree
import time

import config
from mujoco_utils import MuJoCoEnv
from collision import is_collision_free, is_within_limits, edge_collision_free


class Node:
    __slots__ = ["q", "cost", "parent", "children", "idx"]

    def __init__(self, q: np.ndarray, cost: float = 0.0, parent=None, idx: int = 0):
        self.q = q
        self.cost = cost
        self.parent = parent
        self.children = []
        self.idx = idx


def _sample(q_goal: np.ndarray, goal_bias: float) -> np.ndarray:
    """Sample a random configuration, with goal bias."""
    if np.random.rand() < goal_bias:
        return q_goal.copy()
    return np.random.uniform(config.ARM_JOINT_MIN, config.ARM_JOINT_MAX)


def _steer(q_from: np.ndarray, q_to: np.ndarray, eta: float) -> np.ndarray:
    """Take a step of at most eta from q_from toward q_to."""
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist <= eta:
        return q_to.copy()
    return q_from + (diff / dist) * eta


def _cost_edge(q1: np.ndarray, q2: np.ndarray) -> float:
    """Edge cost = Euclidean distance in joint space."""
    return float(np.linalg.norm(q2 - q1))


def _propagate_cost(node: Node):
    """Recursively update costs in the subtree after a rewire."""
    for child in node.children:
        child.cost = node.cost + _cost_edge(node.q, child.q)
        _propagate_cost(child)


def rrt_star(
    env: MuJoCoEnv,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    max_iter: int = config.MAX_ITER,
    eta: float = config.ETA,
    gamma_rrt: float = config.GAMMA_RRT,
    goal_bias: float = config.GOAL_BIAS,
    goal_threshold: float = config.GOAL_THRESHOLD,
    verbose: bool = True,
) -> list | None:
    """Run RRT* and return a path (list of configs) or None.

    Args:
        env: MuJoCoEnv instance (collision checker)
        q_start: Start arm configuration (7D)
        q_goal: Goal arm configuration (7D)
        max_iter: Maximum iterations
        eta: Step size
        gamma_rrt: Rewiring radius scaling
        goal_bias: Probability of sampling goal
        goal_threshold: L2 distance to count as goal reached
        verbose: Print progress

    Returns:
        List of np.ndarray (path from start to goal) or None
    """
    dim = len(q_start)

    # Validate start and goal
    if not is_collision_free(env, q_start):
        print("ERROR: Start configuration is in collision!")
        return None
    if not is_collision_free(env, q_goal):
        print("ERROR: Goal configuration is in collision!")
        return None

    root = Node(q_start.copy(), cost=0.0, idx=0)
    nodes = [root]
    configs_array = [q_start.copy()]  # for KDTree rebuilds

    best_goal_node = None
    best_goal_cost = float("inf")

    # Rebuild KDTree periodically for performance
    kdtree = KDTree([q_start])
    kdtree_rebuild_interval = 200
    kdtree_stale = False

    t_start = time.time()

    for k in range(1, max_iter + 1):
        # --- Sample ---
        q_rand = _sample(q_goal, goal_bias)

        # --- Nearest ---
        if kdtree_stale and k % kdtree_rebuild_interval == 0:
            kdtree = KDTree(np.array(configs_array))
            kdtree_stale = False
        _, nearest_idx = kdtree.query(q_rand)
        q_nearest_node = nodes[nearest_idx]

        # --- Steer ---
        q_new = _steer(q_nearest_node.q, q_rand, eta)

        if not is_within_limits(q_new):
            continue

        # --- Check edge to nearest ---
        if not edge_collision_free(env, q_nearest_node.q, q_new):
            continue

        # --- Find nearby nodes for rewiring ---
        n_nodes = len(nodes)
        r = min(gamma_rrt * (np.log(n_nodes) / n_nodes) ** (1.0 / dim), eta)
        r = max(r, 0.01)  # minimum radius

        neighbor_idxs = kdtree.query_ball_point(q_new, r)

        # --- Choose best parent ---
        best_parent = q_nearest_node
        best_cost = q_nearest_node.cost + _cost_edge(q_nearest_node.q, q_new)

        for ni in neighbor_idxs:
            nb = nodes[ni]
            candidate_cost = nb.cost + _cost_edge(nb.q, q_new)
            if candidate_cost < best_cost:
                if edge_collision_free(env, nb.q, q_new):
                    best_parent = nb
                    best_cost = candidate_cost

        # --- Insert new node ---
        new_node = Node(q_new, cost=best_cost, parent=best_parent, idx=n_nodes)
        best_parent.children.append(new_node)
        nodes.append(new_node)
        configs_array.append(q_new.copy())
        kdtree_stale = True

        # --- Rewire ---
        for ni in neighbor_idxs:
            nb = nodes[ni]
            if nb is best_parent:
                continue
            candidate_cost = new_node.cost + _cost_edge(new_node.q, nb.q)
            if candidate_cost < nb.cost:
                if edge_collision_free(env, new_node.q, nb.q):
                    # Rewire
                    nb.parent.children.remove(nb)
                    nb.parent = new_node
                    nb.cost = candidate_cost
                    new_node.children.append(nb)
                    _propagate_cost(nb)

        # --- Check goal ---
        dist_to_goal = np.linalg.norm(q_new - q_goal)
        if dist_to_goal < goal_threshold:
            if new_node.cost < best_goal_cost:
                best_goal_node = new_node
                best_goal_cost = new_node.cost
                if verbose:
                    print(f"  [iter {k}] Goal reached! cost={best_goal_cost:.4f}, "
                          f"nodes={len(nodes)}, dist={dist_to_goal:.4f}")

        # --- Progress ---
        if verbose and k % 1000 == 0:
            elapsed = time.time() - t_start
            status = f"best_cost={best_goal_cost:.4f}" if best_goal_node else "no path yet"
            print(f"  [iter {k}] nodes={len(nodes)}, {status}, time={elapsed:.1f}s")

    # --- Extract path ---
    if best_goal_node is None:
        if verbose:
            print(f"No path found after {max_iter} iterations.")
        return None

    # Trace back to root
    path = []
    node = best_goal_node
    while node is not None:
        path.append(node.q.copy())
        node = node.parent
    path.reverse()

    # Append exact goal if last node isn't close enough
    if np.linalg.norm(path[-1] - q_goal) > 1e-6:
        if edge_collision_free(env, path[-1], q_goal):
            path.append(q_goal.copy())

    elapsed = time.time() - t_start
    if verbose:
        print(f"Path found: {len(path)} waypoints, cost={best_goal_cost:.4f}, "
              f"nodes={len(nodes)}, time={elapsed:.1f}s")

    return path