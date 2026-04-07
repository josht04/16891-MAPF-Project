"""
Space-Time A* planner for individual Allegro fingers on the combined Kinova+Allegro model.

Wrist is positioned via qpos[0:7] from the RRT* wrist trajectory.
Finger joints at qpos[7:23]. Collision filtered to: this finger vs obstacles only.
"""
from collections import defaultdict

import mujoco
import numpy as np
import heapq

import config

SCENE_XML = config.MODEL_PATH

SITE_NAMES = {
    "thumb": "th_grasp",
    "first": "ff_grasp",
    "middle": "mf_grasp",
    "ring": "rf_grasp",
}

MAX_STEP_DICT = {
    "thumb": 0.0013,
    "first": 0.00165,
    "middle": 0.00165,
    "ring": 0.00165,
}

FINGER_QPOS_SLICES = {
    "first":  slice(7, 11),
    "middle": slice(11, 15),
    "ring":   slice(15, 19),
    "thumb":  slice(19, 23),
}

FINGER_BODY_NAMES = {
    "first":  ["ff_base", "ff_proximal", "ff_medial", "ff_distal", "ff_tip"],
    "middle": ["mf_base", "mf_proximal", "mf_medial", "mf_distal", "mf_tip"],
    "ring":   ["rf_base", "rf_proximal", "rf_medial", "rf_distal", "rf_tip"],
    "thumb":  ["th_base", "th_proximal", "th_medial", "th_distal", "th_tip"],
}

ACTUATOR_PREFIXES = {
    "first": "ff",
    "middle": "mf",
    "ring": "rf",
    "thumb": "th",
}


class AllegroDynamicAStar:
    def __init__(self, finger_type, site_name=None, max_step_dist=None,
                 constraints=None, xml_path=None):
        xml = xml_path or SCENE_XML
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.finger_type = finger_type

        site = site_name or SITE_NAMES[finger_type]
        self.site_id = self.model.site(site).id
        self.max_step_dist = max_step_dist or MAX_STEP_DICT[finger_type]
        self.finger_slice = FINGER_QPOS_SLICES[finger_type]

        # Build collision filter: this finger's geoms vs obstacle+floor geoms
        finger_body_ids = set()
        for bname in FINGER_BODY_NAMES[finger_type]:
            finger_body_ids.add(self.model.body(bname).id)

        self.finger_geom_ids = set()
        self.obstacle_geom_ids = set()
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        for gi in range(self.model.ngeom):
            ct = self.model.geom(gi).contype[0]
            body_id = self.model.geom(gi).bodyid[0]
            if body_id in finger_body_ids and ct > 0:
                self.finger_geom_ids.add(gi)
            if ct == 4:
                self.obstacle_geom_ids.add(gi)

        if floor_id >= 0:
            self.obstacle_geom_ids.add(floor_id)

        # Joint limits with 2° buffer (integer degrees)
        prefix = ACTUATOR_PREFIXES[finger_type]
        act_names = [f"{prefix}a{i}" for i in range(4)]
        self.limits = []
        for name in act_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            rng = self.model.actuator_ctrlrange[act_id]
            self.limits.append((int(np.rad2deg(rng[0]) + 2), int(np.rad2deg(rng[1]) - 2)))

        # CBS constraints
        self.constraints = defaultdict(set)
        if constraints:
            for c in constraints:
                if c["finger"] == finger_type:
                    self.constraints[c["timestep"]].add(tuple(c["joints"]))

        # Moves: all 1° steps in 4D INCLUDING wait (zero-move)
        self._moves = [tuple(m) for m in
                       np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
                       .T.reshape(-1, 4)]

    def set_context(self, step_idx, wrist_traj, finger_q_deg):
        w_idx = min(step_idx, len(wrist_traj) - 1)
        self.data.qpos[0:7] = wrist_traj[w_idx]
        self.data.qpos[7:23] = config.FINGER_TRAVEL_POSE
        self.data.qpos[self.finger_slice] = np.deg2rad(finger_q_deg)
        mujoco.mj_fwdPosition(self.model, self.data)

    def is_valid(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            finger_hit = (g1 in self.finger_geom_ids or g2 in self.finger_geom_ids)
            obstacle_hit = (g1 in self.obstacle_geom_ids or g2 in self.obstacle_geom_ids)
            if finger_hit and obstacle_hit:
                return False
        return True

    def plan(self, start_q, goal_xyz, wrist_traj,
             tolerance=None, max_iters=None, heuristic_weight=None):
        tolerance = tolerance or config.FINGER_TOLERANCE
        max_iters = max_iters or config.FINGER_MAX_ITERS
        weight = heuristic_weight or config.FINGER_HEURISTIC_WEIGHT

        goal_xyz = np.asarray(goal_xyz)
        start_node = (0, tuple(start_q))

        self.set_context(0, wrist_traj, start_q)
        h0 = np.linalg.norm(self.data.site_xpos[self.site_id] - goal_xyz) / self.max_step_dist

        pq = [(h0, 0, start_node)]
        came_from = {start_node: None}
        g_score = {start_node: 0}

        iters = 0
        while pq and iters < max_iters:
            iters += 1
            f, g, (t, curr_q) = heapq.heappop(pq)

            self.set_context(t, wrist_traj, curr_q)
            curr_p = self.data.site_xpos[self.site_id].copy()
            dist = np.linalg.norm(curr_p - goal_xyz)

            if iters % 5000 == 0:
                print(f"  [{self.finger_type}] iter={iters} step={t} dist={dist:.4f}m")

            if dist < tolerance:
                print(f"  [{self.finger_type}] path found! steps={t} iters={iters}")
                path = []
                curr = (t, curr_q)
                while curr:
                    path.append(curr)
                    curr = came_from[curr]
                return path[::-1]

            next_t = t + 1
            forbidden = self.constraints.get(next_t)

            for move in self._moves:
                next_q = tuple(curr_q[i] + move[i] for i in range(4))

                if forbidden is not None and next_q in forbidden:
                    continue

                if any(next_q[i] < self.limits[i][0] or next_q[i] > self.limits[i][1]
                       for i in range(4)):
                    continue

                next_node = (next_t, next_q)
                if next_node not in g_score:
                    self.set_context(next_t, wrist_traj, next_q)
                    if self.is_valid():
                        g_score[next_node] = next_t
                        h = np.linalg.norm(self.data.site_xpos[self.site_id] - goal_xyz) \
                            / self.max_step_dist * weight
                        heapq.heappush(pq, (next_t + h, next_t, next_node))
                        came_from[next_node] = (t, curr_q)

        print(f"  [{self.finger_type}] failed after {iters} iterations")
        return None