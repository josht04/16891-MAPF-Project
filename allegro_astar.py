from collections import defaultdict

import mujoco
import numpy as np
import heapq
from scipy.spatial.transform import Rotation as R

class AllegroDynamicAStar:
    def __init__(self, xml_path, finger_type, site_name, max_step_dist, constraints=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(site_name).id
        self.mocap_id = self.model.body_mocapid[self.model.body('palm').id]
        self.max_step_dist = max_step_dist
        
        self.constraints = defaultdict(set)

        if constraints:
            for constraint in constraints:
                if constraint["finger"] == finger_type:
                    self.constraints[constraint["timestep"]].add(tuple(constraint["joints"]))

        # Mapping for joint indices (assuming mocap palm, so joints start at 0)
        prefix = {"thumb": "th", "first": "ff", "middle": "mf", "ring": "rf"}[finger_type]
        self.joint_names = [f"{prefix}a{i}" for i in range(4)]
        
        self.limits = []
        for name in self.joint_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            rng = self.model.actuator_ctrlrange[act_id]
            self.limits.append((int(np.rad2deg(rng[0])+2), int(np.rad2deg(rng[1])-2)))

    def set_context(self, step_idx, wrist_path, finger_q_deg):
        """Updates the MuJoCo world to a specific time step's wrist and finger state."""
        # 1. Set Wrist from Path
        pos, euler = wrist_path[min(step_idx, len(wrist_path)-1)]
        self.data.mocap_pos[self.mocap_id] = pos
        quat = R.from_euler('xyz', euler, degrees=True).as_quat()
        self.data.mocap_quat[self.mocap_id] = [quat[3], quat[0], quat[1], quat[2]]
        
        # 2. Set Fingers
        self.data.qpos[0:4] = np.deg2rad(finger_q_deg)
        
        # 3. Sync Physics
        mujoco.mj_forward(self.model, self.data)

    def is_valid(self):
        if self.data.ncon == 0:
            return True
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1_id = self.model.geom_bodyid[contact.geom1]
            b2_id = self.model.geom_bodyid[contact.geom2]
            
            if b1_id == 0 or b2_id == 0:
                continue
                
            return False
            
        return True

    def plan(self, start_q, goal_xyz, wrist_path, tolerance=0.005, max_iters=5000000):
        start_node = (0, tuple(start_q))
        self.set_context(0, wrist_path, start_q)
        
        pq = [(np.linalg.norm(self.data.site_xpos[self.site_id] - goal_xyz)/self.max_step_dist, 0, start_node)]
        came_from = {start_node: None}
        g_score = {start_node: 0}
        
        moves = [tuple(m) for m in np.array(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1])).T.reshape(-1,4) if not np.all(m==0)]

        iters = 0
        while pq and iters < max_iters:
            iters += 1
            f, g, (t, curr_q) = heapq.heappop(pq)
            
            # Use min() to stay at the final wrist position if t exceeds path length
            wrist_idx = min(t, len(wrist_path) - 1)
            self.set_context(wrist_idx, wrist_path, curr_q)
            curr_p = self.data.site_xpos[self.site_id].copy()
            dist = np.linalg.norm(curr_p - goal_xyz)
            
            # --- Status Prints ---
            if iters % 5000 == 0:
                print(f"Iter: {iters} | Step: {t} | Dist: {dist:.4f}m", end='\n')

            if dist < tolerance:
                print(f"\nPath Found! Steps: {t} | Iters: {iters}")
                path = []
                curr = (t, curr_q)
                while curr:
                    path.append(curr)
                    curr = came_from[curr]
                return path[::-1]

            next_t = t + 1
            constraints = self.constraints.get(next_t)
            for move in moves:
                next_q = tuple(curr_q[i] + move[i] for i in range(4))

                if constraints is not None and next_q in constraints:
                    continue 
                
                if any(next_q[i] < self.limits[i][0] or next_q[i] > self.limits[i][1] for i in range(4)):
                    continue
                
                next_node = (next_t, next_q)
                if next_node not in g_score:
                    # Sync physics for the next potential state
                    w_idx = min(next_t, len(wrist_path) - 1)
                    self.set_context(w_idx, wrist_path, next_q)
                    
                    if self.is_valid():
                        g_score[next_node] = next_t
                        weight = 5
                        h = np.linalg.norm(self.data.site_xpos[self.site_id] - goal_xyz) / self.max_step_dist * weight
                        heapq.heappush(pq, (next_t + h, next_t, next_node))
                        came_from[next_node] = (t, curr_q)
                        
        print(f"\nFailed after {iters} iterations.")
        return None
    
# --- Main Configuration ---
def generate_wrist_path(start_pos, end_pos, start_euler, end_euler, duration, dt=0.1):
    steps = int(duration / dt)
    path = []
    
    for i in range(steps + 1):
        # Calculate interpolation factor (0.0 to 1.0)
        t = i / steps
        
        # Linearly interpolate position: P = P0 + t * (P1 - P0)
        current_pos = [
            start_pos[0] + t * (end_pos[0] - start_pos[0]),
            start_pos[1] + t * (end_pos[1] - start_pos[1]),
            start_pos[2] + t * (end_pos[2] - start_pos[2])
        ]
        
        # Linearly interpolate euler angles
        current_euler = [
            start_euler[0] + t * (end_euler[0] - start_euler[0]),
            start_euler[1] + t * (end_euler[1] - start_euler[1]),
            start_euler[2] + t * (end_euler[2] - start_euler[2])
        ]
        
        path.append((current_pos, current_euler))
    
    return path

MAX_STEP_DICT = {
    "thumb": 0.0013,
    "first": 0.00165,
    "middle": 0.00165,
    "ring": 0.00165,
}

XML_PATHS = {
    "thumb": r'.\wonik_allegro\left_hand_thumb.xml',
    "first": r'.\wonik_allegro\left_hand_first.xml',
    "middle": r'.\wonik_allegro\left_hand_middle.xml',
    "ring": r'.\wonik_allegro\left_hand_ring.xml',
    "thumb_obs": r'.\wonik_allegro\obstacle_ex1\left_hand_thumb_obstacle.xml',
    "first_obs": r'.\wonik_allegro\obstacle_ex1\left_hand_first_obstacle.xml',
    "middle_obs": r'.\wonik_allegro\obstacle_ex1\left_hand_middle_obstacle.xml',
    "ring_obs": r'.\wonik_allegro\obstacle_ex1\left_hand_ring_obstacle.xml',
    "thumb_closed_grab": r'.\wonik_allegro\closed_grab\left_hand_closed_grab_thumb.xml',
    "first_closed_grab": r'.\wonik_allegro\closed_grab\left_hand_closed_grab_first.xml',
    "middle_closed_grab": r'.\wonik_allegro\closed_grab\left_hand_closed_grab_middle.xml',
    "ring_closed_grab": r'.\wonik_allegro\closed_grab\left_hand_closed_grab_ring.xml',
    "thumb_doorknob": r'.\wonik_allegro\doorknob\left_hand_doorknob_thumb.xml',
    "first_doorknob": r'.\wonik_allegro\doorknob\left_hand_doorknob_first.xml',
    "middle_doorknob": r'.\wonik_allegro\doorknob\left_hand_doorknob_middle.xml',
    "ring_doorknob": r'.\wonik_allegro\doorknob\left_hand_doorknob_ring.xml',
}

SITE_NAMES = {
    "thumb": "th_grasp",
    "first": "ff_grasp",
    "middle": "mf_grasp",
    "ring": "rf_grasp"
}

# # Example: Planning for the Ring Finger
# finger = "ring"
# xml_finger = f"{finger}_obs" 
# planner = AllegroDynamicAStar(
#     xml_path=xml_paths[xml_finger],
#     finger_type=finger,
#     site_name=site_names[finger],
#     max_step_dist=max_step_dict[finger],
#     constraints=[{"finger": finger, "timestep": 10, "joints": (10, 10, 10, 10)},
#                  {"finger": finger, "timestep": 53, "joints": (16, 53, 53, 53)},
#                  {"finger": finger, "timestep": 58, "joints": (16, 58, 58, 49)}]
# )
