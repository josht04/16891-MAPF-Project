import mujoco
import numpy as np
import heapq
from scipy.spatial.transform import Rotation as R

class AllegroDynamicAStar:
    def __init__(self, xml_path, finger_type, site_name, max_step_dist):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site(site_name).id
        self.mocap_id = self.model.body_mocapid[self.model.body('palm').id]
        self.max_step_dist = max_step_dist
        
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
        return self.data.ncon == 0

    def plan(self, start_q, goal_xyz, wrist_path, tolerance=0.005, max_iters=50000):
        # start_q: (q0, q1, q2, q3)
        # wrist_path: List of ((x,y,z), (r,p,y)) per step
        
        start_node = (0, tuple(start_q)) # (time_step, angles)
        self.set_context(0, wrist_path, start_q)
        
        pq = [(np.linalg.norm(self.data.site_xpos[self.site_id] - goal_xyz)/self.max_step_dist, 0, start_node)]
        came_from = {start_node: None}
        g_score = {start_node: 0}
        
        moves = [tuple(m) for m in np.array(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1],[-1,0,1])).T.reshape(-1,4) if not np.all(m==0)]

        iters = 0
        while pq and iters < max_iters:
            iters += 1
            f, g, (t, curr_q) = heapq.heappop(pq)
            
            # Check if reached goal at current wrist position
            self.set_context(t, wrist_path, curr_q)
            curr_p = self.data.site_xpos[self.site_id].copy()
            
            if np.linalg.norm(curr_p - goal_xyz) < tolerance:
                print(f"Path Found! Steps: {t}")
                path = []
                curr = (t, curr_q)
                while curr:
                    path.append(curr)
                    curr = came_from[curr]
                return path[::-1]

            # Explore next time step (t + 1)
            next_t = t + 1
            for move in moves:
                next_q = tuple(curr_q[i] + move[i] for i in range(4))
                
                # Fast limit check
                if any(next_q[i] < self.limits[i][0] or next_q[i] > self.limits[i][1] for i in range(4)):
                    continue
                
                next_node = (next_t, next_q)
                if next_node not in g_score:
                    # Update physics for the NEW time and NEW angles
                    self.set_context(next_t, wrist_path, next_q)
                    
                    if self.is_valid():
                        g_score[next_node] = next_t
                        h = np.linalg.norm(self.data.site_xpos[self.site_id] - goal_xyz) / self.max_step_dist
                        heapq.heappush(pq, (next_t + h, next_t, next_node))
                        came_from[next_node] = (t, curr_q)
        return None    

# --- Main Configuration ---
max_step_dict = {
    "thumb": 0.0013,
    "first": 0.00165,
    "middle": 0.00165,
    "ring": 0.00165,
}

xml_paths = {
    "thumb": r'.\wonik_allegro\left_hand_thumb.xml',
    "first": r'.\wonik_allegro\left_hand_first.xml',
    "middle": r'.\wonik_allegro\left_hand_middle.xml',
    "ring": r'.\wonik_allegro\left_hand_ring.xml'
}

site_names = {
    "thumb": "th_grasp",
    "first": "ff_grasp",
    "middle": "mf_grasp",
    "ring": "rf_grasp"
}

# Example: Planning for the Ring Finger
finger = "ring"
planner = AllegroDynamicAStar(
    xml_path=xml_paths[finger],
    finger_type=finger,
    site_name=site_names[finger],
    max_step_dist=max_step_dict[finger]
)

start_q = [0, 0, 0, 0]
target_xyz = np.array([ 0.0722836 ,  0.06396464, -0.0054457 ])
wrist_path = [([0, 0, 0], [0, 0, 0])]

path = planner.plan(start_q, target_xyz, wrist_path)

print(path)

if path:
    print(f"Path for {finger} found in {len(path)} time steps.")
    np.save(f"{finger}_path.npy", path)