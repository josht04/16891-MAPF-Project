import mujoco
import mujoco.viewer
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R

class AllegroHandController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 1. JOINT ORDER DEFINITION
        # Based on your XML: FF (0-3), MF (4-7), RF (8-11), TH (12-15)
        self.joint_names = [
            "ffa0", "ffa1", "ffa2", "ffa3",
            "mfa0", "mfa1", "mfa2", "mfa3",
            "rfa0", "rfa1", "rfa2", "rfa3",
            "tha0", "tha1", "tha2", "tha3"
        ]
        
        # 2. LOOKUP TABLE WITH 2° SAFETY BUFFER
        self.limits = {}
        buffer_rad = np.deg2rad(2.0)
        
        for name in self.joint_names:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            ctrl_range = model.actuator_ctrlrange[actuator_id]
            self.limits[name] = {
                "min": ctrl_range[0] + buffer_rad,
                "max": ctrl_range[1] - buffer_rad
            }

    def set_wrist_pose(self, pos=None, euler_deg=None):
        mocap_id = self.model.body_mocapid[self.model.body('palm').id]
        if pos is not None:
            self.data.mocap_pos[mocap_id] = pos
        if euler_deg is not None:
            quat = R.from_euler('xyz', euler_deg, degrees=True).as_quat()
            self.data.mocap_quat[mocap_id] = [quat[3], quat[0], quat[1], quat[2]]

    def set_joint_angle(self, name, angle_deg):
        if name not in self.limits: return
        angle_rad = np.deg2rad(angle_deg)
        lim = self.limits[name]
        # Clamp to limits instead of crashing with ValueError for smoother playback
        clamped_rad = np.clip(angle_rad, lim["min"], lim["max"])
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.data.ctrl[actuator_id] = clamped_rad

    def check_collisions(self):
        if self.data.ncon > 0:
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            
            collision_found = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # 1. Skip floor
                if contact.geom1 == floor_id or contact.geom2 == floor_id:
                    continue
                
                # 2. Distance check: Ignore the margin buffer
                # dist <= 0 means they are physically touching or penetrating
                if contact.dist <= 0:
                    collision_found = True
                    
                    g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or \
                        f"Body:{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom1])}"
                    g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or \
                        f"Body:{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom2])}"
                    
                    print(f"ACTUAL PENETRATION (dist={contact.dist:.4f}): {g1} <--> {g2}")
            
            return collision_found
        return False

# --- PLAYBACK EXECUTION ---

# 1. Initialization
model = mujoco.MjModel.from_xml_path(r'./wonik_allegro/obstacle_ex1/left_hand_obstacle.xml')
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
data = mujoco.MjData(model)
controller = AllegroHandController(model, data)
model.opt.gravity[:] = [0, 0, 0]

# 2. Load Trajectory (Supports full hand or single finger files)
path_file = "npy_paths/full_hand_trajectory_obstacle_ex1.npy"
if not os.path.exists(path_file):
    raise FileNotFoundError(f"Could not find {path_file}")

traj = np.load(path_file, allow_pickle=True).item()
wrist_data = traj.get("wrist")

# Map npy keys to internal joint ranges
finger_map = {
    "first": slice(0, 4),
    "middle": slice(4, 8),
    "ring": slice(8, 12),
    "thumb": slice(12, 16)
}

current_step = 0
step_interval = 0.05  # Faster playback
last_step_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    def reset_simulation():
        global current_step, last_step_time
        current_step = 0
        last_step_time = time.time()
        mujoco.mj_resetData(model, data)
        print("\n--- Restarting Playback ---")

    while viewer.is_running():
        current_time = time.time()
        
        # Determine total steps based on available finger paths
        active_paths = {k: traj[k] for k in finger_map.keys() if k in traj and traj[k] is not None}
        if not active_paths: break
        max_steps = max(len(p) for p in active_paths.values())

        if current_step < max_steps:
            # A. Update Wrist
            if wrist_data is not None:
                actual_w_step = min(current_step, len(wrist_data) - 1)
                w_pos, w_euler = wrist_data[actual_w_step]
                controller.set_wrist_pose(pos=w_pos, euler_deg=w_euler)

            # B. Update All Active Fingers
            if current_time - last_step_time >= step_interval:
                for finger_name, slice_idx in finger_map.items():
                    finger_path = traj.get(finger_name)
                    if finger_path is not None and current_step < len(finger_path):
                        _, q_vals = finger_path[current_step]
                        
                        # Set each of the 4 joints for this finger
                        for i in range(4):
                            global_idx = slice_idx.start + i
                            joint_name = controller.joint_names[global_idx]
                            controller.set_joint_angle(joint_name, q_vals[i])

                current_step += 1
                last_step_time = current_time
                print(f"Step {current_step}/{max_steps}", end='\r')
        else:
            time.sleep(1.0) # Pause at the end
            reset_simulation()

        mujoco.mj_step(model, data)
        controller.check_collisions()
        viewer.sync()