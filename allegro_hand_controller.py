import random

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import time

class AllegroHandController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 1. JOINT ORDER DEFINITION
        # Based on your XML actuators: FF (0-3), MF (4-7), RF (8-11), TH (12-15)
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
        """Sets wrist [x, y, z] and orientation for a MOCAP body."""
        # Get the internal mocap ID for the body named 'palm'
        mocap_id = self.model.body_mocapid[self.model.body('palm').id]

        if pos is not None:
            # Directly set the mocap position (ignores physics/gravity)
            self.data.mocap_pos[mocap_id] = pos
            
        if euler_deg is not None:
            # Convert Euler Degrees to Quaternion
            quat = R.from_euler('xyz', euler_deg, degrees=True).as_quat()
            # SciPy uses [x, y, z, w], MuJoCo Mocap uses [w, x, y, z]
            self.data.mocap_quat[mocap_id] = [quat[3], quat[0], quat[1], quat[2]]

    def set_joint_angle(self, name, angle_deg):
        """Sets a single joint angle after verifying safety buffer."""
        if name not in self.limits:
            raise ValueError(f"Joint {name} not found.")
            
        angle_rad = np.deg2rad(angle_deg)
        lim = self.limits[name]
        
        if not (lim["min"] <= angle_rad <= lim["max"]):
            raise ValueError(f"Limit Violation: {name} requested {angle_deg}°. "
                             f"Safe range: [{np.rad2deg(lim['min']):.1f}, {np.rad2deg(lim['max']):.1f}]")
        
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.data.ctrl[actuator_id] = angle_rad

    def check_collisions(self):
        if self.data.ncon > 0:
            print(f"\n--- COLLISION DETECTED ({self.data.ncon} contacts) ---")
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # Get names or fallback to Body names
                g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if g1_name is None:
                    g1_name = f"Body:{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom1])}"
                if g2_name is None:
                    g2_name = f"Body:{mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom2])}"
                    
                print(f"  [{i}] {g1_name} <--> {g2_name}")
            
            # DYNAMIC CALCULATION:
            # nu is the number of actuators (16). 
            # We take the LAST 16 elements of qpos, which are always the finger joints.
            num_joints = self.model.nu 
            current_angles = [np.rad2deg(self.data.qpos[i]) for i in range(num_joints)]
            print(f"Joint Angles (deg): {np.round(current_angles, 1)}")
            return True
        return False
        
        
    def print_fingertip_coords(self):
        """Prints global 3D coordinates of fingertip sites."""
        # sites = ["ff_grasp", "mf_grasp", "rf_grasp", "th_grasp"]
        sites = ["rf_grasp"]
        coords = {}
        for s in sites:
            sid = self.model.site(s).id
            pos = self.data.site_xpos[sid]
            coords[s] = pos.copy()
        print(f"Fingertips: {coords}", end='\r')

# --- MAIN EXECUTION ---
model = mujoco.MjModel.from_xml_path(r'.\wonik_allegro\left_hand_ring.xml')
data = mujoco.MjData(model)
controller = AllegroHandController(model, data)

collided = False
model.opt.gravity[:] = [0, 0, 0]

path_file = f"ring_path.npy"
if os.path.exists(path_file):
    planned_path = np.load(path_file, allow_pickle=True)
    print(f"Loaded path with {len(planned_path)} steps.")
else:
    planned_path = None
    print("No path file found. Moving to manual control.")

current_step = 0
last_step_time = 0
step_interval = 0.1  # 0.1 seconds per step



with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set rendering flags for sites (Index 12)
    viewer.opt.flags[12] = True
    viewer.opt.sitegroup[:] = 1
    
    # 1. Set wrist position and rotation (e.g., tilted 45 degrees)
    
    
    while viewer.is_running():
        try:
            
            # 2. Set individual angles (Example: curl first finger)
            # Order: ffa0, ffa1, ffa2, ffa3 ...
            # controller.set_joint_angle("rfa0", 16.0) 
            # controller.set_joint_angle("rfa1", 58.0) 
            # controller.set_joint_angle("rfa2", 58.0)
            # controller.set_joint_angle("rfa3", 49.0)
            # controller.set_joint_angle("mfa0", -20.0) 
            # controller.set_joint_angle("mfa1", 90.0) 
            # controller.set_joint_angle("mfa2", 90.0)
            # controller.set_joint_angle("mfa3", 0.0)

            current_time = time.time()
            controller.set_wrist_pose(pos=[0, 0, 0], euler_deg=[0, 0, 0])
        
            # 2. Play Path Logic
            if planned_path is not None and current_step < len(planned_path):
                # Check if it's time for the next step
                if current_time - last_step_time >= step_interval:
                    target_angles = planned_path[current_step][1]
                    
                    for i in range(8,12):
                        joint_name = controller.joint_names[i]
                        controller.set_joint_angle(joint_name, target_angles[i-8])
                    
                    current_step += 1
                    last_step_time = current_time
                    print(f"Playing step {current_step}/{len(planned_path)}", end='\r')
            pass
            
        except ValueError as e:
            print(e)

        mujoco.mj_step(model, data)
        
        # 3. Features
        # if collided == False:
        #     if controller.check_collisions():
        #         collided = True

        controller.check_collisions()
        # controller.print_fingertip_coords()
        
        viewer.sync()


    