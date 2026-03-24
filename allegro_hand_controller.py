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

        self.joint_names_read = [
            "ffj0", "ffj1", "ffj2", "ffj3",
            "mfj0", "mfj1", "mfj2", "mfj3",
            "rfj0", "rfj1", "rfj2", "rfj3",
            "thj0", "thj1", "thj2", "thj3"
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
        sites = ["ff_grasp", "mf_grasp", "rf_grasp", "th_grasp"]
        # sites = ["rf_grasp"]
        coords = {}
        for s in sites:
            sid = self.model.site(s).id
            pos = self.data.site_xpos[sid]
            coords[s] = pos.copy()
            print(f"{s}: {pos}", end=' \n ')

        # print(f"Fingertips: {coords}", end='\r')

    def print_joint_angles(self, degrees=True):
        """Prints the current angles of all defined joints."""
        angles = {}
        for name in self.joint_names_read:
            # Find the joint ID associated with the actuator/joint name
            # Note: This assumes joint name matches actuator name or you can use mj_name2id
            joint_id = self.model.joint(name).qposadr[0]
            val = self.data.qpos[joint_id]
            
            if degrees:
                angles[name] = np.rad2deg(val)
            else:
                angles[name] = val
        
        # Format printing for readability
        print(" | ".join([f"{k}: {v:6.1f}°" for k, v in angles.items()]), end='\r')

# --- MAIN EXECUTION ---
model = mujoco.MjModel.from_xml_path(r'.\wonik_allegro\doorknob\left_hand_doorknob.xml')
data = mujoco.MjData(model)
controller = AllegroHandController(model, data)

collided = False
model.opt.gravity[:] = [0, 0, 0]

# path_file = f"npy_paths/ring_path_wrist_move_obstacle.npy"
# if os.path.exists(path_file):
#     planned_path = np.load(path_file, allow_pickle=True)
#     print(f"Loaded path with {len(planned_path)} steps.")
# else:
#     planned_path = None
#     print("No path file found. Moving to manual control.")

# wrist_path_file = f"npy_paths/wrist_path_wrist_move_obstacle.npy"
# if os.path.exists(wrist_path_file):
#     wrist_data = np.load(wrist_path_file, allow_pickle=True)
#     print(f"Loaded wrist path with {len(wrist_data)} steps.")
# else:
#     wrist_data = None
#     print("No wrist path file found. Moving to manual control.")

path_file = f"npy_paths/ring_only_trajectory.npy"
if os.path.exists(path_file):
    planned_path = np.load(path_file, allow_pickle=True).item()
    print(planned_path["ring"])


    print(f"Loaded path with {len(planned_path['ring'])} steps.")
else:
    planned_path = None
    print("No path file found. Moving to manual control.")

current_step = 0
last_step_time = 0
step_interval = 0.1  # 0.1 seconds per step

target_xyz = np.array([0.212, 0.00504814, 0.1304538])

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Set rendering flags for sites (Index 12)
    viewer.opt.flags[12] = True
    viewer.opt.sitegroup[:] = 1
    
    def reset_simulation():
        global current_step, last_step_time
        current_step = 0
        last_step_time = time.time()
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        print("\n--- Path Reset ---")

    # Keyboard callback
    def key_callback(key):
        if key == ord('R') or key == ord('r'):
            reset_simulation()
    
    # 1. Set wrist position and rotation (e.g., tilted 45 degrees)
    while viewer.is_running():
        try:
            
            # 2. Set individual angles (Example: curl first finger)
            # Order: ffa0, ffa1, ffa2, ffa3 ...
            # controller.set_joint_angle("ffa0", 10.0) 
            # controller.set_joint_angle("ffa1", 90.0) 
            # controller.set_joint_angle("ffa2", 95.0)
            # controller.set_joint_angle("ffa3", 0.0)
            # controller.set_joint_angle("mfa0", 0.0) 
            # controller.set_joint_angle("mfa1", 90.0) 
            # controller.set_joint_angle("mfa2", 95.0)
            # controller.set_joint_angle("mfa3", 0.0)
            # controller.set_joint_angle("rfa0", 0.0) 
            # controller.set_joint_angle("rfa1", 90.0)
            # controller.set_joint_angle("rfa2", 95.0) 
            # controller.set_joint_angle("rfa3", 0.0)
            # controller.set_joint_angle("tha0", 20.0) 
            # controller.set_joint_angle("tha1", 64.0)
            # controller.set_joint_angle("tha2", 75.0) 
            # controller.set_joint_angle("tha3", 35.0)

            controller.set_joint_angle("ffa0", 8.0) 
            controller.set_joint_angle("ffa1", 13.0) 
            controller.set_joint_angle("ffa2", 60.0)
            controller.set_joint_angle("ffa3", 70.0)
            controller.set_joint_angle("mfa0", 0.0) 
            controller.set_joint_angle("mfa1", 12.0) 
            controller.set_joint_angle("mfa2", 63.0)
            controller.set_joint_angle("mfa3", 70.0)
            controller.set_joint_angle("rfa0", -8.0) 
            controller.set_joint_angle("rfa1", 12.0)
            controller.set_joint_angle("rfa2", 60.0) 
            controller.set_joint_angle("rfa3", 70.0)
            controller.set_joint_angle("tha0", 18.0) 
            controller.set_joint_angle("tha1", 64.0)
            controller.set_joint_angle("tha2", 40.0) 
            controller.set_joint_angle("tha3", 40.0)
            

            current_time = time.time()
            controller.set_wrist_pose(pos=[0.345, -0.05, 0.24], euler_deg=[0, 90, 0])
            controller.print_fingertip_coords()
            # controller.print_joint_angles()
        
            # 2. Play Path Logic
            # if planned_path is not None and current_step < len(planned_path['ring']):
            #     wrist_data = planned_path['wrist']
            #     finger_data = planned_path['ring']
            #     current_time = time.time()
                
            #     # 1. Update WRIST position based on current step
            #     # wrist_data[step] contains ([x,y,z], [r,p,y])
            #     actual_step = min(current_step, len(wrist_data)-1)
            #     w_pos, w_euler = wrist_data[actual_step]
            #     controller.set_wrist_pose(pos=w_pos, euler_deg=w_euler)

            #     # 2. Update FINGERS based on timing interval
            #     if current_time - last_step_time >= step_interval:
            #         target_angles = finger_data[current_step][1]
                    
            #         for i in range(8, 12):
            #             joint_name = controller.joint_names[i]
            #             controller.set_joint_angle(joint_name, target_angles[i-8])
                    
            #         current_step += 1
            #         last_step_time = current_time
            #         print(f"Playing step {current_step}/{len(finger_data)}", end='\r')
            # else:
            #     # OPTIONAL: Uncomment the line below for Automatic Looping
            #     reset_simulation() 

            
        except ValueError as e:
            print(e)
        
        mujoco.mj_step(model, data)
        
        # 3. Features
        # if collided == False:
        #     if controller.check_collisions():
        #         collided = True

        # controller.check_collisions()
        # controller.print_fingertip_coords()
        
        viewer.sync()


    