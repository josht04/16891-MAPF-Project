import mujoco
import mujoco.viewer
import numpy as np

# 1. Load your model
model = mujoco.MjModel.from_xml_path(r'.\wonik_allegro\scene_left.xml')
data = mujoco.MjData(model)

# --- ONE-TIME WRIST POSITIONING ---
# A freejoint has 7 variables: [x, y, z, quat_w, quat_x, quat_y, quat_z]
# We set the wrist 0.5 meters above the ground
data.qpos[0:3] = [0, 0, 0.5] 

# Apply the changes to the physics state immediately
mujoco.mj_forward(model, data)

# 2. Start the Simulation
with mujoco.viewer.launch_passive(model, data) as viewer:

    # print([attr for attr in dir(mujoco.mjtVisFlag) if not attr.startswith('__')])
    viewer.opt.flags[12] = True
    viewer.opt.flags[2] = True
    viewer.opt.sitegroup[:] = 1
    
    # Simple counter to create a waving/moving motion
    step_count = 0
    
    while viewer.is_running():
        # --- EVERY STEP: CHANGE FINGER JOINTS ---
        # We target the 'ctrl' array. 
        # Allegro hands usually have 16 actuators.
        
        # Example: Sinusoidal movement for all fingers
        target_angle = 0
        # 0.5 * np.sin(step_count * 0.01)
        
        # We skip the first 7 qpos values (the wrist) and target actuators
        # model.nu is the number of actuators (controls)
        for i in range(model.nu):
            data.ctrl[i] = target_angle
            
        # --- COLLISION CHECK ---
        for i in range(data.ncon):
            contact = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if "ball" in [g1, g2]:
                print(f"Ball touched by: {g1 if g1 != 'ball' else g2}")

        # Physics Step
        mujoco.mj_step(model, data)
        
        # Sync the visualizer
        viewer.sync()
        step_count += 1