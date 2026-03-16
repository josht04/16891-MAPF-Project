import mujoco
import mujoco.viewer
import numpy as np

# Load your specific finger XML
xml_path = r'.\wonik_allegro\left_hand_ring_obstacle.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 1. Identify the mocap ID for the palm
mocap_id = model.body_mocapid[model.body('palm').id]

# 2. Set Position to [0, 0, 0]
data.mocap_pos[mocap_id] = [0.0, 0.0, 0.0]

# 3. Set Orientation to Identity Quaternion [w, x, y, z]
# [1, 0, 0, 0] represents 0 degrees of rotation on all axes
data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

# 4. Finalize the state and launch
mujoco.mj_forward(model, data)

print("Wrist fixed at [0,0,0] with zero rotation.")
mujoco.viewer.launch(model, data)