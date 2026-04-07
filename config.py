import numpy as np

# ==================== PATHS ====================
MODEL_PATH = "models/scene.xml"

# ==================== JOINT INDICES ====================
ARM_QPOS_IDX = np.array([0, 1, 2, 3, 4, 5, 6])
N_ARM_JOINTS = 7
FINGER_QPOS_IDX = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
N_FINGER_JOINTS = 16

# ==================== TRAVEL POSE ====================
# Order: ff(4), mf(4), rf(4), th(4)
FINGER_TRAVEL_POSE = np.array([
    0.0, 0.3, 0.3, 0.3,
    0.0, 0.3, 0.3, 0.3,
    0.0, 0.3, 0.3, 0.3,
    0.3, 0.4, 1.3, 0.3,
])

# ==================== JOINT LIMITS ====================
ARM_JOINT_MIN = np.array([-6.28318, -2.2497, -6.28318, -2.5796, -6.28318, -2.0996, -6.28318])
ARM_JOINT_MAX = np.array([ 6.28318,  2.2497,  6.28318,  2.5796,  6.28318,  2.0996,  6.28318])

# ==================== RRT* ====================
ETA = 0.15
GAMMA_RRT = 1.0
GOAL_BIAS = 0.08
GOAL_THRESHOLD = 0.10
MAX_ITER = 20000
COLLISION_STEP = 0.03

# ==================== SMOOTHING ====================
SMOOTHING_ATTEMPTS = 300

# ==================== TIME PARAMETERIZATION ====================
MAX_JOINT_VEL = 1.0
DT = 0.05

# ==================== ARM START / GOAL ====================
ARM_START = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ARM_GOAL = np.array([0.86, 1.53, -0.51, 0.72, 0, 1.3, 0])

# ==================== FINGER PLANNING ====================
ACTIVE_FINGERS = ["first", "middle", "ring", "thumb"]

FINGER_STARTS_DEG = {
    "first":  [int(round(np.rad2deg(x))) for x in FINGER_TRAVEL_POSE[0:4]],
    "middle": [int(round(np.rad2deg(x))) for x in FINGER_TRAVEL_POSE[4:8]],
    "ring":   [int(round(np.rad2deg(x))) for x in FINGER_TRAVEL_POSE[8:12]],
    "thumb":  [int(round(np.rad2deg(x))) for x in FINGER_TRAVEL_POSE[12:16]],
}

FINGER_GOALS_RAD = {
    "first":  [0, 0.27, -0.09, 0.14],
    "middle": [0, 0.61, 1.2, 0.44],
    "ring":   [0, 0.67, 0.98, 0.7],
    "thumb":  [1, 0, 0.94, 0],
}

FINGER_TOLERANCE = 0.005
FINGER_MAX_ITERS = 5_000_000
FINGER_HEURISTIC_WEIGHT = 5