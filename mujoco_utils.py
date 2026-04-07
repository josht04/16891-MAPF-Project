"""
MuJoCo interface: load model, set configurations, identify geom ownership.
"""
import mujoco
import numpy as np
import config


class MuJoCoEnv:
    """Wraps the MuJoCo model/data for collision checking and FK."""

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Build geom -> category lookup once
        self._build_geom_categories()

        # Cache body IDs for quick access
        self.palm_body_id = self.model.body("palm").id

    def _build_geom_categories(self):
        """Classify every geom as 'arm', 'finger', 'obstacle', 'floor', or 'other'.

        Uses contype bits:
            1 = arm (Kinova)
            2 = finger (Allegro)
            4 = obstacle
            8 = target object
            0 = visual-only
        """
        self.geom_category = {}
        self.obstacle_geom_ids = set()
        self.arm_geom_ids = set()
        self.finger_geom_ids = set()

        for i in range(self.model.ngeom):
            contype = self.model.geom(i).contype[0]
            name = self.model.geom(i).name or f"geom_{i}"

            if name == "floor":
                self.geom_category[i] = "floor"
            elif contype == 0:
                self.geom_category[i] = "visual"
            elif contype == 1:
                self.geom_category[i] = "arm"
                self.arm_geom_ids.add(i)
            elif contype == 2:
                self.geom_category[i] = "finger"
                self.finger_geom_ids.add(i)
            elif contype == 4:
                self.geom_category[i] = "obstacle"
                self.obstacle_geom_ids.add(i)
            elif contype == 8:
                self.geom_category[i] = "target"
            else:
                self.geom_category[i] = "other"

    def set_arm_config(self, arm_q: np.ndarray):
        """Set arm joints and lock fingers at travel pose. Run fwdPosition."""
        self.data.qpos[config.ARM_QPOS_IDX] = arm_q
        self.data.qpos[config.FINGER_QPOS_IDX] = config.FINGER_TRAVEL_POSE
        mujoco.mj_fwdPosition(self.model, self.data)

    def set_full_config(self, arm_q: np.ndarray, finger_q: np.ndarray):
        """Set both arm and finger joints. Run fwdPosition."""
        self.data.qpos[config.ARM_QPOS_IDX] = arm_q
        self.data.qpos[config.FINGER_QPOS_IDX] = finger_q
        mujoco.mj_fwdPosition(self.model, self.data)

    def get_palm_pose(self) -> tuple:
        """Return (pos, quat) of palm body in world frame."""
        pos = self.data.xpos[self.palm_body_id].copy()
        quat = self.data.xquat[self.palm_body_id].copy()
        return pos, quat

    def is_robot_obstacle_contact(self, geom1_id: int, geom2_id: int) -> bool:
        """Check if a contact pair involves a robot geom and an obstacle geom.

        Robot geoms = arm + finger (for wrist planning, fingers are part of robot).
        """
        cat1 = self.geom_category.get(geom1_id, "other")
        cat2 = self.geom_category.get(geom2_id, "other")

        robot_cats = {"arm", "finger"}
        obstacle_cats = {"obstacle", "floor"}

        if cat1 in robot_cats and cat2 in obstacle_cats:
            return True
        if cat2 in robot_cats and cat1 in obstacle_cats:
            return True
        return False

    def is_arm_obstacle_contact(self, geom1_id: int, geom2_id: int) -> bool:
        """Check if a contact involves specifically an arm geom and obstacle.

        Does NOT flag finger-obstacle contacts (for wrist-only planning,
        fingers are locked so their collisions matter too — use is_robot_obstacle_contact).
        """
        cat1 = self.geom_category.get(geom1_id, "other")
        cat2 = self.geom_category.get(geom2_id, "other")

        obstacle_cats = {"obstacle", "floor"}

        if cat1 == "arm" and cat2 in obstacle_cats:
            return True
        if cat2 == "arm" and cat1 in obstacle_cats:
            return True
        return False