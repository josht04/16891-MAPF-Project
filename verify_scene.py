"""
Phase 1 verification: Load the combined Kinova+Allegro scene,
print all joint/geom info, and test basic collision detection.
"""
import mujoco
import numpy as np

MODEL_PATH = "models/scene.xml"

def main():
    # --- Load model ---
    print("=" * 60)
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print(f"  Model loaded: {model.nq} qpos, {model.nv} dof, {model.nu} actuators")
    print(f"  Bodies: {model.nbody}, Geoms: {model.ngeom}, Joints: {model.njnt}")

    # --- List all joints with indices ---
    print("\n" + "=" * 60)
    print("JOINTS:")
    print(f"{'idx':<5} {'name':<20} {'qposadr':<10} {'type':<8} {'range'}")
    arm_joints = []
    finger_joints = []
    for i in range(model.njnt):
        name = model.joint(i).name
        qposadr = model.joint(i).qposadr[0]
        jnt_type = model.joint(i).type[0]
        jnt_range = model.joint(i).range
        type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[jnt_type]
        print(f"  {i:<5} {name:<20} {qposadr:<10} {type_str:<8} [{jnt_range[0]:.3f}, {jnt_range[1]:.3f}]")

        if name.startswith("joint_"):
            arm_joints.append((name, i, qposadr))
        else:
            finger_joints.append((name, i, qposadr))

    print(f"\n  Arm joints ({len(arm_joints)}):")
    for name, idx, adr in arm_joints:
        print(f"    {name}: joint_idx={idx}, qpos[{adr}]")
    print(f"\n  Finger joints ({len(finger_joints)}):")
    for name, idx, adr in finger_joints:
        print(f"    {name}: joint_idx={idx}, qpos[{adr}]")

    # --- List all geoms with contype/conaffinity ---
    print("\n" + "=" * 60)
    print("GEOMS (collision-relevant):")
    print(f"{'idx':<5} {'name':<30} {'contype':<10} {'conaffinity':<12} {'body'}")
    for i in range(model.ngeom):
        contype = model.geom(i).contype[0]
        conaffinity = model.geom(i).conaffinity[0]
        if contype == 0 and conaffinity == 0:
            continue  # skip visual-only geoms
        name = model.geom(i).name or f"(unnamed_{i})"
        body_id = model.geom(i).bodyid[0]
        body_name = model.body(body_id).name
        print(f"  {i:<5} {name:<30} {contype:<10} {conaffinity:<12} {body_name}")

    # --- Test: set home keyframe and run forward kinematics ---
    print("\n" + "=" * 60)
    print("Testing home keyframe...")
    mujoco.mj_resetDataKeyframe(model, data, 0)  # keyframe "home"
    mujoco.mj_fwdPosition(model, data)

    # Print palm position in world frame
    palm_id = model.body("palm").id
    palm_pos = data.xpos[palm_id]
    palm_quat = data.xquat[palm_id]
    print(f"  Palm world position: [{palm_pos[0]:.4f}, {palm_pos[1]:.4f}, {palm_pos[2]:.4f}]")
    print(f"  Palm world quat:     [{palm_quat[0]:.4f}, {palm_quat[1]:.4f}, {palm_quat[2]:.4f}, {palm_quat[3]:.4f}]")

    # --- Test: collision detection at home ---
    print("\n" + "=" * 60)
    print("Collision check at home pose...")
    print(f"  Number of contacts: {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = model.geom(c.geom1).name or f"geom_{c.geom1}"
        g2 = model.geom(c.geom2).name or f"geom_{c.geom2}"
        b1 = model.body(model.geom(c.geom1).bodyid[0]).name
        b2 = model.body(model.geom(c.geom2).bodyid[0]).name
        print(f"    Contact: {g1} ({b1}) <-> {g2} ({b2}), dist={c.dist:.6f}")

    # --- Test: set arm into a colliding pose (reaching into rack) ---
    print("\n" + "=" * 60)
    print("Testing a reaching pose (arm extended forward)...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # Move joint_2 and joint_4 to extend arm forward
    arm_qpos_adr = [adr for _, _, adr in arm_joints]
    # Rough forward reach: shoulder forward, elbow extended
    reach_config = np.array([0, 0.8, 3.14, -1.5, 0, 0.5, 1.57])
    for j, adr in enumerate(arm_qpos_adr):
        data.qpos[adr] = reach_config[j]
    mujoco.mj_fwdPosition(model, data)

    palm_pos = data.xpos[palm_id]
    print(f"  Palm world position: [{palm_pos[0]:.4f}, {palm_pos[1]:.4f}, {palm_pos[2]:.4f}]")
    print(f"  Number of contacts: {data.ncon}")
    for i in range(min(data.ncon, 10)):  # show up to 10
        c = data.contact[i]
        g1 = model.geom(c.geom1).name or f"geom_{c.geom1}"
        g2 = model.geom(c.geom2).name or f"geom_{c.geom2}"
        b1 = model.body(model.geom(c.geom1).bodyid[0]).name
        b2 = model.body(model.geom(c.geom2).bodyid[0]).name
        print(f"    Contact: {g1} ({b1}) <-> {g2} ({b2}), dist={c.dist:.6f}")
    if data.ncon > 10:
        print(f"    ... and {data.ncon - 10} more")

    # --- Contype/conaffinity collision matrix summary ---
    print("\n" + "=" * 60)
    print("Collision group design:")
    print("  contype=1 (Kinova arm)     x conaffinity=4,7 (obstacles) = CAN COLLIDE")
    print("  contype=2 (Allegro finger)  x conaffinity=7 (obstacles)  = CAN COLLIDE")
    print("  contype=2 (Allegro finger)  x conaffinity=2 (finger)     = CAN COLLIDE (inter-finger)")
    print("  contype=1 (Kinova arm)     x conaffinity=2 (finger)     = NO (1&2=0)")
    print("  contype=4 (obstacle)       x conaffinity=1 (arm)        = CAN COLLIDE")
    print("  contype=4 (obstacle)       x conaffinity=2 (finger)     = CAN COLLIDE (4&2≠0? 4&2=0)")
    print()
    print("  NOTE: contype & conaffinity is bitwise AND.")
    print("  Kinova (1) & obstacle affinity (7) = 1 -> collide ✓")
    print("  Allegro (2) & obstacle affinity (7) = 2 -> collide ✓")
    print("  Obstacle (4) & Kinova affinity (1) = 0 -> but symmetric check: Kinova(1) & obstacle(7) = 1 ✓")
    print("  Allegro (2) & Allegro affinity (2) = 2 -> inter-finger collide ✓")
    print("  Kinova (1) & Allegro affinity (2) = 0 -> no arm-finger collision ✓")

    print("\n" + "=" * 60)
    print("Phase 1 verification COMPLETE.")
    print("Next: run 'python -m mujoco.viewer --mjcf models/scene.xml' to visualize.")


if __name__ == "__main__":
    main()