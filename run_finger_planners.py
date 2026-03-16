import numpy as np
from allegro_astar import AllegroDynamicAStar, generate_wrist_path, XML_PATHS, SITE_NAMES, MAX_STEP_DICT

def plan_full_hand():
    # 1. Define Wrist Start/End
    wrist_start_pos = [-0.4, -0.05, 0.0]
    wrist_end_pos = [0.2, -0.05, 0.0]
    wrist_start_euler = [0, 0, 0]  # No rotation
    wrist_end_euler = [0, 0, 0]    # No rotation    
    timer = 20.0  # Duration of the wrist movement in seconds
    
    # 2. Generate Wrist Path (Used by all fingers)
    wrist_path = generate_wrist_path(wrist_start_pos, wrist_end_pos, wrist_start_euler, wrist_end_euler, timer)

    # 3. Define Finger Start Angles and XYZ Goals
    # Order: Thumb, First, Middle, Ring
    finger_starts = [[18,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    finger_goals = [
        np.array([0.237, -0.2062, -0.085]), # Thumb Goal
        np.array([0.212, -0.105, 0.13]), # First Goal
        np.array([0.212, -0.05, 0.133]), # Middle Goal
        np.array([0.212, 0.005, 0.13])  # Ring Goal
    ]
    
    # e.g constraint = {"finger":"thumb","timestep":5, "joints":(10,10,10,10)}
    constraints = []

    finger_names = ["thumb", "first", "middle", "ring"]
    all_paths = {"wrist": wrist_path}

    # 4. Run Loop
    for i, name in enumerate(finger_names):
        planner = AllegroDynamicAStar(
            xml_path=XML_PATHS[f"{name}_obs"],
            finger_type=name,
            site_name=SITE_NAMES[name],
            max_step_dist=MAX_STEP_DICT[name],
            constraints=constraints
        )
        
        print(f"Planning {name}...")
        path = planner.plan(finger_starts[i], finger_goals[i], wrist_path)
        all_paths[name] = path
        print(f"Planned {len(path)} steps for {name}.")

    # 5. Save everything into ONE file
    np.save("npy_paths/full_hand_trajectory.npy", all_paths)
    print("Saved all paths to full_hand_trajectory.npy")

def plan_single_finger(finger_name, start_q, goal_xyz, wrist_start, wrist_end, duration=20.0):
    """
    Plans a path for a single specified finger.
    """
    # 1. Generate Wrist Path (Required context for Dynamic A*)
    wrist_path = generate_wrist_path(
        wrist_start[0], wrist_end[0], 
        wrist_start[1], wrist_end[1], 
        duration
    )

    planner = AllegroDynamicAStar(
        xml_path=XML_PATHS[f"{finger_name}_obs"],
        finger_type=finger_name,
        site_name=SITE_NAMES[finger_name],
        max_step_dist=MAX_STEP_DICT[finger_name],
        constraints=[] 
    )

    # 3. Plan
    print(f"--- Planning Single Finger: {finger_name} ---")
    path = planner.plan(start_q, goal_xyz, wrist_path)

    if path:
        print(f"Successfully planned {len(path)} steps.")
        
        # 4. Save individual result
        save_data = {
            "wrist": wrist_path,
            finger_name: path
        }
        filename = f"npy_paths/{finger_name}_only_trajectory.npy"
        np.save(filename, save_data)
        print(f"Saved to {filename}")
        return path
    else:
        print(f"Failed to find path for {finger_name}")
        return None

if __name__ == "__main__":
    # w_start = ([-0.4, -0.05, 0.0], [0, 0, 0])
    # w_end = ([0.2, -0.05, 0.0], [0, 0, 0])
    
    # # Plan just the thumb
    # plan_single_finger(
    #     finger_name="ring",
    #     start_q=[18, 0, 0, 0],
    #     goal_xyz=np.array([0.212, 0.005, 0.13]),
    #     wrist_start=w_start,
    #     wrist_end=w_end
    # )

    plan_full_hand()