import numpy as np
from allegro_astar import AllegroDynamicAStar, generate_wrist_path, XML_PATHS, SITE_NAMES, MAX_STEP_DICT

def plan_full_hand(wrist_start_pos, wrist_end_pos, wrist_start_euler, wrist_end_euler, timer, finger_starts, finger_goals, xml_subscript="obs"):
    
    # 2. Generate Wrist Path (Used by all fingers)
    wrist_path = generate_wrist_path(wrist_start_pos, wrist_end_pos, wrist_start_euler, wrist_end_euler, timer)

    # 3. Define Finger Start Angles and XYZ Goals
    # Order: Thumb, First, Middle, Ring
    
    # e.g constraint = {"finger":"thumb","timestep":5, "joints":(10,10,10,10)}
    constraints = []

    finger_names = ["thumb", "first", "middle", "ring"]
    all_paths = {"wrist": wrist_path}

    # 4. Run Loop
    for i, name in enumerate(finger_names):
        planner = AllegroDynamicAStar(
            xml_path=XML_PATHS[f"{name}_{xml_subscript}"],
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
    np.save(f"npy_paths/{xml_subscript}_full_hand_trajectory.npy", all_paths)
    print(f"Saved all paths to {xml_subscript}_full_hand_trajectory.npy")

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
    
    # # FOR OBSTACLE EXAMPLE 1
    # wrist_start_pos = [-0.4, -0.05, 0.0]
    # wrist_end_pos = [0.2, -0.05, 0.0]
    # wrist_start_euler = [0, 0, 0]  # No rotation
    # wrist_end_euler = [0, 0, 0]    # No rotation    
    # timer = 20.0  # Duration of the wrist movement in seconds
    # finger_starts = [[18,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    # finger_goals = [
    #     np.array([0.237, -0.2062, -0.085]), # Thumb Goal
    #     np.array([0.212, -0.105, 0.13]), # First Goal
    #     np.array([0.212, -0.05, 0.133]), # Middle Goal
    #     np.array([0.212, 0.005, 0.13])  # Ring Goal
    # ]
    # xml_subscript="obs"
    # plan_full_hand(wrist_start_pos, wrist_end_pos, wrist_start_euler, wrist_end_euler, timer, finger_starts, finger_goals, xml_subscript)

# ----------------------------------------------------------------------------------------------------

    # # FOR CLOSED GRAB
    # finger_starts = [[20,64,75,35], [10,90,95,0],[0,90,95,0],[0,90,95,0]]
    # finger_goals = [
    #     np.array([0.35409264, -0.08491088, -0.02898133]), # Thumb Goal
    #     np.array([0.46286335, -0.05693109, -0.02389508]), # First Goal
    #     np.array([0.475353206,  0, -0.0191418818]), # Middle Goal
    #     np.array([0.46501551,  0.05308591, -0.02152683])  # Ring Goal
    # ]
    # # finger_goal_angles = [[18,64,40,40],[-6,18,52,29], [0,12,51,27],[3,15,54,29]]
    # wrist_start_pos = [0, 0, 0.05]
    # wrist_end_pos = [0.4, 0, 0.05]
    # wrist_start_euler = [0, 90, 0]  
    # wrist_end_euler = [0, 90, 0] 
    # timer = 20
    # xml_subscript="closed_grab"
    # plan_full_hand(wrist_start_pos, wrist_end_pos, wrist_start_euler, wrist_end_euler, timer, finger_starts, finger_goals, xml_subscript)

# ----------------------------------------------------------------------------------------------------

    # FOR DOORKNOB

    finger_starts = [[18,64,40,40],[-6,18,52,29], [0,12,51,27],[3,15,54,29]]
    finger_goals = [
        np.array([0.29909264, -0.13491088,  0.16101867]), # Thumb Goal
        np.array([0.39799531, -0.09079037,  0.18697228]), # First Goal
        np.array([0.3985618,  -0.05,        0.18791728]), # Middle Goal
        np.array([0.39891251, -0.00903634,  0.1876318])  # Ring Goal
    ]
    wrist_start_pos = [-0.1, 0, 0.3]
    wrist_end_pos = [0.345, -0.05, 0.24]
    wrist_start_euler = [0, 90, 0]  
    wrist_end_euler = [0, 90, 0] 
    timer = 40
    xml_subscript="doorknob"
    plan_full_hand(wrist_start_pos, wrist_end_pos, wrist_start_euler, wrist_end_euler, timer, finger_starts, finger_goals, xml_subscript)

    

    