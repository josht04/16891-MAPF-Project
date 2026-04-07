# Usage

1. python -m mujoco.viewer --mjcf models/scene.xml to set and record qpos

2. python run.py

3. python playback.py --trajectory output/combined_trajectory.npy


# Stuff to fix

- Still need to test with an example that actually triggers CBS.
- Valid goal joint angles for the arm can be rejected as collisions, because the RRT* planner checks with the finger travel pose rather than the actual grasp pose at the goal.
