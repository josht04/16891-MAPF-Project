python -m mujoco.viewer -mjcf models/scene.xml
python run.py 
python playback.py --trajectory output/combined_trajectory.npy

Still need to test with an example actually using CBS.
Need to fix issue where valid goal joint angles for arm are considered collisions, due to RRT planner including curled hand position for planning
