# ucla_gazebo_RL

Repository for the 239AS project.

Note: make sure to install controllers with sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers

# Current instructions (5/1/21)

Run "roslaunch robot_agent_gazebo kuka_ball_balancing_scene.launch". This will launch Gazebo and spawn the robot and ball. The sim will be paused, so you will have to press the play button at the bottom of the GUI to activate the physics.

Then, to play with the robot controller, you can run (in another terminal session [terminator makes it easy to do this]) "rosrun robot_agent_gazebo kuka_teleop.py". This will open a keyboard-interactive python script that with allow you to command joint positions to the robot arm. The script will prompt you (via console output) with the instructions for using it. Make sure this script (located in the scripts folder) is executable, or else rosrun won't work.

# TO DO
1. Play with physics (friction of plate, mass/inertia of ball, etc.) until we are satisfied with how the simulation works
2. Setup state feedback of the ball via the appropriate ros topic or service
3. Configure scene to be "reset-able" for quick episode generation
4. Implement algorithm or technique to determine where ball is on the plate (i.e., distance from plate center) for use in reward function
