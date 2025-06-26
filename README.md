# Motion Planning for Kinova Gen3 in Mujuco
Authors: Connor Mattson

## Installation

First, install mujuco if you haven't yet.
```
pip install mujoco
```

### Test Loading Robot Model + Gripper

Check that the robot model is loading correctly by running the following command.

```
python -m mujoco.viewer --mjcf robot_models/kinova_gen3/scene.xml
```

You should see a mujoco model that looks like this:
![image](docs/media/viewer.png)

If you get an error while running this, it is likely that your mujoco is not correctly installed, check [https://github.com/google-deepmind/mujoco_menagerie?tab=readme-ov-file#prerequisites] for more.

### Test Simulation, Gravity Compensation

To set the robot to a "home" position and test forward simulation with velocity control, run the following command

```
mjpython -m src.examples.init_home
```

you should see the robot controlling the joints to hold perfectly still (gravity compensation).

### Test Velocity Control
As a robotics researcher (Connor), I'd really like to see my robot move please. Let's make sure we can make it move with the following test.

```
mjpython -m src.examples.test_vel_ctrl
```

## Motion Planning

Test naive RRT to go to a couple of positions
```
mjpython -m src.examples.working_motion_demo
```
