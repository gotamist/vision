# Landmark Detection & Robot Tracking (SLAM)

## Project Overview

In this project, I implement SLAM (Simultaneous Localization and
Mapping) for a 2 dimensional world! Robot sensor measurements and
movement information, both noisy, are combined to create a map of an
environment and the location of the robot in it.  The information is
updated over time as more data is obtained by the robot.


SLAM gives us a way to track the location of a robot in the
world in real-time and identify the locations of landmarks such as
buildings, trees, rocks, and other world features. This is an active
area of research in the fields of robotics and autonomous systems.

*Below is an example of a 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using *only* sensor and motion data collected by that robot. This is one example for a 50x50 grid world.  The notebook shows a few of these maps as we keep track of robot poses and the landmarks.

<p align="center">
  <img src="./images/robot_world.png" width=50% height=50% />
</p>

The starting point is the iPython notebook.  The robot class is implemented in the `robot_class.py` file.

LICENSE: This project is licensed under the terms of the MIT license.
