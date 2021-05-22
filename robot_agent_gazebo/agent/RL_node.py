#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import sys, select, os
import argparse
import csv
import math
import rosbag
import rospy
import roslaunch
from datetime import datetime
import time
import copy
import numpy as np
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)



    
     


if __name__ == '__main__':
    try:
        rospy.init_node('RL_agent')
        pub = rospy.Publisher('/iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=10)
        sub = rospy.Subscriber("/iiwa/joint_states", JointState, subCB)
        

    except rospy.ROSInterruptException:
        pass

