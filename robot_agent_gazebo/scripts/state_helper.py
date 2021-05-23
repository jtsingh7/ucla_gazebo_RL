#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import sys, select, os
import csv
import math
import rosbag
import rospy
import roslaunch
from datetime import datetime
import time
import copy
import numpy as np

def link_state_CB(data):
	


if __name__ == '__main__':

	rospy.init_node('state_helper')
    #pub = rospy.Publisher('/gazebo/link_states', JointTrajectory, queue_size=10)
    sub = rospy.Subscriber("/gazebo/link_states", JointState, link_state_CB)