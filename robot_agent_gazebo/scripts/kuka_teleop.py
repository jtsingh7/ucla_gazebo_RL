#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
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
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)
#parser = argparse.ArgumentParser(description='Kuka arm contact testing')
#parser.add_argument('--tele',action='store_true',help='Teleop control of kuka joints')
#args = parser.parse_args()



def getKey():
    if os.name == 'nt':
      return msvcrt.getch()

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def get_jt(jtp_list,t):
    jt = JointTrajectory()
    jt.joint_names = ["iiwa_joint_1","iiwa_joint_2","iiwa_joint_3","iiwa_joint_4","iiwa_joint_5","iiwa_joint_6","iiwa_joint_7"]
    jtp = JointTrajectoryPoint()
    #jtp_list = [0,0,0,0,0,0,0]
    jtp.positions = jtp_list
    jtp.time_from_start = rospy.Duration.from_sec(t)
    jt.points.append(jtp)

    return jt

def tele_joint():

    msg = """
    Interactively set kuka joint positions.
    ---------------------------
    Key controls:
        0 1 2 3 4 5 6
             w
        a    s    d

    digits: select joint to modify
    a/d : increase/decrease angle (radians)
    s: reset all joints to zero
    w: send command

    CTRL-C to quit
    """

    e = """
    Communication failed.
    """  

    rospy.init_node('iiwa_joint_teleop')
    pub = rospy.Publisher('/iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=10)

    status = 0
    i=0
    jtp_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    t=1

    try:
        print(msg)
        while 1:
            key = getKey()

            try:
                key=int(key)
                if key<7:
                    print("Editing joint {}".format(key))
                    i = key
            except:
                pass
            
            if key == 'a' :
                jtp_list[i]+=math.pi/16
                print(jtp_list)
            elif key == 'd' :
                jtp_list[i]-=math.pi/16
                print(jtp_list)
            elif key == 's' :
                jtp_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                print(jtp_list)
            elif key == 'w':
                print("Sending command")
                jt = get_jt(jtp_list,t)
                pub.publish(jt)

            else:
                if (key == '\x03'):
                    break
    except:
        print(e)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)    



def talker():
    pub = rospy.Publisher('/iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    jt = JointTrajectory()
    #jt.joint_names.append("simple_gripper_right_driver_joint")
    jt.joint_names = ["toe_foot_joint","foot_leg_joint","leg_arm_joint","arm_hand_joint","hand_finger_joint"]
    jt.joint_names = ["iiwa_joint_1","iiwa_joint_2","iiwa_joint_3","iiwa_joint_4","iiwa_joint_5","iiwa_joint_6","iiwa_joint_7"]
    jtp = JointTrajectoryPoint()
    #jtp.positions.append(10)
    #jtp.positions = [1,0.5,0.5,0.5,0]
    jtp.positions = [3.14/2,1.5,0,0,0]
    #jtp.positions = [0,0.5,0,-0.5,0.5,0,0]
    jtp.time_from_start = rospy.Duration.from_sec(1)
    jt.points.append(jtp)

    print(jt)

    while not rospy.is_shutdown():
        
        pub.publish(jt)
        rate.sleep()

if __name__ == '__main__':
    try:      
        #if args.tele:
        #    tele_joint()
        #else:
        tele_joint()
    except rospy.ROSInterruptException:
        pass

