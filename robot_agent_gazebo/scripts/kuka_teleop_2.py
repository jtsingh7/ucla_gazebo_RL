#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Float64
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

class jointStates():
    def __init__(self):
        self.jointP = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])



def subCB(data):
    '''header: 
      seq: 518
      stamp: 
        secs: 5
        nsecs: 192000000
      frame_id: ''
    name: 
      - iiwa_joint_1
      - iiwa_joint_2
      - iiwa_joint_3
      - iiwa_joint_4
      - iiwa_joint_5
      - iiwa_joint_6
      - iiwa_joint_7
    position: [-1.608940705022377e-05, 0.1963626679708783, -7.907939288109844e-05, -1.767154179768859, -6.49173048277163e-05, -0.39268994650126454, 2.2980936021710363e-05]
    velocity: [2.631205285295091e-05, 0.013103138345484738, 0.00014266646349337587, -0.008253983489471482, -0.0031963378370673703, 0.009067769304986859, -0.0018270169719301647]
    effort: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'''
    s.jointP=np.array(list(data.position))
    
    

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


def doPublish(jtp,pub_joint1,pub_joint2,pub_joint3,pub_joint4,pub_joint5,pub_joint6,pub_joint7):
    jtp = jtp + s.jointP
    m1 = Float64()
    m2 = Float64()
    m3 = Float64()
    m4 = Float64()
    m5 = Float64()
    m6 = Float64()
    m7 = Float64()
    m1.data = jtp[0]
    m2.data = jtp[1]
    m3.data = jtp[2]
    m4.data = jtp[3]
    m5.data = jtp[4]
    m6.data = jtp[5]
    m7.data = jtp[6]
    pub_joint1.publish(m1)
    pub_joint2.publish(m2)
    pub_joint3.publish(m3)
    pub_joint4.publish(m4)
    pub_joint5.publish(m5)
    pub_joint6.publish(m6)
    pub_joint7.publish(m7)

def tele_joint():

    msg = """
    Interactively set kuka joint positions.
    ---------------------------
    Key controls:
        1 2 3 4 5 6 7
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
    pub_joint1 = rospy.Publisher('/iiwa/EffortJointInterface_J1_controller/command', Float64, queue_size=10)
    pub_joint2 = rospy.Publisher('/iiwa/EffortJointInterface_J2_controller/command', Float64, queue_size=10)
    pub_joint3 = rospy.Publisher('/iiwa/EffortJointInterface_J3_controller/command', Float64, queue_size=10)
    pub_joint4 = rospy.Publisher('/iiwa/EffortJointInterface_J4_controller/command', Float64, queue_size=10)
    pub_joint5 = rospy.Publisher('/iiwa/EffortJointInterface_J5_controller/command', Float64, queue_size=10)
    pub_joint6 = rospy.Publisher('/iiwa/EffortJointInterface_J6_controller/command', Float64, queue_size=10)
    pub_joint7 = rospy.Publisher('/iiwa/EffortJointInterface_J7_controller/command', Float64, queue_size=10)
    sub = rospy.Subscriber("/iiwa/joint_states", JointState, subCB)
    

    status = 0
    i=0
    t=1

    try:
        print(msg)
        jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        while 1:
            key = getKey()
            try:
                key=int(key)
                if key<8 and key>0:
                    print("Editing joint {}".format(key))
                    i = key-1
                else:
                    print("Not a valid joint index")
            except:
                pass
            if key == 'a' :
                jtp[i]+=math.pi/16
                print("Desired joint displacements:")
                print(jtp)
                print("")
            elif key == 'd' :
                jtp[i]-=math.pi/16
                print("Desired joint displacements:")
                print(jtp)
                print("")
            elif key == 's' :
                jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                print("Desired joint displacements:")
                print(jtp)
                print("")
            elif key == 'w':
                print("Sending command")
                doPublish(jtp,pub_joint1,pub_joint2,pub_joint3,pub_joint4,pub_joint5,pub_joint6,pub_joint7)
                jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            else:
                if (key == '\x03'):
                    break
    except:
        print(e)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)    


if __name__ == '__main__':
    try:
        s = jointStates()
        tele_joint()

    except rospy.ROSInterruptException:
        pass

