#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

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


#Hyperparams
CLIP = 0.2
lr = 1e-2
RUNS = 50
EPI = 100
# tf.random.set_seed(123123)


#storage
epi_reward = []
epi_state = []
epi_action = []
epi_probs = []
epi_term = []


#ROS commands
def subCB(data):                                                                                                                  #copied from kuka_teleop.py, may need editing
    s.jointP=np.array(list(data.position))
def get_jt(jtp_list,t):
    jtp_list = jtp_list + s.jointP
    jt = JointTrajectory()
    jt.joint_names = ["iiwa_joint_1","iiwa_joint_2","iiwa_joint_3","iiwa_joint_4","iiwa_joint_5","iiwa_joint_6","iiwa_joint_7"]
    jtp = JointTrajectoryPoint()
    jtp.positions = jtp_list
    jtp.time_from_start = rospy.Duration.from_sec(t)
    jt.points.append(jtp)
    return jt
pubJoint = rospy.Publisher('/iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=10)
pubBall = rospy.Publisher('/gazebo/set_model_state', JointTrajectory, queue_size=10)                                             #publish ball position???????idk
sub = rospy.Subscriber("/iiwa/joint_states", JointState, subCB)
rospy.init_node('iiwa_joints')                                                                                                  #not sure abt this line
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')                              #random NN, TODO:design network architecture
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    v = self.v(x)
    return v

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')                              #random NN, TODO:design network architecture
    self.a = tf.keras.layers.Dense(2,activation='softmax')

  def call(self, input_data):
    x = self.d1(input_data)
    a = self.a(x)
    return a

class agent():
    def __init__(self):
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = CLIP
    def reset():
        jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        jt = get_jt(jtp,0)
        pubJoint.publish(jt)
        self.state[1:8] = jt
    def pick_action(self,state):
        prob = self.actor(np.array(state))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        epi_probs.append(prob)
        return int(action.numpy()[0])

class environment():
    def __init__(self):
        self.terminal = false
        self.epiStartTime = 0
        self.lastActionTime = 0
    def get_state(self):
        jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        jt = get_jt(jtp,0)
        #State is 7 joint angles + 3 coords for ball
        model_coordinates = rospy.ServiceProxy( '/gazebo/model_states', '')                     #read rostopic for ball coords
        object_coordinates = self.model_coordinates("the_ball")
        z_position = self.object_coordinates.pose.position.z
        y_position = self.object_coordinates.pose.position.y
        x_position = self.object_coordinates.pose.position.x
        state = jt.append([x_position, y_position, z_position])
        return state
    def env_reset(self):                                                            #could/should have used rospy reset service proxy
        self.epiStartTime = 0
        jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        jt = get_jt(jtp,0)
        pubJoint.publish(jt)
        initBallPose = np.array([0.7,0.0,0.8])
        pubBall.publish(initBallPose)                                   #reset ball position??idk
        return jt.append(initBallPose)
    def take_action(self,action):
        #Discrete Actions, #TODO:Make continuous action space
        #Action space: joint + or -, 7 joints--->14 actions
        #Action 1-7 move joints in positive dir
        #Action 8-14 move joints in negative dir
        jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            if action <= 7 :
                jtp[action-1]+=math.pi/16
            elif action >= 8 :
                jtp[action-7-1]-=math.pi/16
        jt = self.get_jt(jtp,0)
        pubJoint.publish(jt)
    def env_step(self,action):
        curr_state = get_state()
        take_action(action)
        new_state = get_state()
        reward = rospy.get_time() - self.epiStartTime - self.lastActionTime                 #time since last action
        self.lastActionTime = 0
        if new_state[-1] <= 0:                                                                  #if ball falls below z=0, terminate episode
            self.terminal = true
        return old_state, new_state, reward, self.terminal


def train(arm,env):
    pause()
    env.reset()
    unpause()
    rospy.sleep(1)                                                      #wait for ball to hit plate after unpausing sim
    curr_state = env.get_state()
    done = false
    epi_state.append(curr_state)
    for i in range(RUNS):                                               #TODO:finish algo
        while done==false:
            action = arm.pick_action(curr_state)
            old_state, curr_state, reward, done = env.step(action)
            epi_state.append(curr_state)
            epi_action.append(action)
            epi_reward.append(reward)
            epi_term.append(done)
        env.reset()


def main():
    arm = agent()
    env = environment()
    train(arm,env)

if __name__ == '__main__':
    main()
