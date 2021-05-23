#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetLinkState
import sys, select, os
import argparse
import csv
import math
import rospy
import roslaunch
from datetime import datetime
import time
import copy
import numpy as np
from std_srvs.srv import Empty
if os.name == 'nt':
  import msvcrt
else:
  import tty, termios
if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)


#Hyperparams
CLIP = 0.2
lr = 1e-3
RUNS = 400
EPI = 100
GAMMA = 0.98
LAMBDA = 0.95
STEP_MAX = 20
BATCH_SIZE = 20
# tf.random.set_seed(123123)


#storage
mem_reward = []
mem_value = []
mem_state = []
mem_action = []
mem_prob = []
mem_term = []
mem_score = []


#ROS commands
class jointStates():
    def __init__(self):
        self.jointP = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
s = jointStates()
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
pubJoint = rospy.Publisher('/iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=10)                                           #publish ball position???????idk
subJoint = rospy.Subscriber("/iiwa/joint_states", JointState, subCB)
state_ros = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
rospy.init_node('iiwa_joints', anonymous=True)
# rospy.spin()
# rate = rospy.Rate(2)                                                                                                  #not sure abt this line
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    # self.d0 = tf.keras.layers.InputLayer(input_shape = (10,))
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.d2 = tf.keras.layers.Dense(128,activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)
  def call(self, input_data):
    # x0 = self.d0(input_data)
    x1 = self.d1(input_data)
    x2 = self.d2(x1)
    v = self.v(x2)
    return v

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    # self.d0 = tf.keras.layers.InputLayer(input_shape = (10,))
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.d2 = tf.keras.layers.Dense(128,activation='relu')
    self.a = tf.keras.layers.Dense(15,activation='softmax')
  def call(self, input_data):
    # x0 = self.d0(input_data)
    x1 = self.d1(input_data)
    x2 = self.d2(x1)
    a = self.a(x2)
    return a

class agent():
    def __init__(self):
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.actor = actor()
        self.critic = critic()
        self.clip = CLIP
        self.gamma = GAMMA
        self.gaelambda = LAMBDA
    def pick_action(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    def GAE(self, states, actions, rewards, values, dones):
        adv = []
        last_adv = 0
        # rewards = rewards[::-1]
        for i in reversed(range(len(rewards)-1)):
            delta = rewards[i] + self.gamma*values[i+1]*(1-done[i]) - values[i]
            adv[i] = last_adv = delta + last_adv*self.gamma*self.gaelambda
        returns = adv[::-1]
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv))/np.std(adv)
        return returns, adv
    def actor_loss(self, probs, actions, adv, old_probs, c_loss):
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs,tf.math.log(probs))))
        ratios = []
        clip_ratios = []
        for pb, op, A in zip(probs, old_probs, adv):
            t = tf.constant(t)
            op = tf.constant(op)
            ratio = tf.divide(pb,op)
            ratios.append(tf.math.multiply(ratio,t))                                                                        #ratio of prob/old_probs * Advantage
            clip_ratios.append(tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip),t))               #clipped
        st1 = tf.stack(ratios)
        st2 = tf.stack(clip_ratios)
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(st1, st2)) - closs + 0.001 * entropy)
        return loss
    def learn(self, states, actions, adv, old_probs, rewards):
        rewards = tf.reshape(rewards, (len(rewards),))
        adv = reshape(adv, (len(adv),))
        old_probs = reshape(old_probs, (len(adv),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(rewards, v)                   #critic loss
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

class environment():
    def __init__(self):
        self.terminal = False
        self.epiStartTime = rospy.get_time()
        # self.lastActionTime = rospy.get_time()
    def get_state(self):
        # jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])         #same as s.jointP
        # jt = get_jt(jtp,1)
        # jt = np.array(jt.points[0].positions)
        #State is 7 joint angles + 3 coords for ball
        # state_ros = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        rospy.wait_for_service('/gazebo/get_link_state')
        od = state_ros('ball', 'world')
        x = od.link_state.pose.position.x
        y = od.link_state.pose.position.y
        z = od.link_state.pose.position.z
        state = np.append(s.jointP,np.array([x,y,z]))
        return state
    def reset(self):                                                            #could/should have used rospy reset service proxy
        self.epiStartTime = rospy.get_time()
        rospy.wait_for_service('/gazebo/reset_simulation')
        reset_simulation()                                      #resets ball
        jtp_list = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        jt = JointTrajectory()
        jt.joint_names = ["iiwa_joint_1","iiwa_joint_2","iiwa_joint_3","iiwa_joint_4","iiwa_joint_5","iiwa_joint_6","iiwa_joint_7"]
        jtp = JointTrajectoryPoint()
        jtp.positions = jtp_list
        jtp.time_from_start = rospy.Duration.from_sec(1)
        jt.points.append(jtp)
        pubJoint.publish(jt)                                    #reset  arm
    def take_action(self,action):
        #Discrete Actions, #TODO:Make continuous action space
        #Action space: joint + or -, 7 joints--->15 actions
        #Action 1-7 move joints in positive dir, 0  is no action
        #Action 8-14 move joints in negative dir
        jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        print('action taken:')
        print(action)
        if action <= 7 and action>0:
            jtp[action-1]+=math.pi/16
        elif action >= 8 and action <= 14 :
            jtp[action-1-7]-=math.pi/16
        jt = get_jt(jtp,1)
        pubJoint.publish(jt)
    def step(self,action):
        curr_state = self.get_state()
        self.take_action(action)
        new_state = self.get_state()
        reward = (rospy.get_time() - self.epiStartTime)/60
        print('reward')
        print(reward)
        # self.lastActionTime = rospy.get_time()
        if new_state[-1] <= 0:                                                                  #if ball falls below z=0, terminate episode
            self.terminal = True
        return curr_state, new_state, reward, self.terminal

def generate_batches(states, actions, probs, values, rewards, dones):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, BATCH_SIZE)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+BATCH_SIZE] for i in batch_start] #all batches created
        state_b = states[batches[0]]
        prob_b = probs[batches[0]]
        action_b = actions[batches[0]]
        reward_b = rewards[batches[0]]
        done_b = dones[batches[0]]
        value_b = values[batches[0]]
        return state_b, prob_b, action_b, reward_b, done_b, value_b

def train(arm,env):
    figure_file = '/mnt/c/Users/Ankit/Desktop/ppo_plots\curve.png'                                 #will need to create plots folder before running
    # pause()
    env.reset()
    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause()
    # rate.sleep()                                                      #wait for ball to hit plate after unpausing sim
    curr_state = env.get_state()
    done = False
    learn_steps = 0
    best_score = 0
    avg_score = 0
    for i in range(RUNS):                                               #TODO:finish algo
        score = 0
        steps = 0
        while not done:
            action = arm.pick_action(curr_state)
            value = arm.critic(np.array([curr_state])).numpy()
            prob = arm.actor(np.array([curr_state])).numpy()
            old_state, curr_state, reward, done = env.step(action)
            mem_state.append(old_state)
            mem_action.append(action)
            mem_reward.append(reward)
            score += reward
            mem_term.append(done)
            mem_prob.append(prob[0])
            mem_value.append(value[0][0])
            if steps % STEP_MAX == 0 and steps > 60:
                state_b, prob_b, action_b, reward_b, done_b, value_b = generate_batches(mem_state, mem_action, mem_prob, mem_value, mem_reward, mem_term)
                ret, adv = arm.GAE(state_b, action_b, reward_b, value_b, done_b)
                arm.learn(state_b, action_b, adv, prob_b, ret)
                learn_steps += 1
            else:
                 steps += 1
            old_state = curr_state
        env.reset()
        mem_score.append(score)
        if len(mem_score)>100:
            avg_score = np.mean(mem_score[-100:])
        if avg_score > best_score:
            best_score = avg_score
            arm.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            arm.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', steps, 'learning_steps', learn_steps)
    x = [game+1 for game in range(len(mem_score))]
    plt.plot(x,mem_score)
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.savefig(figure_file)
def main():
    print('asdfasdf')
    arm = agent()
    env = environment()
    train(arm,env)

if __name__ == '__main__':
    main()
