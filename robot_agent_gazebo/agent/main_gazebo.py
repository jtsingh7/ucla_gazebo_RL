#!/usr/bin/env python3

import os
import gym
import sys
import torch
from agent_gazebo import PPO_gazebo
from evaluate_policy_gazebo import evaluate_trained_agent


if __name__ == '__main__':

	#########################################################################################  SETUP  ##############################################################################################################
	#Test or Train Flag
	Train = True
	Test  = False

	#Hyperparamters
	timesteps_per_batch = 2048             #Number of timesteps for each batch of learning
	max_timesteps_per_episode = 200	       #Maximum number of timesteps in each episode. Each episode is basically a 'game'
	n_updates_per_iteration = 10           #Number of epochs per batch
	gamma = 0.99                           #Discount factor
	alpha_A = 3e-4                         #Learning rate for Actor Network
	alpha_C = 6e-4                         #Learning rate for Critic Network
	clip = 0.2                             #PPO clipping factor
	fc1_dims = 64                          #Dimensions of first hidden layer
	fc2_dims = 64                          #Dimensions of second hidden layer

	#Miscellaneous parameters
	task = 2                               #TODO pull task number from launch file, only placeholder
	save_freq = 10                         #Save every i iterations
	load_previous_networks = True          #Load actor and critic networks from previous training, will default to start training from scratch if networks do not exist
	total_learn_timesteps = 500_000_000    #Total timesteps to learn before exiting training loop, typically set to a high number to continue learning indefinitely
	PolicyNetwork_dir = os.getcwd() + '\\Networks\\ppo_actor.pth'
	CriticNetwork_dir = os.getcwd() + '\\Networks\\ppo_critic.pth'
	figure_file = 'plots/' + 'arm' + '.png' #will need to create plots folder before running

	################################################################################################################################################################################################################







	#####################################################################################  EXECUTING ###############################################################################################################
	#Create the Agent
	agent = PPO_gazebo(task, timesteps_per_batch, max_timesteps_per_episode, n_updates_per_iteration, 
			    gamma, alpha_A, alpha_C, clip, fc1_dims, fc2_dims, save_freq, load_previous_networks, 
			    PolicyNetwork_dir, CriticNetwork_dir,[False,False,True,True,True,True,True])

	#Learn
	if Train == True:
		agent.learn(total_timesteps=total_learn_timesteps)
		agent.plot_training(figure_file)

	#Test
	if Test == True:
		evaluate_trained_agent(PolicyNetwork_dir, env, alpha_A, fc1_dims, fc2_dims, render)
	################################################################################################################################################################################################################

	
