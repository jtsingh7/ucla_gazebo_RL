import os
import gym
import time
import numpy as np
import time
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
import matplotlib.pyplot as plt
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetLinkState, SetModelConfiguration, SetModelConfigurationRequest, ApplyBodyWrench
from tf.transformations import quaternion_matrix as rot
from std_srvs.srv import Empty
import pdb
import geometry_msgs


class PPO_gazebo:
	def __init__(self,task,timesteps_per_batch, max_timesteps_per_episode, n_updates_per_iteration, 
			    gamma, alpha_A, alpha_C, clip, fc1_dims, fc2_dims, save_freq, load_previous_networks, 
			    PolicyNetwork_dir, CriticNetwork_dir, figure_file, joints_in_use=[True,True,True,True,True,True,True]):

		# ROS-related initialization
		rospy.init_node('RL_agent')
		self.task = task # The task to solve
		self.joints_in_use = joints_in_use # this is a boolean 7-vector indicating which joints are in use for the problem
		#ASSUMPTIONS - joint 7 always in use. for any unused joint i, joint i-1 is also unused
		self.num_joints_in_use = 0
		for j in self.joints_in_use:
			if j==True:
				self.num_joints_in_use+=1
		self.dt = 0.1 #sec

		# Publishers and subscribers
		if self.task == 1:
			self.pub_joint1 = rospy.Publisher('/iiwa/EffortJointInterface_J1_controller/command', Float64, queue_size=10)
			self.pub_joint2 = rospy.Publisher('/iiwa/EffortJointInterface_J2_controller/command', Float64, queue_size=10)
			self.pub_joint3 = rospy.Publisher('/iiwa/EffortJointInterface_J3_controller/command', Float64, queue_size=10)
			self.pub_joint4 = rospy.Publisher('/iiwa/EffortJointInterface_J4_controller/command', Float64, queue_size=10)
			self.pub_joint5 = rospy.Publisher('/iiwa/EffortJointInterface_J5_controller/command', Float64, queue_size=10)
			self.pub_joint6 = rospy.Publisher('/iiwa/EffortJointInterface_J6_controller/command', Float64, queue_size=10)
			self.pub_joint7 = rospy.Publisher('/iiwa/EffortJointInterface_J7_controller/command', Float64, queue_size=10)
		elif self.task == 2:
			self.pub_joint1 = rospy.Publisher('/iiwa/EffortJointInterface_J1_controller_E/command', Float64, queue_size=10)
			self.pub_joint2 = rospy.Publisher('/iiwa/EffortJointInterface_J2_controller_E/command', Float64, queue_size=10)
			self.pub_joint3 = rospy.Publisher('/iiwa/EffortJointInterface_J3_controller_E/command', Float64, queue_size=10)
			self.pub_joint4 = rospy.Publisher('/iiwa/EffortJointInterface_J4_controller_E/command', Float64, queue_size=10)
			self.pub_joint5 = rospy.Publisher('/iiwa/EffortJointInterface_J5_controller_E/command', Float64, queue_size=10)
			self.pub_joint6 = rospy.Publisher('/iiwa/EffortJointInterface_J6_controller_E/command', Float64, queue_size=10)
			self.pub_joint7 = rospy.Publisher('/iiwa/EffortJointInterface_J7_controller_E/command', Float64, queue_size=10)
		self.sub_joint_states = rospy.Subscriber("/iiwa/joint_states", JointState, self.subCB_joint_states)
		
		# State tracking
		self.ball_state = []
		self.iiwa_link_0_state = []
		self.iiwa_link_1_state = []
		self.iiwa_link_2_state = []
		self.iiwa_link_3_state = []
		self.iiwa_link_4_state = []
		self.iiwa_link_5_state = []
		self.iiwa_link_6_state = []
		self.iiwa_link_7_state = []
		self.iiwa_joint_states = []
		self.ball_in_plate_frame = []
		self.ball_dist_from_plate_center = []

		# Kuka joint limits (degrees)
		self.joint1_angle_limits = (-170,170)
		self.joint2_angle_limits = (-120,120)
		self.joint3_angle_limits = (-170,170)
		self.joint4_angle_limits = (-120,120)
		self.joint5_angle_limits = (-170,170)
		self.joint6_angle_limits = (-120,120)
		self.joint7_angle_limits = (-175,175)

 		# Setting device to CPU
		global device 
		device= 'cpu'

		# Extract input
		n = self.num_joints_in_use*6 + 2 + 6 + 1 # 51 total if all joints used
		self.obs_dims = (n,) # xyz pos/vel for links, plate theta, plate phi, ball xyz pos/vel, ball dist from plate center
		a=0
		for i in range(len(self.joints_in_use)):
			if self.joints_in_use[i] == True:
				a+=1
		self.action_dims = a
		
		# Define upper and lower bound joint effort commands (found empirically to provide 'good' behavior)
		scaling_array = np.ones(self.action_dims)
		scaling_array[-1] = 5
		scaling_array[-2] = 5
		scaling_array[-3] = 5
		scaling_array[-4] = 75	
		self.effort_lower_bound = -1*np.ones(self.action_dims)*scaling_array
		self.effort_upper_bound = 1*np.ones(self.action_dims)*scaling_array
		
		# Hyperparameters
		self.gamma = gamma
		self.alpha_A = alpha_A
		self.alpha_C = alpha_C
		self.clip = clip
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.timesteps_per_batch = timesteps_per_batch
		self.max_timesteps_per_episode = max_timesteps_per_episode
		self.n_updates_per_iteration = n_updates_per_iteration
		self.save_freq = save_freq
		self.PolicyNetwork_dir = PolicyNetwork_dir
		self.CriticNetwork_dir = CriticNetwork_dir
		self.figure_file = figure_file
		self.score_history = []

		# Initialize actor and critic networks
		self.actor = ActorNN(input_dims=self.obs_dims, action_dims=self.action_dims, alpha_A=self.alpha_A, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)                                                   # ALG STEP 1
		self.critic = CriticNN(input_dims=self.obs_dims, alpha_C=self.alpha_C, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
		if load_previous_networks:
			try:
				self.actor.load_state_dict(T.load(PolicyNetwork_dir))
				self.critic.load_state_dict(T.load(CriticNetwork_dir))
			except:
				print('No previous networks to load. Continuing from scratch')
		self.device = T.device(device if T.cuda.is_available() else 'cpu')

		# Logger
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations of learning so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# Normalizing advantages
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# Updating the Network
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = T.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = T.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses
				actor_loss = (-T.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)
				
				# Zeroing gradient prior to stepping
				self.actor.optimizer.zero_grad()
				self.critic.optimizer.zero_grad()

				# Calculate total loss and back propogate
				total_loss = actor_loss + critic_loss
				total_loss.backward()
				self.actor.optimizer.step()
				self.critic.optimizer.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print training summary
			self._log_summary()

			# Save our model and plot 
			if i_so_far % self.save_freq == 0:
				print('Saving Networks')
				T.save(self.actor.state_dict(), self.PolicyNetwork_dir)
				T.save(self.critic.state_dict(), self.CriticNetwork_dir)
				print('Plotting Figure')
				self.plot_training(self.figure_file)

	def rollout(self):

		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		ep_rewards = []

		t = 0 
		
		#Creating the batch of data
		while t < self.timesteps_per_batch:
			ep_rewards = [] 
			score = 0

			state = self.gazebo_reset()	# reset gazebo environment and get initial state
			done = False

			for ep_t in range(self.max_timesteps_per_episode):
				
				t += 1
				batch_obs.append(state)
				action_scaled, action, log_prob = self.get_action(state)
				state, reward, done = self.gazebo_step(action_scaled)
				score =+ reward
				ep_rewards.append(reward)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				if done:
					self.pause()
					break
					
			# Track score history for plotting (total reward for each episode)
			self.score_history.append(score)

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rewards)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = T.tensor(batch_obs, dtype=T.float).to(self.device)
		batch_acts = T.tensor(batch_acts, dtype=T.float).to(self.device)
		batch_log_probs = T.tensor(batch_log_probs, dtype=T.float).to(self.device)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):

		batch_rtgs = []

		for ep_rewards in reversed(batch_rews):

			discounted_reward = 0 
			for reward in reversed(ep_rewards):
				discounted_reward = reward + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		batch_rtgs = T.tensor(batch_rtgs, dtype=T.float).to(self.device)

		return batch_rtgs

	def get_action(self, state):

		alpha, beta = self.actor(state)
		policy_dist = Beta(alpha, beta)
		action = policy_dist.sample()
		log_prob = policy_dist.log_prob(action).sum(-1) #summing log prob for each action sampled

		if self.task == 1:
			#Map Action to joint position command range
			action_low = []
			action_high = []
			for j in range(len(self.joints_in_use)):
				if self.joints_in_use[j] == True:
					low, high = self.get_joint_limits(j)
					action_low.append(low)
					action_high.append(high)
			action_low = np.array(action_low)
			action_high = np.array(action_high)
		if self.task == 2:
			action_low = self.effort_lower_bound
			action_high = self.effort_upper_bound

		action_scaled = ((action)/(1))*(action_high - action_low) + action_low

		# Returns the action and the scaled action. The action is used for computing the log prob, and the scaled action is used in the environmnet
		try:
			return action_scaled.detach().numpy(), action.detach().numpy(), log_prob.detach()
		except:
			return action_scaled.cpu().detach().numpy(), action.cpu().detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		V = self.critic(batch_obs).squeeze()

		alpha, beta = self.actor(batch_obs)
		policy_dist = Beta(alpha, beta)
		log_probs = policy_dist.log_prob(batch_acts).sum(-1) #summing log prob for each action sampled

		return V, log_probs

	def _log_summary(self):

		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.item() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration of learning #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		
		
	def plot_training(self, filename):
		running_avg = np.zeros(len(self.score_history))
		bucket = 100
		for i in range(len(running_avg)):
			running_avg[i] = np.mean(self.score_history[max(0, i-bucket):(i+1)])
		x = np.arange(1,len(running_avg)+1,1)
		plt.plot(x, running_avg)
		plt.title(('Running average of total return from previous 100 episodes'))
		plt.xlabel('Episode')
		plt.ylabel('Average Total Return')
		plt.savefig(filename)


	def get_gazebo_state(self):
		
		link_ros = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
		self.ball_state = link_ros('ball','world')
		self.iiwa_link_1_state = link_ros('iiwa_link_1','world')
		self.iiwa_link_2_state = link_ros('iiwa_link_2','world')
		self.iiwa_link_3_state = link_ros('iiwa_link_3','world')
		self.iiwa_link_4_state = link_ros('iiwa_link_4','world')
		self.iiwa_link_5_state = link_ros('iiwa_link_5','world')
		self.iiwa_link_6_state = link_ros('iiwa_link_6','world')
		self.iiwa_link_7_state = link_ros('iiwa_link_7','world')
		self.ball_in_plate_frame = link_ros('ball','iiwa_link_7')
		'''
		link_state: 
		  link_name: "iiwa_link_7"
		  pose: 
		    position: 
		      x: 0.5265679583380167
		      y: 2.959748300857525e-05
		      z: 0.5725562939026705
		    orientation: 
		      x: 0.0005520473927489056
		      y: 0.7155809849641402
		      z: 0.00030666804927524844
		      w: 0.6985294948362127
		  twist: 
		    linear: 
		      x: 1.0495646696811689e-05
		      y: 2.8729811809361537e-05
		      z: 0.00010343615669669576
		    angular: 
		      x: -8.324382197011424e-05
		      y: -0.00021473057717805527
		      z: 0.00021159956898025696
		  reference_frame: "world"
		success: True
		status_message: "GetLinkState: got state"
		'''
		theta, phi = self.plate_angles()
		self.ball_plate_dist_calc()

		
		state_list = []

		for j in range(len(self.joints_in_use)): #add link states
			if self.joints_in_use[j]==True:
				px,py,pz,vx,vy,vz = self.get_link_state(j)
				state_list.append(px)
				state_list.append(py)
				state_list.append(pz)
				state_list.append(vx)
				state_list.append(vy)
				state_list.append(vz)
			
		state_list.append(theta)
		state_list.append(phi)
		state_list.append(self.ball_state.link_state.pose.position.x)
		state_list.append(self.ball_state.link_state.pose.position.y)
		state_list.append(self.ball_state.link_state.pose.position.z)
		state_list.append(self.ball_state.link_state.twist.linear.x)
		state_list.append(self.ball_state.link_state.twist.linear.y)
		state_list.append(self.ball_state.link_state.twist.linear.z)
		state_list.append(self.ball_dist_from_plate_center)

		state = np.array(state_list)
		print(state)

		return state

	def plate_angles(self):

		hm = rot([self.iiwa_link_7_state.link_state.pose.orientation.x,
					self.iiwa_link_7_state.link_state.pose.orientation.y,
					self.iiwa_link_7_state.link_state.pose.orientation.z,
					self.iiwa_link_7_state.link_state.pose.orientation.w])
		theta_rad = np.arcsin(-hm[2,2])
		phi_rad = np.arcsin(-hm[2,1])

		theta_deg = theta_rad * (180/np.pi)
		phi_deg = phi_rad * (180/np.pi)

		return theta_deg, phi_deg

	def ball_plate_dist_calc(self):

		# Suppose the iiwa_link_7 frame was centered at the plate center:
		ball_plate_y = self.ball_in_plate_frame.link_state.pose.position.y
		ball_plate_z = self.ball_in_plate_frame.link_state.pose.position.z - 0.2

		self.ball_dist_from_plate_center = np.linalg.norm(np.array((ball_plate_z,ball_plate_y)))

	def subCB_joint_states(self,data):

		self.iiwa_joint_states = data

		# A message looks like this:
		'''header: 
		  seq: 175010
		  stamp: 
		    secs: 1750
		    nsecs: 112000000
		  frame_id: ''
		name: 
		  - iiwa_joint_1
		  - iiwa_joint_2
		  - iiwa_joint_3
		  - iiwa_joint_4
		  - iiwa_joint_5
		  - iiwa_joint_6
		  - iiwa_joint_7
		position: [-6.947010247948526e-06, 0.19634947740073727, 6.537230698100416e-06, -1.7671458774993676, -1.75296373683409e-06, -0.3926990854628787, 2.4836897667412927e-06]
		velocity: [-0.010000948512075985, -0.00012689598800587644, 0.010264604016930764, -1.9681708510272394e-05, -0.0013172871692685053, -7.528297296069183e-06, 0.0013265745694800732]
		effort: [-1.1170659660860227, -42.19300206382932, -0.28039639378316916, 30.654842491443432, -1.7280505378803077, -3.1108899151416978, 0.004992888739265563]
		'''

	def reward_from_state(self,state,control):
		'''Calculate reward based on state and control effort.

		Params
		----------
		state : np.array
			array of [joint positions 1-7,joint velocities 1-7,plate roll,plate pitch,plate xyz,ball xyz,ball velocity xyz,ball dist from plate center]
		control : np.array
			The position or effort commands, depending on task
		'''

		#constants to be tuned
		c1 = 1
		#c2 = 1
		#c3 = 0
		#c4 = 1
		#c5 = 100
		#c6 = 100 #floor gain
		#c7 = 15  #floor shape

		#calculated reward of ball position
		ball_dist_from_plate_center = state[-1]
		r_ball = np.exp(-c1*(ball_dist_from_plate_center)) #larger penalty for larger distances from center of plate

		#calculated reward of plate angle
		#theta = state[14]
		#phi = state[15]
		
		#r_plate = -c2*(phi**2 + theta**2) #penalty for large plate angles


		#if self.task == 1: # Position control
			
		#	r_goal = 0
		#	num_violations = self.check_position_limits(control)
		#	r_action = -c5*num_violations

		#elif self.task == 2: # Effort control
		#	r_action = -c3*(np.dot(control,control)) #penalty for torque commands

			#TODO make the ball cartesian goal not hardcoded
		#	xgoal=0.67
		#	ygoal=0
		#	zgoal=0.7
		#	xerr = state[19] - xgoal
		#	yerr = state[20] - ygoal
		#	zerr = state[21] - zgoal

		#	r_goal = -c4*(xerr**2 + yerr**2 + zerr**2)
			
		#calculated Floor Penalty 
		#Description: At heights >0.3, Reward is ~0. At heights ~0.3, Reward is ~-1. Rapid dropoff with max negative reward of ~-50 at the min ball height of ~0.05
		#r_floor = -c6*np.exp(-c7*self.ball_state.link_state.pose.position.z)

		#Total Reward
		R = r_ball 
		
		return R

	def check_position_limits(self,commands):
		num_violations = 0
		i=0
		for j in range(len(self.joints_in_use)):
			if self.joints_in_use[j] == True:
				action = commands[i]
				lower, upper = self.get_joint_limits(j)
				if action>upper or action<lower:
					num_violations+=1
				i+=1
		return num_violations


	def get_link_state(self,j):
		if j == 0:
			return self.iiwa_link_1_state.link_state.pose.position.x, \
					self.iiwa_link_1_state.link_state.pose.position.y, \
					self.iiwa_link_1_state.link_state.pose.position.z, \
					self.iiwa_link_1_state.link_state.twist.linear.x, \
					self.iiwa_link_1_state.link_state.twist.linear.y, \
					self.iiwa_link_1_state.link_state.twist.linear.z 

		elif j == 1:
			return self.iiwa_link_2_state.link_state.pose.position.x, \
					self.iiwa_link_2_state.link_state.pose.position.y, \
					self.iiwa_link_2_state.link_state.pose.position.z, \
					self.iiwa_link_2_state.link_state.twist.linear.x, \
					self.iiwa_link_2_state.link_state.twist.linear.y, \
					self.iiwa_link_2_state.link_state.twist.linear.z 

		elif j == 2:
			return self.iiwa_link_3_state.link_state.pose.position.x, \
					self.iiwa_link_3_state.link_state.pose.position.y, \
					self.iiwa_link_3_state.link_state.pose.position.z, \
					self.iiwa_link_3_state.link_state.twist.linear.x, \
					self.iiwa_link_3_state.link_state.twist.linear.y, \
					self.iiwa_link_3_state.link_state.twist.linear.z 

		elif j == 3:
			return self.iiwa_link_4_state.link_state.pose.position.x, \
					self.iiwa_link_4_state.link_state.pose.position.y, \
					self.iiwa_link_4_state.link_state.pose.position.z, \
					self.iiwa_link_4_state.link_state.twist.linear.x, \
					self.iiwa_link_4_state.link_state.twist.linear.y, \
					self.iiwa_link_4_state.link_state.twist.linear.z 

		elif j == 4:
			return self.iiwa_link_5_state.link_state.pose.position.x, \
					self.iiwa_link_5_state.link_state.pose.position.y, \
					self.iiwa_link_5_state.link_state.pose.position.z, \
					self.iiwa_link_5_state.link_state.twist.linear.x, \
					self.iiwa_link_5_state.link_state.twist.linear.y, \
					self.iiwa_link_5_state.link_state.twist.linear.z 

		elif j == 5:
			return self.iiwa_link_6_state.link_state.pose.position.x, \
					self.iiwa_link_6_state.link_state.pose.position.y, \
					self.iiwa_link_6_state.link_state.pose.position.z, \
					self.iiwa_link_6_state.link_state.twist.linear.x, \
					self.iiwa_link_6_state.link_state.twist.linear.y, \
					self.iiwa_link_6_state.link_state.twist.linear.z 

		elif j == 6:
			return self.iiwa_link_7_state.link_state.pose.position.x, \
					self.iiwa_link_7_state.link_state.pose.position.y, \
					self.iiwa_link_7_state.link_state.pose.position.z, \
					self.iiwa_link_7_state.link_state.twist.linear.x, \
					self.iiwa_link_7_state.link_state.twist.linear.y, \
					self.iiwa_link_7_state.link_state.twist.linear.z 


	def get_joint_limits(self,joint_num):
		# returns joint limits (in radians)

		c = np.pi/180

		if joint_num == 0:
			return c*self.joint1_angle_limits[0], c*self.joint1_angle_limits[1]

		elif joint_num == 1:
			return c*self.joint2_angle_limits[0], c*self.joint2_angle_limits[1]

		elif joint_num == 2:
			return c*self.joint3_angle_limits[0], c*self.joint3_angle_limits[1]

		elif joint_num == 3:
			return c*self.joint4_angle_limits[0], c*self.joint4_angle_limits[1]

		elif joint_num == 4:
			return c*self.joint5_angle_limits[0], c*self.joint5_angle_limits[1]

		elif joint_num == 5:
			return c*self.joint6_angle_limits[0], c*self.joint6_angle_limits[1]

		elif joint_num == 6:
			return c*self.joint7_angle_limits[0], c*self.joint7_angle_limits[1]


	def gazebo_step(self,action):

		# 1. Publish action
		# 2. Wait small amount of time
		# 3. Get new state
		# 4. Calculate reward based on new state

		if self.task == 1:
			self.apply_wind()

		self.send_actions(action)

		self.wait()

		new_state = self.get_gazebo_state()

		reward = self.reward_from_state(new_state,action)

		#Done if ball is on ground
		if self.ball_state.link_state.pose.position.z <= 0.05:
			done = True
		else:
			done = False

		return new_state, reward, done


	def gazebo_reset(self):

		rospy.wait_for_service('/gazebo/reset_simulation') 
		reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		reset()

		rospy.wait_for_service('/gazebo/pause_physics')
		pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		pause()

		rospy.wait_for_service('/gazebo/set_model_configuration')
		config = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
		req = SetModelConfigurationRequest()
		req.model_name = 'kuka_with_plate'
		req.urdf_param_name = 'robot_description'
		req.joint_names = ['iiwa_joint_1','iiwa_joint_2','iiwa_joint_3','iiwa_joint_4','iiwa_joint_5','iiwa_joint_6','iiwa_link_7']
		req.joint_positions = [0, 0.19634954084936207, 0, -1.7671458676442586, 0, -0.39269908169872414, 0]
		config(req)

		rospy.wait_for_service('/gazebo/unpause_physics')
		unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		unpause()

		if not self.iiwa_joint_states:
			print("Waiting for joint states...")
			while not self.iiwa_joint_states:
				pass

		state = self.get_gazebo_state()
		
		return state

	def wait(self):
		rospy.sleep(self.dt)
	
	def pause(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		
	def send_actions(self,actions):
		# TODO: see if publishers have a "receipt" ability to let you know when message was actually received?

		# j: joint index
		# i: action index
		# this loop maps actions to joints
		i=0
		for j in range(len(self.joints_in_use)):
			if self.joints_in_use[j] == True:
				self.doPublish(j,actions[i])
				i+=1

	def doPublish(self,joint,action):
		if joint == 0:
			self.pub_joint1.publish(action)
		elif joint == 1:
			self.pub_joint2.publish(action)
		elif joint == 2:
			self.pub_joint3.publish(action)
		elif joint == 3:
			self.pub_joint4.publish(action)
		elif joint == 4:
			self.pub_joint5.publish(action)
		elif joint == 5:
			self.pub_joint6.publish(action)
		elif joint == 6:
			self.pub_joint7.publish(action)

	def apply_wind(self):
		
		x_coord = np.random.uniform(0,1)
		y_coord = np.random.uniform(0,1)
		z_coord = np.random.uniform(0,1)
		norm = np.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
		x_coord = x_coord/norm
		y_coord = y_coord/norm
		z_coord = z_coord/norm
		ref_pt = geometry_msgs.msg.Point(x = 0, y = 0, z = 0)
		wrench = geometry_msgs.msg.Wrench(force = geometry_msgs.msg.Vector3( x = x_coord, y = y_coord, z = z_coord), torque = geometry_msgs.msg.Vector3( x = 0, y = 0, z = 0))
		start_time = rospy.Time(secs = 0, nsecs = 0)
		duration = rospy.Duration(secs = 0.1, nsecs = 0)
		rospy.wait_for_service('/gazebo/apply_body_wrench')
		apply_force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
		apply_force('ball', 'world', ref_pt, wrench, start_time, duration)


class ActorNN(nn.Module):
	def __init__(self, input_dims, action_dims, alpha_A, fc1_dims, fc2_dims):
		super(ActorNN, self).__init__()
		self.action_dims = action_dims
		self.actor = nn.Sequential(
                     nn.Linear(*input_dims, fc1_dims), 
                     nn.ReLU(),
                     nn.Linear(fc1_dims, fc2_dims),
                     nn.ReLU(),
                     nn.Linear(fc2_dims, action_dims),
		     nn.Softplus(),
		     nn.Linear(action_dims, action_dims),
		     nn.Softplus()
                     )

		self.optimizer = optim.Adam(self.parameters(), lr=alpha_A)

		self.device = T.device(device if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		if isinstance(state, np.ndarray):
			state = T.tensor(state, dtype=T.float).to(self.device)
		# Alpha from second to last layer
		state_alpha = state
		for layer in range(len(self.actor)):
			state_alpha = self.actor[layer](state_alpha)
			if layer == 5:
				alpha = state_alpha
				break
		# Beta from last layer
		beta = self.actor(state)
			
		return alpha, beta

	

class CriticNN(nn.Module):
	def __init__(self, input_dims, alpha_C, fc1_dims, fc2_dims):
		super(CriticNN, self).__init__()

		self.critic = nn.Sequential(
                      nn.Linear(*input_dims, fc1_dims),
                      nn.ReLU(),
                      nn.Linear(fc1_dims, fc2_dims),
                      nn.ReLU(),
                      nn.Linear(fc2_dims, 1)
                      )

		self.optimizer = optim.Adam(self.parameters(), lr=alpha_C)
		self.device = T.device(device if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		if isinstance(state, np.ndarray):
			state = T.tensor(state, dtype=T.float).to(self.device)

		value = self.critic(state)
		return value

