import os
import gym
import time
import numpy as np
import time
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import GetModelState


class PPO_gazebo:
	def __init__(self, timesteps_per_batch, max_timesteps_per_episode, n_updates_per_iteration, 
			    gamma, alpha_A, alpha_C, clip, fc1_dims, fc2_dims, render, render_every_i, save_freq, load_previous_networks, PolicyNetwork_dir, CriticNetwork_dir):

		# ROS-related initialization
		rospy.init_node('RL_agent')
		self.pub_joint1 = rospy.Publisher('/iiwa/EffortJointInterface_J1_controller/command', Float64, queue_size=10)
		self.pub_joint2 = rospy.Publisher('/iiwa/EffortJointInterface_J2_controller/command', Float64, queue_size=10)
		self.pub_joint3 = rospy.Publisher('/iiwa/EffortJointInterface_J3_controller/command', Float64, queue_size=10)
		self.pub_joint4 = rospy.Publisher('/iiwa/EffortJointInterface_J4_controller/command', Float64, queue_size=10)
		self.pub_joint5 = rospy.Publisher('/iiwa/EffortJointInterface_J5_controller/command', Float64, queue_size=10)
		self.pub_joint6 = rospy.Publisher('/iiwa/EffortJointInterface_J6_controller/command', Float64, queue_size=10)
		self.pub_joint7 = rospy.Publisher('/iiwa/EffortJointInterface_J7_controller/command', Float64, queue_size=10)
		self.sub_joint_states = rospy.Subscriber("/iiwa/joint_states", JointState, self.subCB_joint_states)
		
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

		# Debugging (I found that it's about 10x faster to run on a CPU than a GPU, which is counterintuitive and different from previous agents I've run. Will look into, but setting default device to be CPU for now)
		global device 
		device= 'cpu'

		# Extract input
		self.obs_dims = (10,)
		self.action_dims = 7

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
		self.render = render
		self.render_every_i = render_every_i
		self.save_freq = save_freq

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

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = T.full(size=(self.action_dims,), fill_value=0.5)
		self.cov_mat = T.diag(self.cov_var).to(self.device)

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

				# Calculate actor and critic losses.
				actor_loss = (-T.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate total loss and back propogate
				total_loss = actor_loss + critic_loss
				total_loss.backward()
				self.actor.optimizer.step()
				self.critic.optimizer.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print training summary
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				print('Saving Networks')
				actor_network_dir = os.getcwd() + '\\Networks\\ppo_actor.pth'
				critic_network_dir = os.getcwd() + '\\Networks\\ppo_critic.pth'
				T.save(self.actor.state_dict(), actor_network_dir)
				T.save(self.critic.state_dict(), critic_network_dir)



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

			#state = self.env.reset()
			state = self.get_gazebo_state()
			done = False

			for ep_t in range(self.max_timesteps_per_episode):
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()
				t += 1 
				batch_obs.append(state)
				action, log_prob = self.get_action(state)
				state, reward, done, _ = self.env.step(action)
				ep_rewards.append(reward)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				if done:
					break


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

		mean_actions = self.actor(state)
		policy_dist = MultivariateNormal(mean_actions, self.cov_mat)
		action = policy_dist.sample()
		log_prob = policy_dist.log_prob(action)
		try:
			return action.detach().numpy(), log_prob.detach()
		except:
			return action.cpu().detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		V = self.critic(batch_obs).squeeze()

		mean_actions = self.actor(batch_obs)
		policy_dist = MultivariateNormal(mean_actions, self.cov_mat)
		log_probs = policy_dist.log_prob(batch_acts)

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

	def get_gazebo_state(self):
		
		link_ros = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
		self.ball_state = link_ros('ball','world')
		self.iiwa_link_0_state = link_ros('iiwa_link_0','world')
		self.iiwa_link_1_state = link_ros('iiwa_link_1','world')
		self.iiwa_link_2_state = link_ros('iiwa_link_2','world')
		self.iiwa_link_3_state = link_ros('iiwa_link_3','world')
		self.iiwa_link_4_state = link_ros('iiwa_link_4','world')
		self.iiwa_link_5_state = link_ros('iiwa_link_5','world')
		self.iiwa_link_6_state = link_ros('iiwa_link_6','world')
		self.iiwa_link_7_state = link_ros('iiwa_link_7','world')
		self.ball_in_plate_frame = link_ros('ball','iiwa_link_7')

		self.ball_plate_dist_calc()

		#return self.ball_dist_from_plate_center #TODO: decide the format of the whole state
		return Float64()

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






class ActorNN(nn.Module):
	def __init__(self, input_dims, action_dims, alpha_A, fc1_dims, fc2_dims):
		super(ActorNN, self).__init__()
		self.actor = nn.Sequential(
                     nn.Linear(*input_dims, fc1_dims), 
                     nn.ReLU(),
                     nn.Linear(fc1_dims, fc2_dims),
                     nn.ReLU(),
                     nn.Linear(fc2_dims, action_dims),
                     )

		self.optimizer = optim.Adam(self.parameters(), lr=alpha_A)

		self.device = T.device(device if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		if isinstance(state, np.ndarray):
			state = T.tensor(state, dtype=T.float).to(self.device)

		mean_actions = self.actor(state)
		return mean_actions




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

