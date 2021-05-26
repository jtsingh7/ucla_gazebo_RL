#!/usr/bin/python3

import torch as T
from Agent import ActorNN



def evaluate_trained_agent(PolicyNetwork_dir, env, alpha_A, fc1_dims, fc2_dims, render):

	#Create the Actor (Note: that learning rate is arbitrary as we are only evaluating network)
	Actor = ActorNN(input_dims=env.observation_space.low.shape, action_dims=env.action_space.shape[0],
				     alpha_A=alpha_A, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
	Actor.load_state_dict(T.load(PolicyNetwork_dir))

	#Evaluate the Network forever (until process is killed by user)
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(Actor, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)




def rollout(Actor, env, render):

	while True:
		obs = env.reset()
		done = False
		t = 0
		ep_len = 0
		ep_ret = 0
		while not done:
			t += 1
			if render:
				env.render()
			action = Actor(obs).detach().numpy()
			obs, rew, done, _ = env.step(action)
			ep_ret += rew
		ep_len = t
		yield ep_len, ep_ret



def _log_summary(ep_len, ep_ret, ep_num):
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))


		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
