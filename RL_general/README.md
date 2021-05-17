Initial upload of the PPO agent. Agent.py is where the PPO agent lives, utils.py has a few extra functions for plotting. Uses Pytorch



To Do:
1) Update the PPO agent to work with continuous action spaces. I've spent some time attempting to do this, but keep hitting dead ends. Will keep working on it
2) Look into hyperparameters and see if that alleviates the occasional 'forgetting' that I've seen occur when training
3) Add entropy bonus to the loss function
4) Keep testing the agent on various OpenAI Gym environments to see that it works and adress as necessary
