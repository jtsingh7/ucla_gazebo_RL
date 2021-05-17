import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from utils import init_weights

class Memory: 
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states) 
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start] #all batches created

        memory = np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches
        return memory

    def store_memory(self, state, action, probs, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []






class ActorNN(nn.Module):
    def __init__(self, n_actions, input_dims, alpha_A, fc1_dims, fc2_dims, checkpt_dir='Networks', std=0):
        super(ActorNN, self).__init__()
        self.checkpt_file = os.path.join(checkpt_dir, 'Actor_PPO')
        
        # Architecture to be adjusted if needed
        self.actor = nn.Sequential(
                     nn.Linear(*input_dims, fc1_dims), 
                     nn.ReLU(),
                     nn.Linear(fc1_dims, fc2_dims),
                     nn.ReLU(),
                     nn.Linear(fc2_dims, n_actions),
                     nn.Softmax(dim=-1)
                     )
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha_A)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(init_weights)

    def forward(self, state):   
        policy_dist = self.actor(state)
        policy_dist = Categorical(policy_dist)
        return policy_dist

    def save_checkpt(self):
        T.save(self.state_dict(), self.checkpt_file)

    def load_checkpt(self):
        self.load_state_dict(T.load(self.checkpt_file))







class CriticNN(nn.Module):
    def __init__(self, input_dims, alpha_C, fc1_dims, fc2_dims, checkpt_dir='Networks'):
        super(CriticNN, self).__init__()
        self.checkpt_file = os.path.join(checkpt_dir, 'Critic_PPO')

        # Architecture TBD
        self.critic = nn.Sequential(
                      nn.Linear(*input_dims, fc1_dims),
                      nn.ReLU(),
                      nn.Linear(fc1_dims, fc2_dims),
                      nn.ReLU(),
                      nn.Linear(fc2_dims, 1)
                      )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha_C)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.apply(init_weights)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpt(self):
        T.save(self.state_dict(), self.checkpt_file)

    def load_checkpt(self):
        self.load_state_dict(T.load(self.checkpt_file))





class Agent:
    def __init__(self, input_dims, n_actions, gamma, lambda_smooth, alpha_A, alpha_C, fc1_dims, fc2_dims, clip, batch_size, horizon, n_epochs, c1):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.lambda_smooth = lambda_smooth
        self.alpha_A = alpha_A
        self.alpha_C = alpha_C
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.clip = clip
        self.batch_size = batch_size
        self.horizon = horizon
        self.n_epochs = n_epochs
        self.c1 = c1

        self.actor = ActorNN(n_actions=self.n_actions, input_dims=self.input_dims, alpha_A=self.alpha_A, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.critic = CriticNN(input_dims=self.input_dims, alpha_C=self.alpha_C, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims)
        self.memory = Memory(batch_size)

    def remember(self, state, action, probs, values, reward, done):
        self.memory.store_memory(state, action, probs, values, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpt()
        self.critic.save_checkpt()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpt()
        self.critic.load_checkpt()

    def choose_action(self, obs):
        state = T.tensor([obs], dtype=T.float).to(self.actor.device)
        policy_dist = self.actor.forward(state)
        value_tensor = self.critic(state)
        action_tensor = policy_dist.sample()
        probs = policy_dist.log_prob(action_tensor).item()
        
        action = action_tensor.item()
        value = value_tensor.item()

        return action, probs, value


    def learn(self):
        for i in range(self.n_epochs):
            state_array, action_array, old_probs_array, value_array, reward_array, done_array, batches = self.memory.generate_batches()

            #Calculating generalized advantage estimates GAE (using equation 11 of PPO Paper)
            GAE = np.zeros(len(reward_array), dtype=np.float32)
            for t in range(len(reward_array)-1):
                adv = 0
                for tt in range(t, len(reward_array)-1):
                    adv = adv + ((self.gamma*self.lambda_smooth)**(tt-t+1))*(reward_array[tt] + self.gamma*value_array[tt+1]*(1-int(done_array[tt])) - value_array[tt])
                    disc_Debug = ((self.gamma*self.lambda_smooth)**(tt-t+1))
                GAE[t] = adv
            returns = GAE + value_array
            returns = T.tensor([returns]).to(self.critic.device)
            returns = T.squeeze(returns)
            GAE = T.tensor([GAE]).to(self.actor.device)
            GAE = T.squeeze(GAE)


            #Calculating surrogate loss function 
            for batch in batches:
                states = T.tensor(state_array[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_array[batch], dtype=T.float).to(self.actor.device)
                actions = T.tensor(action_array[batch], dtype=T.float).to(self.actor.device)

                policy_dist = self.actor(states)
                critic_value = self.critic(states)
          
                new_probs = policy_dist.log_prob(actions)
                prob_ratio = (new_probs-old_probs).exp() 
                
                surrloss1 = prob_ratio*GAE[batch]
                surrloss2 = T.clamp(prob_ratio, 1.0-self.clip, 1.0+self.clip)*GAE[batch]
                actor_loss_avg = -T.min(surrloss1, surrloss2).mean()                
                critic_loss_avg = ((returns[batch] - critic_value)**2).mean()

                total_loss = actor_loss_avg +self.c1*critic_loss_avg #NOTE: There is no entropy bonus in this implementation, yet...

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()