#PPO implemenation
#TO DO: Implement continouous action space functionality. Consider adding entropy bonus to the loss function.
#Despite being PPO, the agent does appear to occassionally 'forget' and exerperience catastriphic loss on the cartpole environment. May just by a hyperparameter adjustment
#Should work on most openAI gym discrete action environments, although tuning of hyperparameters is likely required
import numpy as np
import gym
from Agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    n_games = 1000

    #HyperParameters
    batch_size = 5
    n_epochs = 4
    gamma = 0.99
    lambda_smooth = 0.95
    alpha_A = 0.0005 
    alpha_C = 3*alpha_A
    fc1_dims = 256
    fc2_dims = 256
    clip = 0.2
    horizon = 20
    c1 = 0.5

    learn_iters = 0
    input_dims = env.observation_space.low.shape
    n_actions = env.action_space.n
    agent = Agent(input_dims=input_dims, n_actions=n_actions, gamma=gamma, lambda_smooth = lambda_smooth, alpha_A=alpha_A, alpha_C=alpha_C,
                  fc1_dims=fc1_dims, fc2_dims=fc2_dims, clip=clip, batch_size=batch_size, horizon=horizon, n_epochs=n_epochs, c1=c1)

    figure_file = 'plots/' + env_name + '.png' #will need to create plots folder before running
    best_score = env.reward_range[0]
    score_history = []
    avg_score = 0
    n_steps = 0
    render = False

    for game in range(n_games):

        done = False
        score = 0
        obs = env.reset()

        while not done:
            if render: env.render()
            action, probs, value = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(obs, action, probs, value, reward, done)
            if n_steps % horizon == 0: #learning at the end of each horizon
                agent.learn()
                learn_iters += 1
            obs = next_obs
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', game, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters, 'action', action)
    x = [game+1 for game in range(len(score_history))]
    plot_learning_curve(x,score_history,figure_file)