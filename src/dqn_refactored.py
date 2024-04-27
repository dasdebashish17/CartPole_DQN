"""
The reference code has been taken from the official pytorch website:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
The code has been refactored to have code in a more presentable manner. The code was executed on CUDA enabled system.
"""

import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample from the batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor ([[leftoexp, right@exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class CartPole:
    def __init__(self):
        # initialize parameters
        self.init_params()
        # initialize the cartpole environment
        self.init_env()
        # initialize device based on availability of CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize policy and target network
        self.init_neural_network_models()
        # initialize replay buffer
        self.init_replay_buffer()
        # initialize list to contain returns for each episode
        self.return_list = []
        # initialize the number of episodes
        self.num_episodes = 0

    def init_replay_buffer(self):
        self.memory = ReplayMemory(10000)

    def init_params(self):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS START is the starting value of epsilon # EPS END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the Adamw optimizer
        self.BATCH_SIZE = 1000
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.0001
        self.EPS_DECAY = 2000
        self.TAU = 0.005
        self.LR = 1e-4

    def init_steps_done(self):
        self.steps_done = 0

    def init_env(self):
        # Initialize the gym environment
        self.env = gym.make("CartPole-v1", render_mode="human")
        state = self.reset_env()
        return state

    def init_neural_network_models(self):
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

    def reset_env(self):
        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n

        # Get the number of state observations
        state, info = self.env.reset()
        self.n_observations = len(state)
        return state

    def select_action_from_policy(self, state):
        return self.policy_net(state).max(1).indices.view(1, 1)

    def select_action_from_target(self, state):
        return self.target_net(state).max(1).indices.view(1, 1)

    def select_epsilon_greedy_action(self, state):
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                             math.exp(-1. * self.steps_done / self.EPS_DECAY)

        self.steps_done += 1
        sample = random.random()
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.select_action_from_policy(state)

        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def update_policy_from_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, model_file='cartpole_target_net.model'):
        # Save the hyperparameter and the reward
        hyperparam_dict = {'BATCH_SIZE': self.BATCH_SIZE,
                           'GAMMA': self.GAMMA,
                           'EPS_START': self.EPS_START,
                           'EPS_END': self.EPS_END,
                           'EPS_DECAY': self.EPS_DECAY,
                           'TAU': self.TAU,
                           'LR': self.LR,
                           'NUM_EPISODE': self.num_episodes,
                           'REWARD_SEQUENCE': self.return_list}

        with open('parameter.json', 'w') as f:
            json.dump(hyperparam_dict, f, indent=6)

        torch.save({
            'model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_file)

    def load_model(self, model_file='cartpole_target_net.model'):
        checkpoint = torch.load(model_file)
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # either eval() or train() should be sufficient
        self.target_net.eval()
        self.target_net.train()

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_(t+1)) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or o in case the state was final.

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train(self):
        # Initialize the environment and get its state
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        episode_return = 0
        for t in count():
            action = self.select_epsilon_greedy_action(state)
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            episode_return += reward
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated
            self.env.render()

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ' - τθ + (1)θ'
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

            if done:
                print(f"steps count: {t+1}, with total reward: {episode_return}")
                self.return_list.append({'return': episode_return, 'epsilon': self.eps_threshold})
                break

    def play(self):
        # Initialize the environment and get its state
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        episode_return = 0
        for t in count():
            action = self.select_action_from_target(state)
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            episode_return += reward
            done = terminated or truncated
            self.env.render()

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                print(f"steps count: {t+1}, with total reward: {episode_return}")
                break

        return episode_return




class Experiment:
    def __init__(self):
        self.cartpole = CartPole()
        self.init_episode_count()


    def init_episode_count(self):
        if torch.cuda.is_available():
            self.num_episodes = 600
        else:
            self.num_episodes = 600


    def set_model_file_name(self, model_file):
        self.model_file_name = model_file


    def simulate_and_train(self):
        self.cartpole.init_steps_done()

        for i_episode in range(self.num_episodes):
            self.cartpole.train()

        print('Complete')
        self.cartpole.save_model(self.model_file_name)
        print(f"Model saved into file: {self.model_file_name}")


    def load_and_play_model(self, num_experiment=1000):
        returns_per_episode_list = []
        self.cartpole.load_model(self.model_file_name)
        # play game for num_experiment times and capture the reward for each play
        for episode in range(num_experiment):
            return_per_episode = self.cartpole.play()
            returns_per_episode_list.append(return_per_episode)
        with open("play_mode_statistics.log", 'w') as f:
            f.write("*********************\n")
            f.write(f"{num_experiment} EPISODES EXECUTED: \n")
            f.write("**********************\n")
            f.write("Returns observed per episode:\n")
            data = [f"Episode (episode): {returns_per_episode_list[episode]}" for episode in range(num_experiment)]
            f.write("\n".join(data))
            f.write("\n********************")
            f.write(f"\nAverage return: {np.mean(returns_per_episode_list)}")
            f.write("\n********************")


if __name__ == "__main__":
    TRAIN = 0
    PLAY = 1
    mode = PLAY

    experiment = Experiment()
    experiment.set_model_file_name("cartpole_dqn_model.mdl")
    if mode == TRAIN:
        experiment.simulate_and_train()
    elif mode == PLAY:
        experiment.load_and_play_model(10)