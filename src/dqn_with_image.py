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

import cv2



# STATES
INIT = 0
NORMAL = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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






class CartPoleImage:
    """
    Class to transform the image data rendered from cartpole environment
    We reduce the image to black and white with image size as 240x160.
    For extracting meaningful information (direction, speed and acceleration) from the cartpole image,
    we use a stack of 4 such images.
    For every new frame of data captured, we flush the last frame and add the new frame to the stack of 4
    """
    def __init__(self, COLS=240, ROWS=160):
        self.COLS = COLS
        self.ROWS = ROWS


    def set_cartpole_rendered_image(self, rendered_image:np.ndarray):
        """
        Set the image as provided by rendered cartpole environment
        :param rendered_image: image data as obtained from rendering cartpole environment
        :return:
        """
        self.rendered_image = rendered_image


    def reduce_image(self):
        """
        Reduce the size of image.
        Also, change to black and white image for ease of processing
        :return:
        """
        # convert the image to greyscale
        img_rgb = cv2.cvtColor(self.rendered_image, cv2.COLOR_RGB2GRAY)
        # reduce the size of image
        self.reduced_image = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        # convert to full black and white
        self.reduced_image[self.reduced_image < 255] = 0


    def get_reduced_image(self)->np.ndarray:
        """
        :return: Returns the reduced image
        """
        return self.reduced_image

"""
if __name__ == "__main__":
    obj = CartPoleImage(COLS=240, ROWS=160)
    gym_env = gym.make("CartPole-v1", render_mode='rgb_array')
    gym_env.reset()
    img = gym_env.render()
    obj.set_cartpole_rendered_image(img)
    obj.reduce_image()
    obj.get_reduced_image()
"""






class ImageStack:
    """
    Stack of images to be used for training
    """
    def __init__(self):
        self.STACK_SIZE = 4
        self.COLS = 240
        self.ROWS = 160
        self.STATE = INIT
        self.cartpole_img_obj = CartPoleImage(self.COLS, self.ROWS)
        self.image_stack = np.zeros((self.STACK_SIZE, self.ROWS, self.COLS))


    def reset(self):
        """
        Resets the state flag
        :return:
        """
        self.STATE = INIT


    def add_rendered_image(self, rendered_image: np.ndarray):
        """
        Add the rendered image to the images stack.
        :param rendered_image: rendered image
        :return:
        """
        # add the rendered image
        self.cartpole_img_obj.set_cartpole_rendered_image(rendered_image)
        # resize the rendered image
        self.cartpole_img_obj.reduce_image()
        # during init, add same image to each index of the stack
        if self.STATE == INIT:
            for _ in range(self.STACK_SIZE):
                self.image_stack = np.roll(self.image_stack, 1, axis=0)
                self.image_stack[0, :, :] = self.cartpole_img_obj.get_reduced_image()
            self.STATE = NORMAL
        else:
            self.image_stack = np.roll(self.image_stack, 1, axis=0)
            self.image_stack[0, :, :] = self.cartpole_img_obj.get_reduced_image()


    def get_image_stack(self):
        """
        returns the stack of processed images
        :return:
        """
        return self.image_stack


"""
if __name__ == "__main__":
    gym_env = gym.make("CartPole-v1", render_mode='rgb_array')
    gym_env.reset()
    img = gym_env.render()
    obj = ImageStack()
    obj.reset()
    obj.add_rendered_image(img)
    img_stack = obj.get_image_stack()
    print(img_stack.shape)
"""









class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.conv_layer1 = nn.Conv2d(4, 8, (5, 5), stride=(3, 3))
        self.conv_layer2 = nn.Conv2d(8, 8, (5, 5), stride=(2, 2))
        self.conv_layer3 = nn.Conv2d(8, 8, (5, 5), stride=(1, 1))

        self.layer1 = nn.Linear(5440, 128) # x4.shape[1] = 5440
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor ([[leftoexp, right@exp]...]).
    def forward(self, x):
        x1 = F.relu(self.conv_layer1(x))
        x2 = F.relu(self.conv_layer2(x1))
        x3 = F.relu(self.conv_layer3(x2))

        x4 = torch.flatten(x3)
        if x4.size()[0] == 5440:
            x5 = torch.tensor(x4, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            dim0 = int(x4.size()[0] / 5440)
            x5 = torch.tensor(x4.reshape(dim0, 5440), dtype=torch.float32, device=device)

        x6 = F.relu(self.layer1(x5))
        x7 = F.relu(self.layer2(x6))
        return self.layer3(x7)

"""
if __name__ == "__main__":
    gym_env = gym.make("CartPole-v1", render_mode='rgb_array')
    gym_env.reset()
    img = gym_env.render()
    obj = ImageStack()
    obj.reset()
    obj.add_rendered_image(img)
    img_stack = obj.get_image_stack()
    print(img_stack.shape)

    policy_net = DQN(gym_env.action_space.n).to(device)
    states = torch.tensor(img_stack, dtype=torch.float32, device=device).unsqueeze(0)
    policy_net(states).max(1).indices.view(1, 1)
    print("END")
"""



class CartPole:
    def __init__(self):
        # initialize the image stack object
        self.img_stack_obj = ImageStack()
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



    def reset(self):
        state, info = self.env.reset()
        self.img_stack_obj.reset()
        image_stack = self.render()
        return image_stack, info


    def render(self):
        rendered_image = self.env.render()
        self.img_stack_obj.add_rendered_image(rendered_image)
        return self.img_stack_obj.get_image_stack()


    def step(self, action_item):
        observation, reward, terminated, truncated, _ = self.env.step(action_item)
        image_stack = self.render()
        return image_stack, reward, terminated, truncated, _


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
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n
        # Get the number of state observations
        self.env.reset()

    def init_neural_network_models(self):
        self.policy_net = DQN(self.n_actions).to(self.device)
        self.target_net = DQN(self.n_actions).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)


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
        state, info = self.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        episode_return = 0
        for t in count():
            action = self.select_epsilon_greedy_action(state)
            observation, reward, terminated, truncated, _ = self.step(action.item())
            episode_return += reward
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated
            #self.env.render()

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
                #if key not in target_net_state_dict:
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
    mode = TRAIN

    experiment = Experiment()
    experiment.set_model_file_name("cartpole_dqn_model.mdl")
    if mode == TRAIN:
        experiment.simulate_and_train()
    elif mode == PLAY:
        experiment.load_and_play_model(10)