from base import BaseEnvironment
import numpy as np
import math
import random

import itertools
import pickle

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

angle_mode = 24
velocity_mode = 3
velocity_scale = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment(BaseEnvironment):

    def __init__(self, cfg):
        """Declare environment variables."""
        self.num_UAV = cfg['num_UAV']
        self.num_dest = cfg['num_dest']
        self.grid_size = cfg['grid_size']
        self.obs_type = cfg['obs_type']
        self.state = None
        self.terminal = None
        self.terminal_idx = None
        self.view_length = cfg['view_length']
        self.critic_view_length = cfg['critic_view_length']
        self.arrived = None

        self.reward_terminal = cfg['reward_terminal']
        self.safe_distance_control_param = cfg['safe_dist_control']
        self.destination_control_param = cfg['dest_control']
        self.step_control = cfg['step_control']
        self.safe_dist = cfg['safe_dist']
        self.last_action = []
        self.last_reward = []

        self.action_space = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def env_init(self, maze):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """

    def env_start(self, maze, start, terminal, terminal_idx):
        """
        :param maze: Initial maze
        :param start: Initial state, tuple of tuple, eg.((0, 0), (0, 29), (29, 0), (29, 29))
        :param goal: Destinations, tuple of tuple, eg. ((29, 29), (29, 0), (0, 29), (0, 0))
        :return: self.state: list of state, eg, ((0, 0), (0, 29), (29, 0), (29, 29))
        state_observation: list of local observation, list of torch.
        """

        self.state = start
        self.terminal = terminal
        self.terminal_idx = terminal_idx
        self.arrived = [False] * self.num_UAV
        if len(self.state) != self.num_UAV or len(self.terminal) != self.num_UAV:
            print("Error Input: number of UAV")

        self.last_action = [0] * self.num_UAV
        self.last_reward = [0.0] * self.num_UAV

        state_input = self.get_state_input(self.state, self.obs_type == 1)

        actor_obs_list = self.generate_agent_obs_CNN(self.state, self.obs_type == 1)

        critic_obs_list, neighbors, min_dist_list = self.generate_critic_obs_CNN(self.state, self.obs_type == 1)
        # collision_list = [d < self.safe_dist for d in min_dist_list]

        return self.state, state_input, actor_obs_list, critic_obs_list, neighbors, min_dist_list

    def env_step(self, action):
        """
        :param action: list of action
        :return:
        state: tuple of tuple, state of all agents
        observation: list of torch, observation of all agents, eg.[torch.from_numpy(state_image)] * self.num_UAV
        r_in: numpy with size (numUAV * 2), intrinsic reward of all agents, 2 terms in it, the first term relate to
        distance toward destination, next term related to distance towards intruders.
        r_ex: float, extrinsic reward
        extrinsic_critic_advantage: float, advantage of external_critic
        arrive: bool, if all agent arrived
        """
        if len(action) != self.num_UAV:
            print("Error:  number of UAV")
        testState = []

        for aa in range(self.num_UAV):
            testState.append(tuple(map(sum, zip(self.state[aa], self.action_space[action[aa]]))))

        next_state = []

        reward = [0.0] * self.num_UAV
        for uav in range(self.num_UAV):
            if 0 <= testState[uav][0] < self.grid_size and 0 <= testState[uav][1]\
                    < self.grid_size and self.state[uav] != self.terminal[uav]:
                next_state.append(testState[uav])
            else:
                next_state.append(self.state[uav])

        next_state = tuple(next_state)
        self.state = next_state

        for uav in range(self.num_UAV):
            if self.state[uav] == self.terminal[uav] and not self.arrived[uav]:
                self.arrived[uav] = True
                reward[uav] = self.reward_terminal
            elif self.state[uav] == self.terminal[uav]:
                reward[uav] = 0.0
            else:
                dist_to_dest = math.sqrt(sum(map(lambda x, y: (x - y) ** 2, self.state[uav], self.terminal[uav])))
                reward[uav] += self.step_control + self.destination_control_param * dist_to_dest
        if all(self.arrived):
            return self.state, [None] * self.num_UAV, [None] * self.num_UAV, [None] * self.num_UAV,\
                   torch.zeros(self.num_UAV, self.num_UAV), [65535] * self.num_UAV, tuple(reward), True

        state_input = self.get_state_input(self.state, self.obs_type == 1)

        actor_obs_list = self.generate_agent_obs_CNN(self.state, self.obs_type == 1)

        critic_obs_list, neighbors, min_dist_list = self.generate_critic_obs_CNN(self.state, self.obs_type == 1)
        collision_list = [d < self.safe_dist for d in min_dist_list]

        for uav in range(self.num_UAV):
            reward[uav] += self.safe_distance_control_param if collision_list[uav] else 0

        # for uav in range(self.num_UAV):
        #     reward[uav] = uav
        self.last_reward = reward
        self.last_action = [a + 1 for a in action]

        return self.state, state_input, actor_obs_list, critic_obs_list, \
               neighbors, min_dist_list, tuple(reward), False

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        if in_message == "return":
            return self.state

    def update_start_goal(self, start, goal):
        self.state = start
        self.terminal = goal

    def get_state_input(self, state, meta=False):
        state_input_list = []
        for ii in range(self.num_UAV):
            if state[ii] == self.terminal[ii]:
                state_input_list.append(None)
                continue
            else:
                state_input = torch.zeros(self.num_dest + 4 * self.grid_size - 2)
                relative_x = self.terminal[ii][0] - state[ii][0] + self.grid_size - 1
                relative_y = self.terminal[ii][1] - state[ii][1] + self.grid_size - 1
                state_input[self.terminal_idx[ii]] = 1
                state_input[self.num_dest + relative_x] = 1
                state_input[self.num_dest + 2 * self.grid_size - 1 + relative_y] = 1
                action_reward = torch.zeros(len(self.action_space) + 2)
                action_reward[self.last_action[ii]] = 1
                action_reward[-1] = self.last_reward[ii]
                if meta:
                    state_input_list.append(torch.cat((state_input, action_reward)).view(-1).unsqueeze(0))
                else:
                    state_input_list.append(state_input.view(-1).unsqueeze(0))
        return state_input_list

    def generate_agent_obs_binary(self, state):
        actor_obs_list = []
        for uav in range(self.num_UAV):
            if state[uav] == self.terminal[uav]:
                actor_obs_list.append(None)
                continue
            else:
                actor_obs = torch.zeros(3, 2 * self.view_length - 1, 2 * self.view_length - 1)
                for intruder in range(self.num_UAV):
                    if state[intruder] == self.terminal[intruder] or uav == intruder:
                        continue
                    relative_x = state[uav][0] - state[intruder][0]
                    relative_y = state[uav][1] - state[intruder][1]
                    if abs(relative_x) < self.view_length and abs(relative_y) < self.view_length:
                        actor_obs[0, relative_x + self.view_length - 1,
                                  relative_y + self.view_length - 1] += 1
                        if random.uniform(0, 1) < 1 / actor_obs[0, relative_x + self.view_length - 1,
                                  relative_y + self.view_length - 1]:
                            actor_obs[2, relative_x + self.view_length - 1,
                                      relative_y + self.view_length - 1] += self.terminal_idx[intruder]
                actor_obs_list.append(actor_obs.view(-1).unsqueeze(0))
        return actor_obs_list

    def generate_agent_obs_CNN(self, state, meta=False):
        actor_obs_list = []
        for uav in range(self.num_UAV):
            actor_obs = torch.zeros(5, 2 * self.view_length - 1, 2 * self.view_length - 1)
            if state[uav] == self.terminal[uav]:
                actor_obs_list.append(None)
                continue
            for intruder in range(self.num_UAV):
                if state[intruder] == self.terminal[intruder] or uav == intruder:
                    continue
                relative_x = state[uav][0] - state[intruder][0]
                relative_y = state[uav][1] - state[intruder][1]
                if abs(relative_x) < self.view_length and abs(relative_y) < self.view_length:
                    relative_x_idx = relative_x + self.view_length - 1
                    relative_y_idx = relative_y + self.view_length - 1
                    actor_obs[0, relative_x_idx, relative_y_idx] += 1
                    if random.uniform(0, 1) < 1 / actor_obs[0, relative_x_idx, relative_y_idx]:
                        actor_obs[1, relative_x_idx, relative_y_idx] = self.terminal_idx[intruder] + 1
                        des_dist = np.linalg.norm(np.array(self.terminal[intruder]) - np.array(state[intruder])) + 1
                        actor_obs[2, relative_x_idx, relative_y_idx] = des_dist/self.grid_size
                        actor_obs[3, relative_x_idx, relative_y_idx] = self.last_action[intruder]
                        actor_obs[4, relative_x_idx, relative_y_idx] = self.last_reward[intruder]
            if meta:
                actor_obs_list.append(actor_obs.unsqueeze(0))
            else:
                actor_obs_list.append(actor_obs[:3, :, :].unsqueeze(0))
        return actor_obs_list

    def generate_critic_obs_CNN(self, state, meta=False):
        critic_obs_list = []
        min_dist_list = []
        neighbors = torch.eye(self.num_UAV)
        for uav in range(self.num_UAV):
            critic_obs = torch.zeros(5, 2 * self.critic_view_length - 1, 2 * self.critic_view_length - 1)
            min_dist = self.grid_size * 2
            if state[uav] == self.terminal[uav]:
                critic_obs_list.append(None)
                neighbors[uav, uav] = 0
                min_dist_list.append(min_dist)
                continue
            for intruder in range(self.num_UAV):
                if state[intruder] == self.terminal[intruder] or uav == intruder:
                    continue
                relative_x = state[uav][0] - state[intruder][0]
                relative_y = state[uav][1] - state[intruder][1]
                if abs(relative_x) < self.critic_view_length and abs(relative_y) < self.critic_view_length:
                    relative_x_idx = relative_x + self.critic_view_length - 1
                    relative_y_idx = relative_y + self.critic_view_length - 1
                    neighbors[uav, intruder] = 1
                    critic_obs[0, relative_x_idx, relative_y_idx] += 1
                    if random.uniform(0, 1) < 1 / critic_obs[0, relative_x_idx, relative_y_idx]:
                        critic_obs[1, relative_x_idx, relative_y_idx] = self.terminal_idx[intruder] + 1
                        des_dist = np.linalg.norm(np.array(self.terminal[intruder]) - np.array(state[intruder])) + 1
                        critic_obs[2, relative_x_idx, relative_y_idx] = des_dist/self.grid_size
                        critic_obs[3, relative_x_idx, relative_y_idx] = self.last_action[intruder]
                        critic_obs[4, relative_x_idx, relative_y_idx] = self.last_reward[intruder]
                    if math.sqrt(relative_x ** 2 + relative_y ** 2) < min_dist:
                        min_dist = math.sqrt(relative_x ** 2 + relative_y ** 2)

            if meta:
                critic_obs_list.append(critic_obs.unsqueeze(0))
            else:
                critic_obs_list.append(critic_obs[:3, :, :].unsqueeze(0))
            min_dist_list.append(min_dist)

        return critic_obs_list, neighbors, min_dist_list




