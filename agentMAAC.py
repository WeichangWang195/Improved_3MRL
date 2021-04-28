from base import BaseAgent
import numpy as np
from utils.rl_utils import Transition
import random
import itertools
import math
import pickle
import math
import random
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition = namedtuple('Transition', ['state', 'obs', 'prob', 'v_eval', 'action', 'r_in',
#                                        'value_return', 'extrinsic_critic_advantage',
#                                        'baseline_value'])


class Agent(BaseAgent):

    def __init__(self, idx, agent_cfg):

        """Declare agent variables."""
        # self.num_UAV = num_UAV
        self.idx = idx
        self.grid_size = agent_cfg['grid_size']
        self.alpha = 0.00
        self.gamma = 0.99
        self.mode = agent_cfg['mode']
        self.action_space = [0, 1, 2, 3]

        self.action = None
        self.action_scores = None
        self.state = None
        self.actor_obs = None
        self.state_input = None

        self.goal = None
        self.arrived = None
        self.actor = agent_cfg['actor']
        self.action_to_idx = dict(zip(self.action_space, range(len(self.action_space))))
        self.V = None

    def agent_init(self, mode="Train"):
        """
        :param load_idx: load the theta parameters. Used in final test
        :param mode: For agent, "Train" mode mean need to update policy used in update theta. "Test" mode shows the
        algorithm, used in final test.
        :return: None
        """
        # reward leads by the difference of the algorithms.
        self.mode = mode

    def agent_start(self, state, state_input, actor_obs, eps, goal):
        """
        :param state: tuple, the position of the agent, (x, y)
        :param obs: numpy matrix, the neighbor observation
        :param goal: tuple, set the goal of the agent, (d_x, d_y)
        :param r_diff: reward results from the difference in policy
        :return: action: choose from [0, 1, 2, 3]
        """
        # x,y=state
        self.state = state
        self.goal = goal
        self.state_input = state_input
        self.actor_obs = actor_obs
        self.arrived = False

        self.action = self._chooseAction(state_input, actor_obs)

        return self.action

    def agent_step(self, state_new, state_input, actor_obs):
        """
        :param state_new: tuple, the position of the agent, (x, y)
        :param obs_new: numpy matrix, the neighbor observation
        :param r_in: list of list, intrinsic reward
        :param r_ex: float, extrinsic reward
        :param extrinsic_critic_advantage: advantage of extrinsic critic
        :return: action: choose from [0, 1, 2, 3]
        """
        if self.state == self.goal and (not self.arrived):
            self.arrived = True
        # self.episode_trajectory.append([self.state_input,
        #                                 self.obs_input, self.action, r_ex, state_input_new, obs_input_new])
        action_new = self._chooseAction(state_input, actor_obs)
        self.state = state_new
        self.action = action_new

        return self.action

    def agent_end(self):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        if not self.arrived:
            self.arrived = True

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        
        pass

    def _chooseAction(self, state_input, actor_obs):
        """
        :param relative_state: relative positions of the UAVs compared to corresponding destinations;
                tuple(tuple(x, y) * numUAV)
        :param view: the view of each UAV; tuple(observations * numUAV)
        :return: actions, tuple(string * num_UAV); coord, tuple( tuple(0or1, 0or1) * num_UAV)
        """
        if state_input is None:
            return np.random.choice(self.action_space)

        probs = self.actor(state_input.to(device), actor_obs.to(device)).squeeze(0)
        if self.mode == "Train":
            m = Categorical(probs)
            uav_action = m.sample()
        else:
            uav_action = torch.argmax(probs)

        return uav_action.item()





