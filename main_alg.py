import copy
from collections import namedtuple
import itertools
from agentMAAC import Agent
from modules.AgentPolicyDQN import (agentCritic_CNN,
                                    agentActor_binary,
                                    agentCritic_binary,
                                    agentActor_CNN)
from modules.ReplayMemory import ReplayMemory
from utils.rl_utils import Transition
import pygame
import pickle
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_UNTIL = 500
EPS_START = 0.1
EPS_END = 0.1
EPS_DECAY = 500
BATCH_SIZE = 256

Record = namedtuple('Record', ['state', 'command', 'Reward'])

def list_to_torch(list, type):
    if type == "long":
        output = torch.zeros(len(list), dtype=torch.long)


class rl_alg:
    """
    Facilitates interaction between an agent and environment for
    reinforcement learning experiments.

    args:
        env_obj: an object that implements BaseEnvironment
        agent_obj: an object that implements BaseAgent
    """

    def __init__(self, env_obj, cfg, surface=None):

        self._environment = env_obj
        self.label = cfg['label']
        self.num_UAV = cfg['num_UAV']
        self.num_dest = cfg['num_dest']
        self.obs_type = cfg['obs_type']
        self.grid_size = cfg['grid_size']
        self.view_length = cfg['view_length']
        self.critic_view_length = cfg['critic_view_length']
        self.load_param = cfg['load_param']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.mode = cfg['mode']
        self.alg = cfg['alg']
        self.safe_dist = cfg['safe_dist']
        self.action_space = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        self.start_set = None
        self.start = None
        self.goal_set = None
        self.goal = None
        self.goal_idx = None
        self.state = None
        self.state_input = None
        self.actor_obs_list = None
        self.critic_obs_list = None
        self.neighbors = None
        self.collision_list = None

        # Network Creation
        # Homogeneous Agents
        net_cfg = {
            'hidden1': 128,
            'hidden2': 50,
            'init_w': 3e-3
        }
        if self.obs_type == 0:
            # obs include intruders' distribution, dest idx, direction, dist to dest. Actor & critic has different size.
            len_state = self.num_dest + 4 * self.grid_size - 2
            size_actor_obs = 2 * self.view_length - 1
            layers_agent_obs = 3
            size_critic_obs = self.critic_view_length * 2 - 1
            layers_critic_obs = 3
            self.actor = agentActor_CNN(layers_agent_obs, size_actor_obs, size_actor_obs, len_state,
                                        len(self.action_space)).to(device)
            self.critic = agentCritic_CNN(layers_critic_obs, size_critic_obs, size_critic_obs, len_state).to(device)

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-5)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

            self.actor.eval()
            self.critic.eval()
        else:
            # obs include intruders' distribution, dest idx, direction, dist to dest, last action, last reward.
            # Actor & critic has different size.
            # On Policy Only
            len_state = self.num_dest + 4 * self.grid_size - 2 + len(self.action_space) + 2
            size_actor_obs = 2 * self.view_length - 1
            layers_agent_obs = 5
            size_critic_obs = self.critic_view_length * 2 - 1
            layers_critic_obs = 5
            self.actor = agentActor_CNN(layers_agent_obs, size_actor_obs, size_actor_obs, len_state,
                                        len(self.action_space)).to(device)
            self.critic = agentCritic_CNN(layers_critic_obs, size_critic_obs, size_critic_obs, len_state).to(device)

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-5)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

            self.actor.eval()
            self.critic.eval()

        self.replay_memory = ReplayMemory(10000)

        self._agent = []
        agent_cfg = {
            'mode': self.mode,
            'grid_size': self.grid_size,
            'actor': self.actor,
        }
        for ii in range(self.num_UAV):
            self._agent.append(Agent(ii, agent_cfg))

        # Initialize useful statistics
        self.actor_obs = None
        self.critic_obs = None

        # Records
        self._total_reward = None
        self._total_discounted_reward = None
        self._num_ep_steps = None
        self._num_episodes = 0
        self._num_updates = None
        self._min_intr_dist = None
        self._num_ep_steps_record = []
        self._total_reward_record = []
        self._total_discounted_reward_record = []
        self._min_intr_dist_record = []

        # the most recent action taken by the agent
        self._last_action = None

        ###############################
        # Attributes of Env
        self.target_update = 10
        self.maze = None

        ###############################
        # Pygame Setting
        self.surface = surface
        self.pause_time = 0.04
        self.continue_game = True
        self.bg_color = pygame.Color('black')
        self.u_color = (pygame.Color('blue'),
                        pygame.Color('aquamarine2'),
                        pygame.Color('brown'),
                        pygame.Color('darkgoldenrod1'),
                        pygame.Color('chartreuse'),
                        pygame.Color('chocolate1'),
                        pygame.Color('darkseagreen4'),
                        pygame.Color('darkorchid'),
                        pygame.Color('darkorange'),
                        pygame.Color('darkslategray'),
                        pygame.Color('deeppink1'),
                        pygame.Color('firebrick'),
                        pygame.Color('gray23'),
                        pygame.Color('gray23'),)
        self.pause_time = 0.04
        self.close_clicked = False
        self.continue_game = True
        self.target_update = 10
        self.normal_color = pygame.Color('white')
        self.wall_color = pygame.Color('gray')
        self.w = 10
        self.margin = 0
        self.maze = [[0]*100 for n in range(100)]

        # if self.mode == "Test":
        #     with open('Data/start_dest_list.pickle', "rb") as handle:
        #         self.start_list, self.dest_list = pickle.load(handle)

    def rl_init(self, load_param=None):
        """
        Load parameters of network if mode is "Test"
        :return: None
        """
        self.load_param = load_param
        if self.load_param is not None and self.mode == "Test":
            if self.alg == 0 or self.alg == 1:
                self.actor.load_state_dict(torch.load('Data/MAHL_v1_' + self.load_param+'_trainedCMD_3'))
            self.critic.load_state_dict(torch.load('Data/MAHL_v1_' + self.load_param+'_trainedDCD_3'))

        for ii in range(self.num_UAV):
            self._agent[ii].agent_init()
        self._environment.env_init(self.maze)

    def rl_start(self, mode="Train"):
        """
        First Step in the episode
        Returns: state: list of position [x, y]
                 action: list of int
        """
        # self.start_set = (((29, 29), (29, 28), (28, 29), (28, 28)),
        #                   ((0, 29), (0, 28), (1, 29), (1, 28)),
        #                   ((29, 0), (28, 0), (29, 1), (28, 1)),
        #                   ((0, 0), (1, 0), (0, 1), (1, 1)))
        self.start_set = (((29, 29), (29, 28), (28, 29), (28, 28)),
                          ((0, 29), (0, 28), (1, 29), (1, 28)),
                          ((29, 0), (28, 0), (29, 1), (28, 1)),
                          ((0, 0), (1, 0), (0, 1), (1, 1)))
        self.goal_set = ((0, 0), (29, 0), (0, 29), (29, 29))

        # self.start = ((31, 49), (42, 42), (49, 31), (49, 18), (42, 7), (31, 0), (18, 0), (7, 7), (0, 18), (0, 31), (7, 42), (18, 49))
        # self.goal = ((18, 0), (7, 7), (0, 18), (0, 31), (7, 42), (18, 49), (31, 49), (42, 42), (49, 31), (49, 18), (42, 7), (31, 0))

        # self.start_set = (((0, 8), (0, 9), (1, 8), (1, 9)),
        #                   ((0, 18), (0, 19), (1, 18), (1, 19)),
        #                   ((29, 8), (29, 9), (28, 8), (28, 9)),
        #                   ((29, 18), (29, 19), (28, 18), (28, 19)),
        #                   ((8, 28), (9, 29), (8, 28), (9, 29)),
        #                   ((18, 28), (19, 29), (18, 28), (19, 29)),
        #                   ((18, 0), (19, 0), (18, 1), (19, 1)),
        #                   ((8, 0), (9, 0), (8, 1), (9, 1)))
        # self.goal_set = ((29, 19), (29, 9), (0, 19), (0, 9), (19, 0), (9, 0), (9, 29), (19, 29))



        self.start = tuple([random.choice(start_set) for start_set in self.start_set])
        self.goal = self.goal_set
        self.goal_idx = range(self.num_UAV)

        self._num_ep_steps = 1
        self._total_reward = 0.0
        self._total_discounted_reward = 0.0
        self._min_intr_dist = self.grid_size * 2.0

        state, state_input, actor_obs_list, critic_obs_list, neighbors, min_dist_list = \
            self._environment.env_start(self.maze,
                                        self.start[0:self.num_UAV],
                                        self.goal[0:self.num_UAV],
                                        self.goal_idx[0:self.num_UAV])

        self.state = state
        self.state_input = state_input
        self.actor_obs_list = actor_obs_list
        self.critic_obs_list = critic_obs_list
        self.neighbors = neighbors
        self._min_intr_dist = min(self._min_intr_dist, min(min_dist_list))
        self.state = self.start[0:self.num_UAV]
        self._last_action = []
        eps = 1.0 if self._num_episodes <= RANDOM_UNTIL else EPS_END + (EPS_START - EPS_END) * \
                                                           math.exp(-1. * (self._num_episodes-RANDOM_UNTIL) / EPS_DECAY)
        for ii in range(self.num_UAV):
            action = self._agent[ii].agent_start(state[ii], self.state_input[ii],
                                                 actor_obs_list[ii], eps, self.goal[ii])
            # self._last_action.append(ii)
            self._last_action.append(action)

        # string = "("+str(state[0][0])+"," + str(state[0][1]) + ") (" \
        #          + str(state[1][0])+"," + str(state[1][1]) + ") (" \
        #          +str(state[2][0]) + "," + str(state[2][1]) + ") (" \
        #          +str(state[3][0])+"," + str(state[3][1]) + ")"
        # print(string)
        # self._last_action = self._agent.agent_start(state, self.goal[0:self.num_UAV])

        return state, self._last_action

    def rl_step(self):
        """Takes a step in the RLGlue experiment.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """
        new_state, new_state_input, new_actor_obs_list, new_critic_obs_list, new_neighbors, new_min_dist_list, reward, terminal = self._environment.env_step(
            self._last_action)

        self._total_discounted_reward += (self.gamma ** (self._num_ep_steps - 1)) * sum(reward)
        self._total_reward += sum(reward)
        self.replay_memory.push(Transition(self.state_input,
                                           self.actor_obs_list,
                                           self.critic_obs_list,
                                           self.neighbors,
                                           torch.tensor(self._last_action, dtype=torch.long).unsqueeze(1),
                                           torch.tensor(reward, dtype=torch.float64).unsqueeze(1),
                                           new_state_input,
                                           new_actor_obs_list,
                                           new_critic_obs_list, 0.0))
        if not terminal:
            self._num_ep_steps += 1
            self._last_action = []
            for ii in range(self.num_UAV):
                action = self._agent[ii].agent_step(new_state[ii], new_state_input[ii], new_actor_obs_list[ii])
                self._last_action.append(action)
                # self._last_action.append(ii)
        else:
            for ii in range(self.num_UAV):
                self._agent[ii].agent_end()

        self.state = new_state
        self.state_input = new_state_input
        self.actor_obs_list = new_actor_obs_list
        self.critic_obs_list = new_critic_obs_list
        self.neighbors = new_neighbors
        self._min_intr_dist = min(self._min_intr_dist, min(new_min_dist_list))

        return reward, self.state, self._last_action, terminal

    def rl_episode(self, max_steps_this_episode=100, idx=0):
        """
        Convenience function to run an episode.

        Args:
            max_steps_this_episode (Int): Max number of steps in this episode.
                A value of 0 will result in the episode running until
                completion.

        returns:
            Boolean: True if the episode terminated within
                max_steps_this_episode steps, else False
        """
        terminal = False
        self.rl_start(self.mode)
        self.display_on_surface(self.start)

        while not terminal and ((max_steps_this_episode <= 0) or
                                (self._num_ep_steps < max_steps_this_episode))and not self.close_clicked:
            if self.surface is not None:
                self.handle_event()
            _, state, _, terminal = self.rl_step()
            # self.policy_update(policy_update=self._num_episodes > 5)
            self.display_on_surface(state)

        # for ii in range(self.num_UAV):
        #     self.replay_memory.push(self._agent[ii].get_trajectory())
        self.policy_update(True)
        self._last_action = None
        self._num_ep_steps_record.append(self._num_ep_steps)
        self._total_reward_record.append(self._total_reward)
        self._total_discounted_reward_record.append(self._total_discounted_reward)

        self._num_episodes += 1
        if self._num_episodes % 1000 == 0 and self.mode == "Train":
            idx = self._num_episodes // 1000
            self.save_parameters(idx)
            with open('Data/eps_records_v1_' + self.label, 'wb') as f:
                np.savez(f, num_ep_steps=self._num_ep_steps_record,
                         total_reward=self._total_reward_record,
                         discounted_reward=self._total_discounted_reward_record)
        # if self._num_episodes == 10000 and self.mode == "Test":
        #     print("Total cost in this episode was %.2f" % (sum(self._total_reward_record)/10000))

        return terminal

    def policy_update(self, policy_update=False):
        """
        :param policy_update:
        :return: None
        """
        if policy_update:

            if self.alg == 0:
                transitions = self.replay_memory.return_all()
                data_size = len(transitions)
                batch = Transition(*zip(*transitions))
                state_input_list = list(itertools.chain.from_iterable(batch.state_input))  # flatten list
                non_final_state_mask = torch.tensor(tuple(map(lambda s: s is not None, state_input_list)),
                                                    device=device, dtype=torch.bool)
                state_input_batch_non_final = torch.cat([s for s in state_input_list if s is not None]).to(device)
                actor_obs_list = list(itertools.chain.from_iterable(batch.actor_obs_list))  # flatten list
                actor_obs_batch_non_final = torch.cat([s for s in actor_obs_list if s is not None]).to(device)
                critc_obs_list = list(itertools.chain.from_iterable(batch.critic_obs_list))  # flatten list
                critc_obs_batch_non_final = torch.cat([s for s in critc_obs_list if s is not None]).to(device)

                test1 = torch.tensor(tuple(map(lambda s: s is not None, actor_obs_list)), dtype=torch.bool)
                test2 = torch.tensor(tuple(map(lambda s: s is not None, critc_obs_list)), dtype=torch.bool)
                # if not (torch.all(test1 == non_final_state_mask).item() and torch.all(test1 == test2).item()):
                #     print("Error")

                next_state_input_list = list(itertools.chain.from_iterable(batch.next_state_input))
                non_final_next_state_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_input_list)),
                                                         device=device, dtype=torch.bool)
                next_state_input_batch_non_final = torch.cat([s for s in next_state_input_list if s is not None]).to(
                    device)
                next_critic_obs_list = list(itertools.chain.from_iterable(batch.next_critic_obs_list))
                next_critic_obs_batch_non_final = torch.cat([s for s in next_critic_obs_list if s is not None]).to(
                    device)

                values = torch.zeros((data_size * self.num_UAV, 1), device=device)
                values[non_final_state_mask, :] = self.critic(state_input_batch_non_final, critc_obs_batch_non_final)
                next_values = torch.zeros((data_size * self.num_UAV, 1), device=device)
                next_values[non_final_next_state_mask, :] = self.critic(next_state_input_batch_non_final,
                                                                        next_critic_obs_batch_non_final)
                reward_batch = torch.cat(batch.reward).to(device)
                # reward_test = torch.reshape(reward_batch, (data_size, self.num_UAV))
                expected_values = (reward_batch + self.gamma * next_values).detach()

                # Critic Loss
                self.critic_optimizer.zero_grad()
                value_losses = torch.square(expected_values.squeeze(1).detach() - values.squeeze(1)).mean()
                value_losses.backward()
                self.critic_optimizer.step()

                delta_batch = expected_values - values.detach()
                delta_i = torch.reshape(delta_batch, (data_size, self.num_UAV))
                delta_i_matrix = torch.repeat_interleave(delta_i, repeats=self.num_UAV, dim=0)
                neighbor_batch = torch.cat(batch.neighbors).to(device)
                delta_batch = (neighbor_batch * delta_i_matrix).mean(1).unsqueeze(1).detach()
                action_batch = torch.cat(batch.action).to(device)
                probs = self.actor(state_input_batch_non_final, actor_obs_batch_non_final)
                log_probs = torch.zeros((data_size * self.num_UAV, 1), device=device)
                log_probs[non_final_state_mask, :] = torch.log(probs.gather(1, action_batch[non_final_state_mask]))

                self.actor_optimizer.zero_grad()
                policy_losses = (-log_probs.squeeze(1) * delta_batch.squeeze(1)).mean()
                policy_losses.backward()
                self.actor_optimizer.step()

                self.replay_memory.clear_memory()

            else:
                transitions = self.replay_memory.return_all()
                data_size = len(transitions)
                batch = Transition(*zip(*transitions))
                state_input_list = list(itertools.chain.from_iterable(batch.state_input))  # flatten list
                non_final_state_mask = torch.tensor(tuple(map(lambda s: s is not None, state_input_list)),
                                                    device=device, dtype=torch.bool)
                state_input_batch_non_final = torch.cat([s for s in state_input_list if s is not None]).to(device)
                actor_obs_list = list(itertools.chain.from_iterable(batch.actor_obs_list))  # flatten list

                test1 = torch.tensor(tuple(map(lambda s: s is not None, actor_obs_list)), device=device,
                                    dtype=torch.bool)
                if not torch.all(test1 == non_final_state_mask).item():
                    print("Error")

                actor_obs_batch_non_final = torch.cat([s for s in actor_obs_list if s is not None]).to(device)
                critic_obs_list = list(itertools.chain.from_iterable(batch.critic_obs_list))
                critic_obs_batch_non_final = torch.cat([s for s in critic_obs_list if s is not None]).to(device)
                next_state_input_list = list(itertools.chain.from_iterable(batch.next_state_input))
                non_final_next_state_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_input_list)),
                                                         device=device, dtype=torch.bool)
                next_state_input_batch_non_final = torch.cat([s for s in next_state_input_list if s is not None]).to(
                    device)
                next_critic_obs_list = list(itertools.chain.from_iterable(batch.next_critic_obs_list))
                next_critic_obs_batch_non_final = torch.cat([s for s in next_critic_obs_list if s is not None]).to(
                    device)

                values = torch.zeros((data_size * self.num_UAV, 1), device=device)
                values[non_final_state_mask, :] = self.critic(state_input_batch_non_final, critic_obs_batch_non_final)
                next_values = torch.zeros((data_size * self.num_UAV, 1), device=device)
                next_values[non_final_next_state_mask, :] = self.critic(next_state_input_batch_non_final,
                                                                        next_critic_obs_batch_non_final)

                reward_batch = torch.cat(batch.reward).to(device)
                # reward_test = torch.reshape(reward_batch, (data_size, self.num_UAV))
                expected_values = (reward_batch + self.gamma * next_values).detach()
                delta_batch = expected_values - values.detach()
                # Critic Loss
                self.critic_optimizer.zero_grad()
                value_losses = torch.square(expected_values.squeeze(1).detach() - values.squeeze(1)).mean()
                value_losses.backward()
                self.critic_optimizer.step()

                action_batch = torch.cat(batch.action).to(device)
                probs = self.actor(state_input_batch_non_final, actor_obs_batch_non_final)
                log_probs = torch.zeros((data_size * self.num_UAV, 1), device=device)
                log_probs[non_final_state_mask, :] = torch.log(probs.gather(1, action_batch[non_final_state_mask]))

                self.actor_optimizer.zero_grad()
                policy_losses = (-log_probs.squeeze(1) * delta_batch.squeeze(1)).mean()
                policy_losses.backward()
                self.actor_optimizer.step()

                self.replay_memory.clear_memory()


    def save_parameters(self, idx):
        torch.save(self.actor.state_dict(), 'Data/MAAC_v8_' + self.label + '_trainedActor_' + str(idx))
        torch.save(self.critic.state_dict(), 'Data/MAAC_v8_' + self.label + '_trainedCritic_' + str(idx))

    def soft_update(self, net, target_net):
        for target_param, local_param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, net, target_net):
        target_net.load_state_dict(net.state_dict())

    ###################################################################################
    # Return useful numbers
    def total_reward(self):
        return self._total_reward

    def total_discounted_reward(self):
        return self._total_discounted_reward

    def num_episodes(self):
        return self._num_episodes

    def num_ep_steps(self):
        return self._num_ep_steps

    def get_min_intr_dist(self):
        return self._min_intr_dist

    ###################################################################################
    # Following functions are used to update Pygame window
    def handle_event(self):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            self.close_clicked = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            # y = pos[0] // (self.w + self.margin)
            # x = pos[1] // (self.w + self.margin)
            # if self.maze[y][x] == 0:
            #     self.maze[y][x] = 1
            # elif self.maze[y][x] == 1:
            #     self.maze[y][x] = 0
            # self._environment.update_wall(self.maze)

    def drawGrid(self):
        if self.surface is None:
            return
        self.surface.fill(self.bg_color)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                grid = [(self.margin + self.w) * row + self.margin, (self.margin + self.w) * col + self.margin, self.w,
                        self.w]
                if self.maze[row][col] == 1:
                    pygame.draw.rect(self.surface, self.wall_color, grid)
                else:
                    # value = -(self._agent.calValue((row, col)))//15
                    # pygame.draw.rect(self.surface,pygame.Color(self.color[value]),grid)
                    pygame.draw.rect(self.surface, self.normal_color, grid)

    def showChar(self):
        if self.surface is None:
            return
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 10)
        for uav in range(self.num_UAV):
            start = myfont.render('S', False, self.u_color[uav])
            self.surface.blit(start, (self.start[uav][1] * (self.w + self.margin),
                                      self.start[uav][0] * (self.w + self.margin) - 2))
            goal = myfont.render('D', False, self.u_color[uav])
            self.surface.blit(goal, (self.goal[uav][1] * (self.w + self.margin),
                                     self.goal[uav][0] * (self.w + self.margin) - 2))

    def drawUserBox(self, pos):
        if self.surface is None:
            return
        for uav in range(self.num_UAV):
            x, y = pos[uav]
            grid = [(self.margin + self.w) * y + self.margin, (self.margin + self.w) * x + self.margin, self.w,
                    self.w]
            pygame.draw.rect(self.surface, self.u_color[uav], grid)

    def display_on_surface(self, state):
        if self.surface is not None:
            self.drawGrid()
            self.showChar()
            self.drawUserBox(state)
            pygame.display.update()
    # Above functions are used to update Pygame window


