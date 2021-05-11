from main_alg import rl_alg
from envMAAC import Environment
import pickle
from agentMAAC import Agent
import pygame
import numpy as np
import torch
import random
import sys


def run(num_UAV, num_dest, rdm_seed, alg=0, obs_type=0, obs_size=0):
    """
    :param num_UAV: num of agent in the system
    :param num_dest: num of destination
    :param rdm_seed: random seed
    :param alg: parameter update algorithm, alg=0: new algorithm; alg=1: old algorithm, general actor critic.
    :param obs_type: =1: meta learning; =0: general learning.
    :param obs_size: =1: actor obs size = critic obs size; =0, critic obs size >> actor obs size
    :return:
    """
    label = str(rdm_seed) + str(alg)
    setup_seed(rdm_seed)
    grid_size = 50
    view_length = 5
    critic_view_length = 25 if obs_size==0 else view_length
    safe_dist = 3
    reward_normalization = 1/200

    surface = create_window(grid_size)

    maxEpisodes = 50010
    env_cfg = {
        'num_UAV': num_UAV,
        'num_dest': num_dest,
        'grid_size': grid_size,
        'view_length': view_length,
        'obs_type': obs_type,
        'critic_view_length': critic_view_length,
        'reward_terminal': grid_size * 10 * reward_normalization,
        'safe_dist': safe_dist,
        'safe_dist_control': -grid_size * 5 * reward_normalization,
        'dest_control': -0.5 * reward_normalization,
        'step_control': -1 * reward_normalization,
    }
    environment = Environment(env_cfg)

    cfg = {
        'mode': "Train",
        'alg': alg,
        'obs_type': obs_type,
        'num_UAV': num_UAV,
        'num_dest': num_dest,
        'label': label,
        'grid_size': grid_size,
        'safe_dist': safe_dist,
        'view_length': view_length,
        'critic_view_length': critic_view_length,
        'load_param': None,
        'gamma': 0.9,
        'tau': 0.01
        }

    rlglue = rl_alg(environment, cfg, surface=surface)

    # Train Process
    rlglue.rl_init()
    for i in range(maxEpisodes):
        print("Round:", i)
        rlglue.rl_episode()
        print("Steps took in this episode is %d" % (rlglue.num_ep_steps()))
        print("Discounted Cost in this episode is %2f" % (rlglue.total_discounted_reward()))
        print("Cost in this episode is %2f" % (rlglue.total_reward()))
        print("Collision: %2f" % rlglue.get_min_intr_dist())


def test(num_UAV, rdm_seed, alg, delay):
    label = str(rdm_seed) + str(alg) + str(delay)
    setup_seed(rdm_seed)
    grid_size = 15
    view_length = 0
    safe_dist = 2
    reward_normalization = 1/200

    surface = create_window(grid_size)

    maxEpisodes = 7010
    env_cfg = {
        'num_UAV': num_UAV,
        'grid_size': grid_size,
        'view_length': view_length,
        'reward_terminal': grid_size * 10 * reward_normalization,
        'safe_dist': safe_dist,
        'safe_dist_control': -grid_size * 5 * reward_normalization,
        'dest_control': -0.1 * reward_normalization,
        'step_control': -1 * reward_normalization,
    }
    environment = Environment(env_cfg)

    cfg = {
        'mode': "Train",
        'alg': alg,
        'num_UAV': num_UAV,
        'label': label,
        'grid_size': grid_size,
        'safe_dist': safe_dist,
        'view_length': view_length,
        'load_param': None,
        'gamma': 0.99,
        'tau': 0.01
        }

    rlglue = rl_alg(environment, cfg, surface=surface)

    # Train Process
    rlglue.rl_init()
    for i in range(maxEpisodes):
        print("Round:", i)
        rlglue.rl_episode()
        print("Steps took in this episode is %d" % (rlglue.num_ep_steps()))
        print("Discounted Cost in this episode is %2f" % (rlglue.total_discounted_reward()))
        print("Cost in this episode is %2f" % (rlglue._total_reward()))
        print("Collision: %2f" % rlglue.get_min_intr_dist())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_window(grid_size):
    title = "Collision Avoidance"
    size = (10*grid_size, 10*grid_size)
    pygame.init()
    if pygame.display.get_init():
        surface = pygame.display.set_mode(size, 0, 0)
    else:
        surface = None
    pygame.display.set_caption(title)
    return surface


if __name__ == "__main__":
    seed = 0
    run(12, 12, rdm_seed=seed, alg=0, obs_type=0, obs_size=0)
    # run(12, 12, rdm_seed=seed, alg=1, obs_type=0, obs_size=1)
    # run(8, 8, rdm_seed=seed, alg=0, obs_type=0, obs_size=1)