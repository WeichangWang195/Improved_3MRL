# from main_alg import rl_alg
from alg.independent import rl_alg_independent
from alg.scalable import rl_alg_scalable
from alg.scalable_softac import rl_alg_scalable_softac
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
    idx = 0
    label = str(idx) + str(rdm_seed)
    setup_seed(rdm_seed)
    grid_size = 30
    w = 19
    margin = 1
    view_length = 5
    critic_view_length = 25 if obs_size == 0 else view_length
    safe_dist = 2
    reward_normalization = 1/200

    # start_set = (((29, 29), (29, 28), (28, 29), (28, 28)),
    #              ((0, 29), (0, 28), (1, 29), (1, 28)),
    #              ((29, 0), (28, 0), (29, 1), (28, 1)),
    #              ((0, 0), (1, 0), (0, 1), (1, 1)))
    # goal_set = ((0, 0), (29, 0), (0, 29), (29, 29))

    # self.start = ((31, 49), (42, 42), (49, 31), (49, 18), (42, 7), (31, 0), (18, 0), (7, 7), (0, 18), (0, 31), (7, 42), (18, 49))
    # self.goal = ((18, 0), (7, 7), (0, 18), (0, 31), (7, 42), (18, 49), (31, 49), (42, 42), (49, 31), (49, 18), (42, 7), (31, 0))

    start_set = (((0, 8), (0, 9), (1, 8), (1, 9)),
                 ((0, 18), (0, 19), (1, 18), (1, 19)),
                 ((29, 8), (29, 9), (28, 8), (28, 9)),
                 ((29, 18), (29, 19), (28, 18), (28, 19)),
                 ((8, 28), (9, 29), (8, 28), (9, 29)),
                 ((18, 28), (19, 29), (18, 28), (19, 29)),
                 ((18, 0), (19, 0), (18, 1), (19, 1)),
                 ((8, 0), (9, 0), (8, 1), (9, 1)))
    goal_set = ((29, 19), (29, 9), (0, 19), (0, 9), (19, 0), (9, 0), (9, 29), (19, 29))

    # start_set = (((0, 15), (0, 16), (0, 17), (1, 15), (1, 16), (1, 17)),
    #              ((0, 33), (0, 34), (0, 35), (1, 33), (1, 34), (1, 35)),
    #              ((15, 49), (16, 49), (17, 49), (15, 48), (16, 48), (17, 48)),
    #              ((33, 49), (34, 49), (35, 49), (33, 48), (34, 48), (35, 48)),
    #              ((49, 33), (49, 34), (49, 35), (48, 33), (48, 34), (48, 35)),
    #              ((49, 15), (49, 16), (49, 17), (48, 15), (48, 16), (48, 17)),
    #              ((33, 0), (34, 0), (35, 0), (33, 1), (34, 1), (35, 1)),
    #              ((15, 0), (16, 0), (17, 0), (15, 1), (16, 1), (17, 1)),)
    # goal_set = ((49, 34), (49, 16), (34, 0), (16, 0), (0, 16), (0, 34), (16, 49), (34, 49))

    surface = create_window(grid_size, w+margin)

    maxEpisodes = 100010
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
        'obs_type': obs_type,
        'num_UAV': num_UAV,
        'num_dest': num_dest,
        'start_set': start_set,
        'goal_set': goal_set,
        'label': label,
        'grid_size': grid_size,
        'safe_dist': safe_dist,
        'view_length': view_length,
        'critic_view_length': critic_view_length,
        'load_param': None,
        'gamma': 0.9,
        'tau': 0.01,
        'w': w,
        'margin': margin,
        }
    if alg == 0:
        rlglue = rl_alg_independent(environment, cfg, surface=surface)
    elif alg == 1:
        rlglue = rl_alg_scalable(environment, cfg, surface=surface)
    else:
        rlglue = rl_alg_scalable_softac(environment, cfg, surface=surface)

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

    if alg == 0:
        rlglue = rl_alg_scalable(environment, cfg, surface=surface)
    else:
        rlglue = rl_alg_independent(environment, cfg, surface=surface)

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

def create_window(grid, size):
    title = "Collision Avoidance"
    size = (size*grid, size*grid)
    pygame.init()
    if pygame.display.get_init():
        surface = pygame.display.set_mode(size, 0, 0)
    else:
        surface = None
    pygame.display.set_caption(title)
    return surface


if __name__ == "__main__":
    seed = 0
    # run(8, 8, rdm_seed=seed, alg=2, obs_type=0, obs_size=0)
    # run(8, 8, rdm_seed=seed, alg=1, obs_type=0, obs_size=0)
    run(8, 8, rdm_seed=seed, alg=0, obs_type=0, obs_size=1)
    # run(12, 12, rdm_seed=seed, alg=1, obs_type=0, obs_size=1)
    # run(8, 8, rdm_seed=seed, alg=0, obs_type=0, obs_size=1)