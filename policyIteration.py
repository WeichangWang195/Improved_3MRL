import math
import numpy as np
import time
import itertools
import pickle
import random
from collections import namedtuple
from modules.ReplayMemory import ReplayMemory
from utils.rl_utils import TabularQTransition as Transition

BATCH_SIZE = 128
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 100000

def policy_iteration():
    grid_size = 8
    num_UAV = 2
    action_space = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    normalized_term = 1 / 200
    r_terminal = grid_size * 10 * normalized_term
    step_reward = -1.0 * normalized_term
    dest_reward_coef = -0.25 * normalized_term
    safe_dist = 2
    collision_penalty = -grid_size * 5 * normalized_term
    gamma = 0.99
    Q_size = [grid_size for _ in range(num_UAV * 2 * 2)] + [len(action_space)] + [len(action_space)]
    Q = np.zeros(Q_size)
    L = list(itertools.product(range(grid_size), range(grid_size), range(grid_size), range(grid_size),
                               range(len(action_space))))
    # for pos1_x, pos1_y, pos2_x, pos2_y in itertools.product(range(grid_size), repeat=4):
    #     Q[pos1_x, pos1_y, pos2_x, pos2_y, pos1_x, pos1_y, pos2_x, pos2_y, :, :] = r_terminal
    diff = 1
    while diff !=0:
        print("diff = ", diff)
        diff = 0
        for state_action1 in L:
            pos1 = [state_action1[0], state_action1[1]]
            dest1 = [state_action1[2], state_action1[3]]
            action1 = state_action1[4]
            reward1 = 0.0
            if pos1 == dest1:
                next_pos1 = pos1
            else:
                move = action_space[action1]
                test_state = tuple(map(sum, zip(pos1, move)))
                if 0 <= test_state[0] <= grid_size - 1 and 0 <= test_state[1] <= grid_size - 1:
                    next_pos1 = test_state
                else:
                    next_pos1 = pos1
            if pos1 != dest1 and next_pos1 != dest1:
                reward1 = dest_reward_coef * math.sqrt(
                    sum(map(lambda x, y: (x - y) ** 2, next_pos1, dest1))) + step_reward
            elif pos1 != dest1 and next_pos1 == dest1:
                reward1 = r_terminal
            for state_action2 in L:
                collision = False
                pos2 = [state_action2[0], state_action2[1]]
                dest2 = [state_action2[2], state_action2[3]]
                action2 = state_action2[4]
                if pos2 == dest2 and pos1 == dest1:
                    continue
                reward2 = 0.0
                if pos2 == dest2:
                    next_pos2 = pos2
                else:
                    move = action_space[action2]
                    test_state = tuple(map(sum, zip(pos2, move)))
                    if 0 <= test_state[0] <= grid_size - 1 and 0 <= test_state[1] <= grid_size - 1:
                        next_pos2 = test_state
                    else:
                        next_pos2 = pos2
                agent_dist = math.sqrt(sum(map(lambda x, y: (x - y) ** 2, next_pos1, next_pos2)))
                if pos1 != dest1 and pos2 != dest2 and agent_dist < safe_dist:
                    collision = True
                if pos2 != dest2 and next_pos2 != dest2:
                    reward2 = dest_reward_coef * math.sqrt(
                        sum(map(lambda x, y: (x - y) ** 2, next_pos2, dest2))) + step_reward
                    reward2 += collision_penalty if collision else 0
                elif pos2 != dest2 and next_pos2 == dest2:
                    reward2 = r_terminal
                reward = reward1 + reward2
                if pos1 != dest1 and next_pos1 != dest1 and collision:
                    reward += collision_penalty

                next_value = np.max(Q[next_pos1[0], next_pos1[1], next_pos2[0], next_pos2[1],
                                    dest1[0], dest1[1], dest2[0], dest2[1], :, :])
                value = Q[pos1[0], pos1[1], pos2[0], pos2[1], dest1[0], dest1[1], dest2[0], dest2[1], action1, action2]
                if value != reward + gamma * next_value:
                    Q[pos1[0], pos1[1], pos2[0], pos2[1], dest1[0], dest1[1], dest2[0], dest2[1], action1, action2] = reward + gamma * next_value
                    diff += 1
    np.savez('D:\PytorchData\TabularQ\G8U2_optimal', Q=Q)


def tabular_Q_MA():
    grid_size = 8
    num_UAV = 2
    action_space = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    normalized_term = 1 / 200
    r_terminal = grid_size * 10 * normalized_term
    step_reward = -1.0 * normalized_term
    dest_reward_coef = -0.25 * normalized_term
    safe_dist = 2
    collision_penalty = -grid_size * 5 * normalized_term
    gamma = 0.99
    alpha = 0.001
    mode = "StartTrain"
    if mode != "StartTrain":
        npzfile = np.load('D:\PytorchData\TabularQ\G10U2_0')
        Q = npzfile['Q']
    else:
        Q_size = [grid_size for _ in range(num_UAV * 2 * 2)] + [len(action_space) for _ in range(num_UAV)]
        Q = np.zeros(Q_size)
    replay_memory = ReplayMemory(10000)
    State_Tuple = namedtuple('State_Tuple', ['pos_x', 'pos_y', 'intr_pos_x', 'intr_pos_y',
                                             'dest_x', 'dest_y', 'intr_dest_x', 'intr_dest_y'])
    # for pos_x, pos_y in itertools.product(range(grid_size), range(grid_size)):
    #     Q[:, :, pos_x, pos_y, :, :, pos_x, pos_y, :] = r_terminal * normalized_term
    #     Q[pos_x, pos_y, :, :, pos_x, pos_y, :, :, :] = r_terminal * normalized_term
    run_episode = 3000000
    diff = np.zeros(run_episode)
    total_reward = np.zeros(run_episode)
    for episode in range(run_episode):
        print("Episode: %d" % episode)
        if episode % int(run_episode/10) == 0:
            idx = episode // int(run_episode/10)
            np.savez('D:\PytorchData\TabularQ\G10U2_'+str(idx), Q=Q, diff=diff, total_reward=total_reward)
        state = []
        episode_diff = 0.0
        episode_total_reward = 0.0
        collision_in_episode = False
        for _ in range(4*num_UAV):
            state.append(random.randrange(grid_size))
        step = 0
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
        pos = ((state[0], state[1]), (state[2], state[3]))
        dest = ((state[4], state[5]), (state[6], state[7]))
        # pos = ((0, 0), (9, 9))
        # dest = ((0, 9), (9, 0))
        while step < 100 and (pos[0] != dest[0] or pos[1] != dest[1]):
            next_pos = []
            collision = False
            action_idx = np.argmax(Q[pos[0][0], pos[0][1], pos[1][0], pos[1][1],
                                dest[0][0], dest[0][1], dest[1][0], dest[1][1], :, :])
            action = [action_idx // len(action_space)
                      if random.uniform(0, 1) > epsilon else random.randrange(len(action_space)),
                      action_idx % len(action_space)
                      if random.uniform(0, 1) > epsilon else random.randrange(len(action_space))]
            action = tuple(action)
            # action = [np.argmax(Q[pos[0][0], pos[0][1], pos[1][0], pos[1][1],
            #                     dest[0][0], dest[0][1], dest[1][0], dest[1][1], :])
            #           if random.uniform(0, 1) > epsilon else random.randrange(len(action_space)),
            #           np.argmax(Q[pos[1][0], pos[1][1], pos[0][0], pos[0][1],
            #                     dest[1][0], dest[1][1], dest[0][0], dest[0][1], :])
            #           if random.uniform(0, 1) > epsilon else random.randrange(len(action_space))]
            for ii in range(num_UAV):
                if pos[ii] == dest[ii]:
                    next_pos.append(pos[ii])
                else:
                    move = action_space[action[ii]]
                    test_state = tuple(map(sum, zip(pos[ii], move)))
                    if 0 <= test_state[0] <= grid_size-1 and 0 <= test_state[1] <= grid_size-1:
                        next_pos.append(test_state)
                    else:
                        next_pos.append(pos[ii])
            agent_dist = math.sqrt(sum(map(lambda x, y: (x - y) ** 2, next_pos[0], next_pos[1])))
            if pos[0] != dest[0] and pos[1] != dest[1] and agent_dist < safe_dist:
                collision = True
                collision_in_episode = True
            reward = 0
            for ii in range(num_UAV):
                if pos[ii] != dest[ii] and next_pos[ii] != dest[ii]:
                    reward += dest_reward_coef * math.sqrt(sum(map(lambda x, y: (x - y) ** 2, next_pos[ii], dest[ii]))) + step_reward
                    reward += collision_penalty if collision else 0
                elif pos[ii] != dest[ii] and next_pos[ii] == dest[ii]:
                    reward = r_terminal
            state = [pos[0][0], pos[0][1], pos[1][0], pos[1][1],
                         dest[0][0], dest[0][1], dest[1][0], dest[1][1]]
            next_state = [next_pos[0][0], next_pos[0][1], next_pos[1][0], next_pos[1][1],
                              dest[0][0], dest[0][1], dest[1][0], dest[1][1]]
            episode_total_reward += reward
            replay_memory.push(Transition(state, action, reward, next_state))
            step += 1
            pos = tuple(next_pos)

            if episode > 2:
                transitions = replay_memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                state_batch = batch.state
                reward_batch = batch.reward
                action_batch = batch.action
                action_batch_np = np.array(action_batch)
                action_0 = action_batch_np[:, 0].tolist()
                action_1 = action_batch_np[:, 1].tolist()
                next_state_batch = batch.next_state
                state_info = State_Tuple(*zip(*state_batch))
                next_state_info = State_Tuple(*zip(*next_state_batch))

                next_max_value_batch = np.max(Q[next_state_info.pos_x, next_state_info.pos_y,
                                                next_state_info.intr_pos_x, next_state_info.intr_pos_y,
                                                next_state_info.dest_x, next_state_info.dest_y,
                                                next_state_info.intr_dest_x, next_state_info.intr_dest_y, :, :])
                state_value_batch = Q[state_info.pos_x, state_info.pos_y,
                                      state_info.intr_pos_x, state_info.intr_pos_y,
                                      state_info.dest_x, state_info.dest_y,
                                      state_info.intr_dest_x, state_info.intr_dest_y, action_0, action_1]
                td_error = reward_batch + gamma * next_max_value_batch - state_value_batch
                Q[state_info.pos_x, state_info.pos_y,
                  state_info.intr_pos_x, state_info.intr_pos_y,
                  state_info.dest_x, state_info.dest_y,
                  state_info.intr_dest_x, state_info.intr_dest_y, action_0, action_1] += alpha * td_error
                Q[state_info.intr_pos_x, state_info.intr_pos_y,
                  state_info.pos_x, state_info.pos_y,
                  state_info.intr_dest_x, state_info.intr_dest_y,
                  state_info.dest_x, state_info.dest_y, action_1, action_0] += alpha * td_error
                episode_diff += np.abs(td_error).sum()
        diff[episode] = episode_diff
        total_reward[episode] = episode_total_reward
        arrive = True if pos[0] == dest[0] and pos[1] == dest[1] else False
        print("Total TD is %2f" % episode_diff)
        print("Collision: %r" % collision_in_episode)
        print("Toral Reward: %2f" % episode_total_reward)
        print("Arrive: %r" % arrive)


def tabular_Q_MA_test():
    grid_size = 8
    num_UAV = 2
    action_space = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    normalized_term = 1 / 200
    r_terminal = grid_size * 10 * normalized_term
    step_reward = -1.0 * normalized_term
    dest_reward_coef = -0.25 * normalized_term
    safe_dist = 2
    collision_penalty = -grid_size * 5 * normalized_term
    gamma = 0.99

    # start_list = []
    # goal_list = []
    # for tt in range(1000):
    #     start_test = np.random.randint(grid_size, size=(2, 2))
    #     # if the start points suffer from collisions, re-generate it.
    #     for ii in range(1, 2):
    #         count = 0
    #         while np.any(np.square(start_test[ii, :] - start_test[:ii, :]).sum(1) < safe_dist ** 2) and count < 10:
    #             start_test[ii, :] = np.random.randint(grid_size, size=(1, 2))
    #             count += 1
    #     goal = np.random.randint(grid_size, size=(2, 2))
    #     # If any Goal == Start, re generate Goal
    #     while np.any(np.all(goal == start_test, 1)):
    #         goal[np.all(goal == start_test, 1), :] = np.random.randint(num_UAV,
    #                                                                    size=(np.sum(np.all(goal == start_test, 1)), 2))
    #     start_list.append(tuple(map(tuple, start_test)))
    #     goal_list.append(tuple(map(tuple, goal)))
    # np.savez('Data\TabularQ\start_goal', start_list=start_list, goal_list=goal_list)

    npzfile_sg = np.load('Data\TabularQ\start_goal.npz')
    start_list = npzfile_sg['start_list']
    goal_list = npzfile_sg['goal_list']
    run_episode = 1000

    npzfile = np.load('G:\My Drive\Data\G8U2_9.npz')
    Q = npzfile['Q']
    total_reward = np.zeros(run_episode)
    min_dist = np.zeros(run_episode)

    for episode in range(run_episode):
        print("Episode: %d" % episode)
        state = []
        episode_diff = 0.0
        episode_total_reward = 0.0
        episode_min_dist = grid_size * 2
        collision_in_episode = False
        for _ in range(4*num_UAV):
            state.append(random.randrange(grid_size))
        step = 0
        pos = ((start_list[episode][0, 0], start_list[episode][0, 1]), (start_list[episode][1, 0], start_list[episode][1, 1]))
        dest = ((goal_list[episode][0, 0], goal_list[episode][0, 1]), (goal_list[episode][1, 0], goal_list[episode][1, 1]))
        while step < 100 and (pos[0] != dest[0] or pos[1] != dest[1]):
            next_pos = []
            collision = False
            action = [np.argmax(Q[pos[0][0], pos[0][1], pos[1][0], pos[1][1],
                                dest[0][0], dest[0][1], dest[1][0], dest[1][1], :]),
                      np.argmax(Q[pos[1][0], pos[1][1], pos[0][0], pos[0][1],
                                dest[1][0], dest[1][1], dest[0][0], dest[0][1], :])]
            for ii in range(num_UAV):
                if pos[ii] == dest[ii]:
                    next_pos.append(pos[ii])
                else:
                    move = action_space[action[ii]]
                    test_state = tuple(map(sum, zip(pos[ii], move)))
                    if 0 <= test_state[0] <= grid_size-1 and 0 <= test_state[1] <= grid_size-1:
                        next_pos.append(test_state)
                    else:
                        next_pos.append(pos[ii])
            agent_dist = math.sqrt(sum(map(lambda x, y: (x - y) ** 2, next_pos[0], next_pos[1])))
            if agent_dist < episode_min_dist:
                episode_min_dist = agent_dist
            if pos[0] != dest[0] and pos[1] != dest[1] and agent_dist < safe_dist:
                collision = True
                collision_in_episode = True
            for ii in range(num_UAV):
                if pos[ii] != dest[ii] and next_pos[ii] != dest[ii]:
                    reward = dest_reward_coef * math.sqrt(sum(map(lambda x, y: (x - y) ** 2, next_pos[ii], dest[ii]))) + step_reward
                    reward += collision_penalty if collision else 0
                    episode_total_reward += gamma ** step * reward
                elif pos[ii] != dest[ii] and next_pos[ii] == dest[ii]:
                    reward = r_terminal
                    episode_total_reward += gamma ** step * reward
            step += 1
            pos = tuple(next_pos)

        total_reward[episode] = episode_total_reward
        min_dist[episode] = episode_min_dist
        print("Total TD is %2f" % episode_diff)
        print("Collision: %r" % collision_in_episode)
        print("Toral Reward: %2f" % episode_total_reward)
    np.savez('Data\TabularQ\Test9', total_reward=total_reward, min_dist=min_dist)


def tabular_Q_Joint_test():
    grid_size = 8
    num_UAV = 2
    action_space = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    normalized_term = 1 / 200
    r_terminal = grid_size * 10 * normalized_term
    step_reward = -1.0 * normalized_term
    dest_reward_coef = -0.25 * normalized_term
    safe_dist = 2
    collision_penalty = -grid_size * 5 * normalized_term
    gamma = 0.99

    # start_list = []
    # goal_list = []
    # for tt in range(1000):
    #     start_test = np.random.randint(grid_size, size=(2, 2))
    #     # if the start points suffer from collisions, re-generate it.
    #     for ii in range(1, 2):
    #         count = 0
    #         while np.any(np.square(start_test[ii, :] - start_test[:ii, :]).sum(1) < safe_dist ** 2) and count < 10:
    #             start_test[ii, :] = np.random.randint(grid_size, size=(1, 2))
    #             count += 1
    #     goal = np.random.randint(grid_size, size=(2, 2))
    #     # If any Goal == Start, re generate Goal
    #     while np.any(np.all(goal == start_test, 1)):
    #         goal[np.all(goal == start_test, 1), :] = np.random.randint(num_UAV,
    #                                                                    size=(np.sum(np.all(goal == start_test, 1)), 2))
    #     start_list.append(tuple(map(tuple, start_test)))
    #     goal_list.append(tuple(map(tuple, goal)))
    # np.savez('Data\TabularQ\start_goal', start_list=start_list, goal_list=goal_list)

    npzfile_sg = np.load('Data\TabularQ\start_goal.npz')
    start_list = npzfile_sg['start_list']
    goal_list = npzfile_sg['goal_list']
    run_episode = 1000

    npzfile = np.load('G:\My Drive\Data\G8U2_9.npz')
    Q = npzfile['Q']
    total_reward = np.zeros(run_episode)
    min_dist = np.zeros(run_episode)

    for episode in range(run_episode):
        print("Episode: %d" % episode)
        state = []
        episode_diff = 0.0
        episode_total_reward = 0.0
        episode_min_dist = grid_size * 2
        collision_in_episode = False
        for _ in range(4 * num_UAV):
            state.append(random.randrange(grid_size))
        step = 0
        pos = (
        (start_list[episode][0, 0], start_list[episode][0, 1]), (start_list[episode][1, 0], start_list[episode][1, 1]))
        dest = (
        (goal_list[episode][0, 0], goal_list[episode][0, 1]), (goal_list[episode][1, 0], goal_list[episode][1, 1]))
        while step < 100 and (pos[0] != dest[0] or pos[1] != dest[1]):
            next_pos = []
            collision = False
            action = [np.argmax(Q[pos[0][0], pos[0][1], pos[1][0], pos[1][1],
                                dest[0][0], dest[0][1], dest[1][0], dest[1][1], :]),
                      np.argmax(Q[pos[1][0], pos[1][1], pos[0][0], pos[0][1],
                                dest[1][0], dest[1][1], dest[0][0], dest[0][1], :])]
            for ii in range(num_UAV):
                if pos[ii] == dest[ii]:
                    next_pos.append(pos[ii])
                else:
                    move = action_space[action[ii]]
                    test_state = tuple(map(sum, zip(pos[ii], move)))
                    if 0 <= test_state[0] <= grid_size - 1 and 0 <= test_state[1] <= grid_size - 1:
                        next_pos.append(test_state)
                    else:
                        next_pos.append(pos[ii])
            agent_dist = math.sqrt(sum(map(lambda x, y: (x - y) ** 2, next_pos[0], next_pos[1])))
            if agent_dist < episode_min_dist:
                episode_min_dist = agent_dist
            if pos[0] != dest[0] and pos[1] != dest[1] and agent_dist < safe_dist:
                collision = True
                collision_in_episode = True
            for ii in range(num_UAV):
                if pos[ii] != dest[ii] and next_pos[ii] != dest[ii]:
                    reward = dest_reward_coef * math.sqrt(
                        sum(map(lambda x, y: (x - y) ** 2, next_pos[ii], dest[ii]))) + step_reward
                    reward += collision_penalty if collision else 0
                    episode_total_reward += gamma ** step * reward
                elif pos[ii] != dest[ii] and next_pos[ii] == dest[ii]:
                    reward = r_terminal
                    episode_total_reward += gamma ** step * reward
            step += 1
            pos = tuple(next_pos)

        total_reward[episode] = episode_total_reward
        min_dist[episode] = episode_min_dist
        print("Total TD is %2f" % episode_diff)
        print("Collision: %r" % collision_in_episode)
        print("Toral Reward: %2f" % episode_total_reward)
    np.savez('Data\TabularQ\Test9', total_reward=total_reward, min_dist=min_dist)


def tabular_dp_MA():
    grid_size = 7
    num_UAV = 2
    action_space = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    normalized_term = 1 / 200
    r_terminal = grid_size * normalized_term
    step_reward = -1.0 * normalized_term
    dest_reward_coef = -0.25 * normalized_term
    safe_dist = 2
    collision_penalty = -grid_size * 10 * normalized_term
    bad_state_penalty = -5
    gamma = 0.99

    mode = "StartTrain"
    if mode != "StartTrain":
        npzfile = np.load('D:\PytorchData\TabularQ\G10U2_0')
        Q = npzfile['Q']
    else:
        Q_size = [grid_size for _ in range(num_UAV * 2 * 2)] + [len(action_space)]
        Q = np.zeros(Q_size)

    agent_L = list(itertools.product(range(grid_size), range(grid_size), range(grid_size), range(grid_size),
                                     range(len(action_space))))
    intruder_L = list(itertools.product(range(grid_size), range(grid_size), range(grid_size), range(grid_size)))
    # Set bad state: agent_pos == intr_pos, agent_dest == intr_dest
    for state in intruder_L:
        Q[state[0], state[1], state[2], state[3], state[0], state[1], state[2], state[3], :] = bad_state_penalty
    # Set bad state: Out of range.
    Q[0, :, :, :, :, :, :, :, 0] = bad_state_penalty
    Q[grid_size-1, :, :, :, :, :, :, :, 1] = bad_state_penalty
    Q[:, grid_size-1, :, :, :, :, :, :, 2] = bad_state_penalty
    Q[:, 0, :, :, :, :, :, :, 3] = bad_state_penalty
    #Set Arriving State
    for state in list(itertools.product(range(grid_size), range(grid_size))):
        Q[state[0], state[1], state[0], state[1], :, :, :, :, :] = r_terminal

    diff = 1
    while diff !=0:
        print("diff = ", diff)
        diff = 0
        for agent_state_action in agent_L:
            agent_pos = (agent_state_action[0], agent_state_action[1])
            agent_dest = (agent_state_action[2], agent_state_action[3])
            action = agent_state_action[4]
            if agent_pos == agent_dest:
                continue
            else:
                move = action_space[action]
                test_state = tuple(map(sum, zip(agent_pos, move)))
                if 0 <= test_state[0] <= grid_size - 1 and 0 <= test_state[1] <= grid_size - 1:
                    agent_next_pos = test_state
                else:
                    # Out of Range
                    continue
                reward_non_collision_part = dest_reward_coef * math.sqrt(
                    sum(map(lambda x, y: (x - y) ** 2, agent_pos, agent_dest))) + step_reward
                for intr_state in intruder_L:
                    intr_out = False
                    intr_pos = (intr_state[0], intr_state[1])
                    intr_dest = (intr_state[2], intr_state[3])
                    if intr_pos == agent_pos and intr_dest == agent_dest:
                        continue
                    intr_action = np.argmax(Q[intr_pos[0], intr_pos[1], intr_dest[0], intr_dest[1],
                                            agent_pos[0], agent_pos[1], agent_dest[0], agent_dest[1], :])
                    if intr_pos == intr_dest:
                        intr_next_pos = intr_pos
                    else:
                        move = action_space[int(intr_action)]
                        test_state = tuple(map(sum, zip(intr_pos, move)))
                        if 0 <= test_state[0] <= grid_size - 1 and 0 <= test_state[1] <= grid_size - 1:
                            intr_next_pos = test_state
                        else:
                            intr_next_pos = intr_pos
                            intr_out = True
                    agent_dist = math.sqrt(sum(map(lambda x, y: (x - y) ** 2, agent_pos, intr_pos)))
                    if intr_pos != intr_dest and (not intr_out) and agent_dist < safe_dist:
                        reward = reward_non_collision_part + collision_penalty
                    else:
                        reward = reward_non_collision_part

                    next_value = np.max(Q[agent_next_pos[0], agent_next_pos[1], agent_dest[0], agent_dest[1],
                                        intr_next_pos[0], intr_next_pos[1], intr_dest[0], intr_dest[1], :])
                    value = Q[agent_pos[0], agent_pos[1], agent_dest[0], agent_dest[1],
                              intr_pos[0], intr_pos[1], intr_dest[0], intr_dest[1], action]
                    if value != reward + gamma * next_value:
                        Q[agent_pos[0], agent_pos[1], agent_dest[0], agent_dest[1],
                          intr_pos[0], intr_pos[1], intr_dest[0], intr_dest[1], action] = reward + gamma * next_value
                        diff += 1

        np.savez('D:\PytorchData\TabularQ\G7I2_tabular_dp_MA_Q2', Q=Q)




if __name__ == "__main__":
    tabular_dp_MA()