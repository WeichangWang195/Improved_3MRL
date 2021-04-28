import math
import numpy as np
import time
import random
import itertools
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def figure_plot():
    with open("Data/discount_cost_v1_0w.pkl", 'rb') as f:
        cost = pickle.load(f)
    with open("Data/discount_cost_v1_0f.pkl", 'rb') as f:
        cost_no = pickle.load(f)
    plt.figure()
    data_length = 3000
    # optimal = [-2.49433] * data_length
    plt.plot(range(data_length), cost_no[:data_length], label="Without commander")
    plt.plot(range(data_length), cost[:data_length], label="With commander")
    # plt.plot(range(data_length), optimal, label="Optimal")
    plt.legend()
    plt.ylabel('Total Cost')
    plt.xlabel('Episodes')
    plt.show()


def figure_plot_dist(data_len, step):
    x = []
    data = []
    style = []
    data_length = data_len
    step = step
    for ii in range(1):
        with open("Data/eps_records_v1_" + str(ii) + "00", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['discounted_reward']
        for tt in range(data_length):
            x.append(int(round(tt / step)) * step)
            data.append(cost[tt])
            style.append("Normal commander")

    for ii in range(1):
        with open("Data/eps_records_v1_" + str(ii) + "20", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['discounted_reward']
        for tt in range(data_length):
            x.append(int(round(tt / step)) * step)
            data.append(cost[tt])
            style.append("NO commander")

    # for ii in range(1):
    #     with open("Data/discount_cost_v1_" + str(ii)+ "11.pkl", 'rb') as f:
    #         cost = pickle.load(f)
    #     for tt in range(data_length):
    #         x.append(int(round(tt / step)) * step)
    #         data.append(cost[tt])
    #         style.append("Commander Delay=1")

    # fmri = sns.load_dataset("fmri")
    data = pd.DataFrame(list(zip(x, data, style)), columns=['Episode', 'Total Reward', 'style'])
    # plt.figure()
    # sns.catplot(x="Episode", y="Total Cost", hue="style", kind="box", data=data)
    sns.relplot(x="Episode", y="Total Reward", style="style", kind="line", data=data)
    # plt.figure(1)
    # plt.plot(range(len(cost)), cost)
    # plt.ylabel('Total Cost')
    # plt.xlabel('Episodes')
    plt.show()


def plot_max_command():
    x = []
    data = []
    style = []
    data_length =2000
    step = 20

    for ii in range(3):
        with open("Data/max_command_v1_0" + str(ii)+ ".pkl", 'rb') as f:
            cost = pickle.load(f)
        for tt in range(data_length):
            x.append(int(round(tt / step)) * step)
            data.append(cost[tt].item())
            style.append("Normal Commander")

    df = pd.DataFrame( list(zip(x, data, style)), columns = ['Episode', 'Max Command', 'style'])
    sns.relplot(x="Episode", y="Max Command", style="style", kind="line", data=df)
    plt.show()


def plot_collision_rate(data_len, step):
    x = []
    data = []
    style = []
    data_length = data_len
    step = step

    for ii in range(10):
        with open("Data/TabularQ/Test" + str(ii)+ ".npz", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['min_dist']
        if ii == 0:
            x.append(300 * ii)
        else:
            x.append(300 * (10-ii))
        style.append("TabularQ")
        num_collision = 0
        for tt in range(len(cost)):
            if cost[tt] < 2:
                num_collision+=1
        data.append(num_collision/len(cost))

    # for ii in range(1):
    #     with open("Data/min_intr_dist_v1_00" + str(ii)+ ".pkl", 'rb') as f:
    #         cost = pickle.load(f)
    #     for tt in range(data_length):
    #         x.append(int(round(tt / step)) * step)
    #         data.append(cost[tt])
    #         style.append("Normal commander")

    # for ii in range(1):
    #     with open("Data/min_intr_dist_v1_02" + str(ii)+ ".pkl", 'rb') as f:
    #         cost = pickle.load(f)
    #     for tt in range(data_length):
    #         x.append(int(round(tt / step)) * step)
    #         data.append(cost[tt])
    #         style.append("No commander")

    # for ii in range(1):
    #     with open("Data/min_intr_dist_v1_2" + str(ii)+ ".pkl", 'rb') as f:
    #         cost = pickle.load(f)
    #     for tt in range(data_length):
    #         x.append(int(round(tt / step)) * step)
    #         data.append(cost[tt])
    #         style.append("No commander")

    # fmri = sns.load_dataset("fmri")
    data = pd.DataFrame(list(zip(x, data, style)), columns=['Episode (k)', 'Collision Rate', 'style'])
    # plt.figure()
    # sns.catplot(x="Episode", y="Total Cost", hue="style", kind="box", data=data)
    sns.relplot(x="Episode (k)", y="Collision Rate", style="style", kind="line", data=data)
    # plt.figure(1)
    # plt.plot(range(len(cost)), cost)
    # plt.ylabel('Total Cost')
    # plt.xlabel('Episodes')
    plt.show()



def main():
    plot_collision_rate(7000, 100)


if __name__ == "__main__":
    main()