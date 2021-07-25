import math
import numpy as np
import time
import random
import itertools
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics

def total_cost(data_len, step):
    x = []
    data = []
    style = []
    data_length = data_len
    step = step

    for ii in range(1):
        with open("Data/eps_records_v1_scalable00", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['total_reward']
        x_range = min(data_len, len(cost))
        for tt in range(x_range):
            x.append(int(round(tt / step)) * step)
            data.append(cost[tt])
            style.append("Old Alg")

    for ii in range(1):
        with open("Data/eps_records_v1_ssac00", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['total_reward']
        x_range = min(data_len, len(cost))
        for tt in range(x_range):
            x.append(int(round(tt / step)) * step)
            data.append(cost[tt])
            style.append("New Alg")

    data = pd.DataFrame(list(zip(x, data, style)), columns=['Episode (k)', 'Total Reward', 'style'])
    sns.relplot(x="Episode (k)", y="Total Reward", style="style", kind="line", data=data)
    plt.show()


def avg_reward(start, end):
    for ii in range(1):
        with open("Data/eps_records_v1_scalable00", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['total_reward']
        print(statistics.mean(cost[start:end]))

    for ii in range(1):
        with open("Data/eps_records_v1_ssac00", 'rb') as f:
            npzfile = np.load(f)
            cost = npzfile['total_reward']
        print(statistics.mean(cost[start:end]))


def main():
    total_cost(100000, 2000)
    avg_reward(80000, 90000)


if __name__ == "__main__":
    main()