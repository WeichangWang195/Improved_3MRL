from collections import namedtuple
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class agentActor_binary(nn.Module):
    def __init__(self, nb_states, nb_obs, output, hidden1=400, hidden2=300, init_w=3e-3):
        super(agentActor_binary, self).__init__()
        self.fc1 = nn.Linear(nb_states + nb_obs, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_head = nn.Linear(hidden2, output)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc_head.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs, xo):
        if (xs != xs).any() or (xo != xo).any():
            print("Error: NaN")
        out = torch.cat((xs, xo), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        action_scores = self.fc_head(out)
        return F.softmax(action_scores, dim=-1)


class agentCritic_binary(nn.Module):
    def __init__(self, nb_states, nb_obs, hidden1=400, hidden2=300, init_w=3e-3):
        super(agentCritic_binary, self).__init__()
        self.fc1 = nn.Linear(nb_states + nb_obs, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_head = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc_head.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs, xo):
        if (xs != xs).any() or (xo != xo).any():
            print("Error: NaN")
        out = torch.cat((xs, xo), 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        state_values = self.fc_head(out)
        return state_values


class agentActor_CNN(nn.Module):
    def __init__(self, layers, h, w, l, output, hidden_size=300, init_w=3e-3):
        super(agentActor_CNN, self).__init__()
        self.conv1 = nn.Conv2d(layers, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=2):
            return (size + 2 * padding - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)), stride=2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)), stride=2)
        linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)
        self.hid_linear = nn.Linear(linear_input_size + l, hidden_size)
        self.action_head = nn.Linear(hidden_size, output)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.conv3.weight.data = fanin_init(self.conv3.weight.data.size())
        self.hid_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, s, x):
        if (x != x).any() or (s != s).any():
            print("Error: NaN")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feature = F.relu(self.hid_linear(torch.cat((x.view(x.size(0), -1), s), 1)))
        action_scores = self.action_head(feature)
        return F.softmax(action_scores, dim=-1)


class agentCritic_CNN(nn.Module):
    def __init__(self, layers, h, w, l, hidden_size=300, init_w=3e-3):
        super(agentCritic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(layers, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=2):
            return (size + 2 * padding - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)), stride=2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)), stride=2)
        linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)
        self.hid_linear = nn.Linear(linear_input_size + l, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.conv3.weight.data = fanin_init(self.conv3.weight.data.size())
        self.hid_linear.weight.data.uniform_(-init_w, init_w)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, s, x):
        if (x != x).any() or (s != s).any():
            print("Error: NaN")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feature = F.relu(self.hid_linear(torch.cat((x.view(x.size(0), -1), s), 1)))
        state_values = self.value_head(feature)
        return state_values