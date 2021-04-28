import random
from collections import namedtuple


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return self.memory
        else:
            return random.sample(self.memory, batch_size)

    def clear_memory(self):
        self.memory = []
        self.position = 0

    def return_all(self):
        return self.memory

    def return_len(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)