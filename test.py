import torch
import numpy as np
import itertools


def main():
    a = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
    b = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
    test = torch.all(a == b)
    list_of_lists = [[0, 1, 2], [3, 4, 5], [7, 8, 9]]
    test = list(itertools.chain.from_iterable(list_of_lists))
    test1 = torch.tensor(test)
    test2 = torch.reshape(test1, (3, 3))
    test3 = torch.repeat_interleave(test2, repeats=3, dim=0)
    x = (0,1)
    y = x
    x = (1, 1)
    x = np.array([[0, 1, 2, 20], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    action_idx = np.argmax(x)
    action = [action_idx // 4, action_idx % 4]
    test = x[0, 3]
    x = torch.tensor([True, True, True, True, False])
    y = torch.tensor([True, False, True, True])
    test = x * y
    test2 = torch.reshape(test, [2, 4]).sum(1)
    print(" ")


if __name__ == "__main__":
    main()