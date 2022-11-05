import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, state_dim, n_actions, size=256):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.l1 = nn.Linear(state_dim, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, n_actions)

        torch.nn.init.uniform_(self.l1.weight.data, -0.001, 0.001)
        torch.nn.init.uniform_(self.l2.weight.data, -0.001, 0.001)
        torch.nn.init.uniform_(self.l3.weight.data, -0.001, 0.001)

    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)
        return y

    def get_action(self, x, eval=False):
        y = self.forward(x)
        return torch.argmax(y, axis=-1)
