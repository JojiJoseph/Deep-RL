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
        self.v = nn.Linear(size, 1)
        self.a = nn.Linear(size, n_actions)

        torch.nn.init.uniform_(self.l1.weight.data, -0.001, 0.001)
        torch.nn.init.uniform_(self.l2.weight.data, -0.001, 0.001)
        torch.nn.init.uniform_(self.v.weight.data, -0.001, 0.001)
        torch.nn.init.uniform_(self.a.weight.data, -0.001, 0.001)

    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        val = self.v(y)
        adv = self.a(y)
        q = val + adv - torch.mean(adv, axis=-1, keepdims=True)
        return q

    def get_action(self, x, eval=False):
        y = self.forward(x)
        return torch.argmax(y, axis=-1)
