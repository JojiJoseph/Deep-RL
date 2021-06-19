import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, size=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim, size)
        self.l2 = nn.Linear(size, size)
        self.mu = nn.Linear(size, action_dim)

        self.normal = torch.distributions.Normal(0,0.1)

    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        mu = self.mu(y)
        return mu
    def get_action(self, x, eval=False):
        action = self.forward(x)
        if eval:
            return action
        return action# + self.normal.sample((1,)), -1, 1)



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, size=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, 1)
    def forward(self, s, a):
        x = torch.cat((s, a),dim=-1)

        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)

        return y