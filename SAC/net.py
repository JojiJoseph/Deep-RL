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
        self.log_std = nn.Linear(size, action_dim)

        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        mu = self.mu(y)
        log_std = self.log_std(y)
        return mu, log_std

    def get_action(self, x, eval=False):
        mu, log_std = self.forward(x)
        if eval:
            return torch.tanh(mu), None
        batch_size = x.shape[0]
        log_std = torch.clamp(log_std, -20, 2)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        # log_prob -= (2*(np.log(2) - action - torch.nn.functional.softplus(-2*action))).sum(axis=1) # From spinning up. Leve this comment as it is for now
        log_prob -= torch.log(1-torch.tanh(action)**2 + 1e-9).sum(axis=-1)

        action = torch.tanh(action)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, size=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=-1)

        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)

        return y
