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
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

        self.normal = torch.distributions.Normal(0,0.1)

    def forward(self, x):
        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        mu = torch.tanh(self.mu(y))
        return mu, self.log_std
    def get_action(self, x, eval=False):
        mu, log_std = self.forward(x)
        distrib = torch.distributions.Normal(mu, torch.exp(log_std))
        if eval:
            log_prob = distrib.log_prob(mu).sum(-1)
            return mu, log_prob
        action = distrib.sample((1,))[0]
        # print(action.shape)
        log_prob = distrib.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob(self, action, state):
        mu, log_std = self.forward(state)
        distrib = torch.distributions.Normal(mu, torch.exp(log_std))
        return distrib.log_prob(action).sum(-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, size=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, size)
        self.l2 = nn.Linear(size, size)
        self.l3 = nn.Linear(size, 1)
    def forward(self, x):
        # x = torch.cat((s, a),dim=-1)

        y = torch.relu(self.l1(x))
        y = torch.relu(self.l2(y))
        y = self.l3(y)

        return y