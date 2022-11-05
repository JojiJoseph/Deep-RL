import numpy as np


class ReplayBuffer:
    def __init__(self, action_dim, state_dim, size=10_000):
        self.idx = 0
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.size = size
        self.states = np.zeros([size, state_dim])
        self.actions = np.zeros([size, action_dim])
        self.rewards = np.zeros((size,))
        self.next_states = np.zeros([size, state_dim])
        self.dones = np.zeros((size,))

        self.choice_from = [x for x in range(size)]

    def add(self, state, action, reward, next_state, done):
        idx = self.idx

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0

    def get_batch(self, batch_size=128, rg=None):
        if rg is None:
            indices = np.random.choice(self.choice_from, batch_size)
        else:
            indices = np.random.choice(self.choice_from[:rg], batch_size)

        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_batch = self.next_states[indices]
        done_batch = self.dones[indices]

        return state_batch, action_batch, reward_batch, next_batch, done_batch
