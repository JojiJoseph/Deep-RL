import numpy as np

class RolloutBuffer:
    def __init__(self, action_dim, state_dim, size=10_000):
        self.idx = 0
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.size = size
        self.states = np.zeros([size,state_dim])
        self.actions = np.zeros([size, action_dim])
        self.rewards = np.zeros((size,))
        self.next_states = np.zeros([size, state_dim])
        self.dones = np.zeros((size,))
        self.advantages = np.zeros((size,))
        self.returns = np.zeros((size,))
        self.values = np.zeros((size,))

        self.batch_size = 100 # For now it's hardcoded
        self.n_batch_rows = size // self.batch_size

        self.choice_from = [x for x in range(size)]

    def add(self, state, action, reward, next_state, done, val):
        idx = self.idx

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.values[idx] = done

        self.idx+=1
        if self.idx > self.size:
            self.idx = 1

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

    def calc_advatages(self, last_value=0,gamma=0.99, lda=1.0):
        n = self.idx
        prev_adv = 0
        # last_value = last_value.reshape((-1,))
        last_value = 0 # Hardcoded
        for i in range(n-1,-1,-1):
            delta = self.rewards[i] + gamma*last_value*(1-self.dones[i]) - self.values[i]
            adv = delta + lda*gamma*(1-self.dones[i])*prev_adv
            prev_adv = adv
            last_value = self.values[i]
            self.advantages[i] = adv
            self.returns[i] = adv + self.values[i]

    def clear(self):
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self
        
    def __next__(self):
        idx, batch_size , batch_rows= self.idx, self.batch_size, self.n_batch_rows
        if self.idx + self.n_batch_rows<= len(self.states):
            s,a,adv,ret = self.states[idx:idx+batch_rows],self.actions[idx:idx+batch_rows],self.advantages[idx:idx+batch_rows], self.returns[idx:idx+batch_rows]
            self.idx+=batch_rows
            s = s.reshape((-1,self.state_dim))
            a = a.reshape((-1,self.action_dim))
            adv = adv.reshape((-1,))
            ret = adv.reshape((-1,))
            d = self.dones[idx:idx+batch_rows]
            n = self.next_states[idx:idx+batch_rows]
            # l = l.reshape((-1,))
            return s,a,n,d, adv,ret
        else:
            raise StopIteration