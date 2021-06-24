import numpy as np

class RolloutBuffer:
    def __init__(self, action_dim, state_dim, size=10_000, batch_size=100):
        self.idx = 0
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.size = size
        self.batch_size = batch_size

        assert size % batch_size == 0, "Buffer size should be divisible by batch size"
        
        self.states = np.zeros([size,state_dim])
        self.actions = np.zeros([size, action_dim])
        self.rewards = np.zeros((size,))
        self.next_states = np.zeros([size, state_dim])
        self.dones = np.zeros((size,))
        self.advantages = np.zeros((size,))
        self.returns = np.zeros((size,))
        self.values = np.zeros((size,))

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
            self.idx = 0

    def calc_advatages(self, last_value=0,gamma=0.99, lda=0.95):
        n = self.idx
        prev_adv = 0 # Hardcoded
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
        idx, batch_size = self.idx, self.batch_size
        if self.idx + self.batch_size<= len(self.states):
            state_batch = self.states[idx:idx+batch_size]
            action_batch = self.actions[idx:idx+batch_size]
            adv_batch = self.advantages[idx:idx+batch_size]
            ret_batch = self.returns[idx:idx+batch_size]
            done_batch = self.dones[idx:idx+batch_size]
            next_batch = self.next_states[idx:idx+batch_size]
            
            state_batch = state_batch.reshape((-1,self.state_dim))
            action_batch = action_batch.reshape((-1,self.action_dim))
            adv_batch = adv_batch.reshape((-1,))
            ret_batch = ret_batch.reshape((-1,))
            self.idx+=batch_size

            return state_batch,action_batch,next_batch,done_batch, adv_batch,ret_batch
        else:
            raise StopIteration