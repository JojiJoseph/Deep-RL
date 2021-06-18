import torch
import torch.nn as nn
import numpy as np
import gym
from copy import deepcopy
import pybullet_envs

from net import Actor, Critic
from buffer import ReplayBuffer

class SAC:
    def __init__(self,namespace="actor",resume=False,env_name="Pendulum", action_scale=1, alpha=0.2, learning_rate=3e-4):
        self.env_name = env_name
        self.namespace = namespace
        self.action_scale = action_scale
        self.alpha = alpha
        self.learning_rate = learning_rate
    def learn(self):
        env_name = self.env_name
        TAU = 0.005
        env = gym.make(env_name)

        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]

        # Actor and Critics
        actor = Actor(state_dim, action_dim)

        critic_1 = Critic(state_dim, action_dim)
        critic_2 = Critic(state_dim, action_dim)


        critic_target_1 = deepcopy(critic_1)
        critic_target_2 = deepcopy(critic_2)
        
        # Leave the following comments
        # critic_target_1 = Critic(state_dim, action_dim)
        # critic_target_2 = Critic(state_dim, action_dim)
        # critic_target_1.load_state_dict(critic_1.state_dict())
        # critic_target_2.load_state_dict(critic_2.state_dict())

        opt_actor  = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        opt_c1  = torch.optim.Adam(critic_1.parameters(), lr=self.learning_rate)
        opt_c2  = torch.optim.Adam(critic_2.parameters(), lr=self.learning_rate)

        BUFFER_SIZE = 10_000
        BATCH_SIZE = 200
        buffer = ReplayBuffer(action_dim, state_dim, BUFFER_SIZE)

        N_TIMESTEPS = 1_000_000
        UPDATE_EVERY = 50
        EVALUATE_EVERY = 10_000
        ALPHA = self.alpha
        GAMMA = 0.99

        ACTION_SCALE = self.action_scale

        timestep = 0

        env.seed(0)
        _state = env.reset()
        total_reward = 0
        episodic_reward = 0
        episodes_passed = 0

        ep_len = 0
        highscore = -np.inf
        episode_steps = 0
        while timestep < N_TIMESTEPS:
            timestep += 1
            state = torch.from_numpy(_state[None,:]).float()
            with torch.no_grad():
                action, _ = actor.get_action(state)
            action = action[0].detach().numpy()

            if timestep < BUFFER_SIZE:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action*ACTION_SCALE)
            total_reward += reward
            episodic_reward += reward
                
            episode_steps += 1

            buffer.add(_state.copy(), action.copy(), reward, next_state.copy(),not(done and episode_steps == env._max_episode_steps) and done)

            _state = next_state
            if done:
                print(f"Episode: {episodes_passed}, reward: {episodic_reward}")
                episodic_reward = 0
                episodes_passed += 1
                episode_steps = 0
                _state = env.reset()

            if timestep % UPDATE_EVERY == 0 and timestep > BUFFER_SIZE:
                for i in range(UPDATE_EVERY):
                    state_batch, action_batch, reward_batch, next_batch, done_batch = buffer.get_batch(BATCH_SIZE)
                    
                    state_batch = torch.from_numpy(state_batch).float()
                    action_batch = torch.from_numpy(action_batch).float()
                    reward_batch = torch.from_numpy(reward_batch).float()
                    next_batch = torch.from_numpy(next_batch).float()
                    done_batch = torch.from_numpy(done_batch).long()

                    with torch.no_grad():
                        next_actions, log_prob = actor.get_action(next_batch)
                        predicted_q = torch.minimum(critic_target_1(next_batch, next_actions),critic_target_2(next_batch, next_actions))
                        target = reward_batch[:,None] + GAMMA*(1-done_batch[:,None])*(predicted_q - ALPHA*log_prob[:,None].detach())

                    # Update critics
                    loss = (target.flatten().detach()- critic_1(state_batch,action_batch).flatten())**2
                    opt_c1.zero_grad()
                    loss = loss.mean()
                    loss.backward()
                    opt_c1.step()

                    loss = (target.flatten().detach()- critic_2(state_batch,action_batch).flatten())**2
                    loss = loss.mean()
                    opt_c2.zero_grad()
                    loss.backward()
                    opt_c2.step()

                    # Update actor
                    actions, log_prob = actor.get_action(state_batch)
                    predicted_q = torch.minimum(critic_1(state_batch,actions).flatten(), critic_2(state_batch,actions).flatten())#.detach()
                    loss = predicted_q - ALPHA*log_prob.flatten()
                    loss = -loss.mean()
                    opt_actor.zero_grad()
                    loss.backward()
                    opt_actor.step()

                    with torch.no_grad():
                        for pt, p in zip(critic_target_1.parameters(), critic_1.parameters()):
                            pt.data.mul_(1-TAU)
                            pt.data.add_(TAU*p.data)
                        for pt, p in zip(critic_target_2.parameters(), critic_2.parameters()):
                            pt.data.mul_(1-TAU)
                            pt.data.add_(TAU*p.data)
            
            if timestep % EVALUATE_EVERY == 0:
                eval_env = gym.make(env_name)
                total = 0
                for episode in range(10):
                    state = eval_env.reset()
                    done = False
                    while not done:
                        with torch.no_grad():
                            state = state[None,:]
                            state = torch.from_numpy(state).float()
                            action, _ = actor.get_action(state)
                            action = action[0].detach().cpu().numpy()
                            state, reward, done,_ = eval_env.step(action*ACTION_SCALE)
                            total += reward
                

                avg = total/10
                if avg >= highscore:
                    highscore = avg
                    torch.save(actor.state_dict(), f"./{self.namespace}.pt")
                print("highscore:", highscore)
                eval_env.close()

