import torch
import torch.nn as nn
import numpy as np
import gym
from copy import deepcopy
import pybullet_envs
import csv

from net import Actor, Critic
from buffer import ReplayBuffer
from logger import Logger

class DDPG:
    def __init__(self,namespace="actor",resume=False,env_name="Pendulum", action_scale=1, alpha=0.2, learning_rate=3e-4,
    gamma=0.99, tau=0.005, n_eval_episodes=10, evaluate_every=10_000, update_every=50, buffer_size=10_000, n_timesteps=1_000_000,
    batch_size=100):
        self.env_name = env_name
        self.namespace = namespace
        self.action_scale = action_scale
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_eval_episodes = n_eval_episodes
        self.evaluate_every = evaluate_every
        self.update_every = update_every
        self.buffer_size = buffer_size
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
    def learn(self):
        env_name = self.env_name
        TAU = self.tau
        env = gym.make(env_name)

        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]

        # Actor and Critics
        actor = Actor(state_dim, action_dim)

        critic = Critic(state_dim, action_dim)

        actor_target = deepcopy(actor)
        critic_target = deepcopy(critic)
        
        opt_actor  = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        opt_critic  = torch.optim.Adam(critic.parameters(), lr=self.learning_rate)

        BUFFER_SIZE = self.buffer_size
        BATCH_SIZE = self.batch_size
        buffer = ReplayBuffer(action_dim, state_dim, BUFFER_SIZE)

        N_TIMESTEPS = self.n_timesteps
        UPDATE_EVERY = self.update_every
        EVALUATE_EVERY = self.evaluate_every
        ALPHA = self.alpha
        GAMMA = self.gamma

        ACTION_SCALE = self.action_scale

        timestep = 0

        env.seed(0)
        _state = env.reset()
        total_reward = 0
        episodic_reward = 0
        episodes_passed = 0

        # Setup the CSV
        log_filename = f"./{self.namespace}.csv"
        log_data = [["Episode", "End Step", "Episodic Reward"]]

        ep_len = 0
        highscore = -np.inf
        episode_steps = 0
        normal = torch.distributions.Normal(0,0.1)
        while timestep < N_TIMESTEPS:
            timestep += 1
            state = torch.from_numpy(_state[None,:]).float()
            with torch.no_grad():
                action = actor.get_action(state)
                action  = np.clip(actor.get_action(state) + normal.sample((1,)), -1, 1)
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
                episodes_passed += 1
                log_data.append([episodes_passed, timestep, episodic_reward])

                Logger.print_boundary()
                Logger.print("Episode", episodes_passed)
                Logger.print("Episodic Reward", episodic_reward)
                Logger.print("Timesteps", timestep)
                Logger.print_boundary()

                episodic_reward = 0
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
                        next_actions = np.clip(actor_target.get_action(next_batch) + np.clip(normal.sample((self.batch_size,1)),-0.5,0.5), -1, 1)
                        predicted_q = critic_target(next_batch, next_actions)
                        target = reward_batch[:,None] + GAMMA*(1-done_batch[:,None])*predicted_q

                    # Update critic
                    loss = (target.flatten().detach()- critic(state_batch,action_batch).flatten())**2
                    opt_critic.zero_grad()
                    loss = loss.mean()
                    loss.backward()
                    opt_critic.step()

                    # Update actor
                    actions = actor.get_action(state_batch)
                    predicted_q = critic(state_batch, actions).flatten()
                    loss = predicted_q
                    loss = -loss.mean()
                    opt_actor.zero_grad()
                    loss.backward()
                    opt_actor.step()

                    with torch.no_grad():
                        if i % 2 == 0:
                            for pt, p in zip(actor_target.parameters(), actor.parameters()):
                                pt.data.mul_(1-TAU)
                                pt.data.add_(TAU*p.data)
                        for pt, p in zip(critic_target.parameters(), critic.parameters()):
                            pt.data.mul_(1-TAU)
                            pt.data.add_(TAU*p.data)
            
            if timestep % EVALUATE_EVERY == 0 and timestep > BUFFER_SIZE:
                # print("\nEvaluation\n==========")
                eval_env = gym.make(env_name)
                total = 0
                eval_returns = []
                for episode in range(self.n_eval_episodes):
                    state = eval_env.reset()
                    done = False
                    eval_return = 0
                    while not done:
                        with torch.no_grad():
                            state = state[None,:]
                            state = torch.from_numpy(state).float()
                            action = actor.get_action(state, eval=True)
                            action = action[0].detach().cpu().numpy()
                            state, reward, done,_ = eval_env.step(action*ACTION_SCALE)
                            eval_return += reward
                        if done:
                            eval_returns.append(eval_return)

                avg = np.mean(eval_returns)
                std = np.std(eval_returns)
                best = np.max(eval_returns)
                worst = np.min(eval_returns)

                Logger.print_boundary()
                Logger.print_title("Evaluation")
                Logger.print_double_boundary()
                Logger.print("Eval Episodes", self.n_eval_episodes)
                Logger.print("Avg", avg)
                Logger.print("Std", std)
                Logger.print("Best", best)
                Logger.print("Worst", worst)
                Logger.print_boundary()

                if avg >= highscore:
                    highscore = avg
                    torch.save(actor.state_dict(), f"./{self.namespace}.pt")
                    print("New High (Avg) Score! Saved!")
                print(f"highscore: {highscore}\n")
                eval_env.close()

                # Save log
                with open(log_filename,'w',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(log_data)

        print("\nTraining is Over!\n")

