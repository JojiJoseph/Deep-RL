import os
from copy import deepcopy
import csv
import torch
import torch.nn as nn
import numpy as np
import gym
import pybullet_envs

from net import Net
from buffer import ReplayBuffer
from logger import Logger


class DQN:
    def __init__(self, namespace="actor", resume=False, env_name="Pendulum", learning_rate=3e-4,
                 gamma=0.99, n_eval_episodes=10, evaluate_every=10_000, update_every=50,
                 buffer_size=10_000, n_timesteps=1_000_000, batch_size=100, epsilon=0.2, simple_log=True):
        self.env_name = env_name
        self.namespace = namespace
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_eval_episodes = n_eval_episodes
        self.evaluate_every = evaluate_every
        self.update_every = update_every
        self.buffer_size = buffer_size
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.simple_log = simple_log

    def learn(self):
        env_name = self.env_name
        env = gym.make(env_name)

        n_actions = env.action_space.n
        state_dim = env.observation_space.shape[0]

        # DQN network
        agent = Net(state_dim, n_actions)
        agent_target = deepcopy(agent)

        optim = torch.optim.Adam(agent.parameters(), lr=self.learning_rate)

        buffer = ReplayBuffer(1, state_dim, self.buffer_size)

        timestep = 0

        _state = env.reset()
        total_reward = 0
        episodic_reward = 0
        episodes_passed = 0

        # Setup the CSV
        # Create folder if doesn't exist
        os.makedirs("./results", exist_ok=True)
        log_filename = f"./results/{self.namespace}.csv"
        log_data = [["Episode", "End Step", "Episodic Reward"]]

        highscore = -np.inf
        episode_steps = 0

        while timestep < self.n_timesteps:
            timestep += 1
            state = torch.from_numpy(_state[None, :]).float()
            with torch.no_grad():
                action = agent.get_action(state)
            action = action[0].detach().numpy()
            if np.random.random() < self.epsilon:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            episodic_reward += reward

            episode_steps += 1

            buffer.add(_state, action, reward, next_state, not(
                done and episode_steps == env._max_episode_steps) and done)

            _state = next_state
            if done:
                episodes_passed += 1
                log_data.append([episodes_passed, timestep, episodic_reward])

                if self.simple_log:
                    print(
                        f"Episode: {episodes_passed}, return: {episodic_reward}, timesteps elapsed: {timestep}")
                else:
                    Logger.print_boundary()
                    Logger.print("Episode", episodes_passed)
                    Logger.print("Episodic Reward", episodic_reward)
                    Logger.print("Timesteps", timestep)
                    Logger.print_boundary()

                episodic_reward = 0
                episode_steps = 0
                _state = env.reset()

            if timestep % self.update_every == 0 and timestep > self.buffer_size:
                for i in range(self.update_every):
                    state_batch, action_batch, reward_batch, next_batch, done_batch = buffer.get_batch(
                        self.batch_size)

                    state_batch = torch.from_numpy(state_batch).float()
                    action_batch = torch.from_numpy(action_batch).long()
                    reward_batch = torch.from_numpy(reward_batch).float()
                    next_batch = torch.from_numpy(next_batch).float()
                    done_batch = torch.from_numpy(done_batch).long()

                    with torch.no_grad():
                        max_values, _ = torch.max(agent_target(
                            next_batch), dim=-1, keepdim=True)
                        target = reward_batch[:, None] + self.gamma * \
                            (1 - done_batch[:, None]) * max_values

                    prediction = torch.sum(agent(
                        state_batch) * (torch.eye(n_actions)[action_batch.flatten()]), dim=-1, keepdim=True)

                    loss = (target - prediction)**2
                    optim.zero_grad()
                    loss = loss.mean()
                    loss.backward()

                    optim.step()

            if timestep % (10 * self.update_every) == 0 and timestep > self.buffer_size:
                agent_target.load_state_dict(agent.state_dict())
            if timestep % self.evaluate_every == 0 and timestep > self.buffer_size:

                eval_env = gym.make(env_name)

                eval_returns = []
                for episode in range(self.n_eval_episodes):
                    state = eval_env.reset()
                    done = False
                    eval_return = 0
                    while not done:
                        with torch.no_grad():
                            state = state[None, :]
                            state = torch.from_numpy(state).float()
                            action = agent.get_action(state, eval=True)
                            action = action[0].detach().cpu().numpy()
                            state, reward, done, _ = eval_env.step(action)
                            eval_return += reward
                        if done:
                            eval_returns.append(eval_return)

                eval_avg = np.mean(eval_returns)
                eval_std = np.std(eval_returns)
                eval_best = np.max(eval_returns)
                eval_worst = np.min(eval_returns)

                Logger.print_boundary()
                Logger.print_title("Evaluation")
                Logger.print_double_boundary()
                Logger.print("Eval Episodes", self.n_eval_episodes)
                Logger.print("Avg", eval_avg)
                Logger.print("Std", eval_std)
                Logger.print("Best", eval_best)
                Logger.print("Worst", eval_worst)
                Logger.print_boundary()

                if eval_avg >= highscore:
                    highscore = eval_avg
                    torch.save(agent.state_dict(),
                               f"./results/{self.namespace}.pt")
                    print("New High (Avg) Score! Saved!")
                print(f"highscore: {highscore}\n")
                eval_env.close()

                # Save log
                with open(log_filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(log_data)

        print("\nTraining is Over!\n")
