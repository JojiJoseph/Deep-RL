import torch
import torch.nn as nn
import numpy as np
import gym
import pybullet_envs
import csv

from net import Actor,ActorDiscrete,Critic
from buffer import RolloutBuffer
from logger import Logger

class VPG:
    def __init__(self,namespace="actor",resume=False,env_name="Pendulum", action_scale=1, learning_rate=3e-4,
    gamma=0.99, n_eval_episodes=10, evaluate_every=10_000, buffer_size=10_000, n_timesteps=1_000_000,
    batch_size=100,lda=1.0):
        self.env_name = env_name
        self.namespace = namespace
        self.action_scale = action_scale
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_eval_episodes = n_eval_episodes
        self.evaluate_every = evaluate_every
        self.buffer_size = buffer_size
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.lda = lda
        self.simple_log = True # Hardcoded for now
    def learn(self):
        env_name = self.env_name

        env = gym.make(env_name)

        state_dim = env.observation_space.shape[0]
        action_dim = None
        n_actions = None

        # Actor and Critic
        if type(env.action_space) == gym.spaces.Discrete:
            n_actions = env.action_space.n
            actor = ActorDiscrete(state_dim, n_actions)
        else:
            action_dim = env.action_space.shape[0]
            actor = Actor(state_dim, action_dim)

        critic = Critic(state_dim, None)
        
        opt_actor  = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        opt_critic  = torch.optim.Adam(critic.parameters(), lr=self.learning_rate)


        buffer = RolloutBuffer(action_dim or 1, state_dim, size=self.buffer_size, batch_size=self.batch_size)

        N_TIMESTEPS = self.n_timesteps

        ACTION_SCALE = self.action_scale

        timestep = 0

        _state = env.reset()

        episodic_reward = 0
        episodes_passed = 0

        # Setup the CSV
        log_filename = f"./results/{self.namespace}.csv"
        log_data = [["Episode", "End Step", "Episodic Reward"]]

        highscore = -np.inf
        episode_steps = 0

        while timestep < N_TIMESTEPS:
            timestep += 1
            state = torch.from_numpy(_state[None,:]).float()
            with torch.no_grad():
                action, _ = actor.get_action(state)
                action = action[0].detach().cpu().numpy()
                val = critic(state)[0].cpu().numpy()
            if type(env.action_space) == gym.spaces.Discrete:
                action_clipped = action[0]
            else:
                action_clipped  = np.clip(action, -1, 1)
            next_state, reward, done, _ = env.step(action_clipped*ACTION_SCALE)

            episodic_reward += reward
                
            episode_steps += 1

            buffer.add(_state.copy(), action.copy(), reward, next_state.copy(),not(done and episode_steps == env._max_episode_steps) and done, val)

            _state = next_state
            if done:
                episodes_passed += 1
                log_data.append([episodes_passed, timestep, episodic_reward])
                if self.simple_log:
                    print(f"Episode: {episodes_passed}, return: {episodic_reward}, timesteps elapsed: {timestep}")
                else:
                    Logger.print_boundary()
                    Logger.print("Episode", episodes_passed)
                    Logger.print("Episodic Reward", episodic_reward)
                    Logger.print("Timesteps", timestep)
                    Logger.print_boundary()

                episodic_reward = 0
                episode_steps = 0
                _state = env.reset()

            done = not(done and episode_steps == env._max_episode_steps) and done

            if timestep % self.buffer_size == 0:
                val = 0 if done else critic(torch.from_numpy(_state[:]).float())[0].detach().cpu().numpy().item()
                buffer.calc_advatages(gamma=self.gamma,lda=self.lda, last_value=val)
                for batch in buffer:
                    state_batch, action_batch, next_batch, done_batch, adv_batch, ret_batch = batch
                    
                    state_batch = torch.from_numpy(state_batch).float()
                    if type(env.action_space) == gym.spaces.Discrete:
                        action_batch = torch.from_numpy(action_batch).long()
                    else:
                        action_batch = torch.from_numpy(action_batch).float()

                    next_batch = torch.from_numpy(next_batch).float()
                    done_batch = torch.from_numpy(done_batch).long()
                    adv_batch = torch.from_numpy(adv_batch).float()
                    ret_batch = torch.from_numpy(ret_batch).float()

                    # Update critic
                    loss = (ret_batch.flatten().detach()- critic(state_batch).flatten())**2
                    opt_critic.zero_grad()
                    loss = loss.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5) # Leave this comment as it is
                    opt_critic.step()

                    # Update actor
                    log_prob = actor.log_prob(action_batch, state_batch)

                    # Normalize advantages
                    adv_batch = (adv_batch - adv_batch.mean())/(adv_batch.std() + 1e-9)
                    
                    loss =  log_prob * adv_batch
                    loss = -loss.mean()
                    
                    opt_actor.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) # Leave this comment as it is
                    opt_actor.step()

                buffer.clear()
            
            if timestep % self.evaluate_every == 0 and timestep > 0:

                eval_env = gym.make(env_name)

                eval_returns = []
                for episode in range(self.n_eval_episodes):
                    state = eval_env.reset()
                    done = False
                    eval_return = 0
                    while not done:
                        with torch.no_grad():
                            state = state[None,:]
                            state = torch.from_numpy(state).float()
                            action, _ = actor.get_action(state, eval=True)
                            action = action[0].detach().cpu().numpy()
                            if type(env.action_space) == gym.spaces.Discrete:
                                action_clipped = action[0]
                            else:
                                action_clipped  = np.clip(action, -1, 1)
                            state, reward, done,_ = eval_env.step(action_clipped*ACTION_SCALE)
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
                    torch.save(actor.state_dict(), f"./results/{self.namespace}.pt")
                    print("New High (Avg) Score! Saved!")
                print(f"highscore (Avg. of eval trials): {highscore}\n")
                eval_env.close()

                # Save log
                with open(log_filename,'w',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(log_data)

        print("\nTraining is Over!\n")

