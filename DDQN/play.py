import torch
import torch.nn as nn
import numpy as np
import gym
import time
import pybullet_envs
import argparse
import yaml
from gym.wrappers import Monitor

from net import Net

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True,
                    help="The experiment name as defined in the yaml file")
parser.add_argument("-E", "--eval", action="store_true", help="Evaluation run")

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()
experiment = args.exp
eval = args.eval
hyperparams = experiments[experiment]
fps = 60

if not eval:
    eval_env = Monitor(gym.make(
        hyperparams['env_name']), './results/video/{}_{}'.format(experiment, time.time()), force=True)
else:
    eval_env = gym.make(hyperparams['env_name'])

if "action_scale" in hyperparams:
    ACTION_SCALE = hyperparams["action_scale"]
else:
    ACTION_SCALE = 1

state_dim = eval_env.observation_space.shape[0]
n_actions = None
action_dim = None

n_actions = eval_env.action_space.n
actor = Net(state_dim, n_actions)

actor.load_state_dict(torch.load(f"./results/{experiment}.pt"))
try:
    if not eval:
        eval_env.render()
except:
    pass

n_episodes = 100 if eval else 1

returns = []
for episode in range(n_episodes):
    state = eval_env.reset()
    done = False
    total_return = 0
    while not done:
        with torch.no_grad():
            state = state[None, :]
            state = torch.from_numpy(state).float()
            action = actor.get_action(state, eval=True)
            action = action[0].detach().cpu().numpy()
            if type(eval_env.action_space) == gym.spaces.Discrete:
                action_clipped = action
            else:
                action_clipped = np.clip(action, -1, 1)
            if not eval:
                eval_env.render()
                time.sleep(1 / fps)
            state, reward, done, _ = eval_env.step(action_clipped * ACTION_SCALE)
            total_return += reward
    returns.append(total_return)
    print(f"Episode: {episode+1}, Return: {total_return}")
    total_return = 0

print("best", np.max(returns))
print("mean:", np.mean(returns))
print("std:", np.std(returns))
