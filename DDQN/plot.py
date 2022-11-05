import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Deque
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True,
                    help="The experiment name as defined in the yaml file")

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()
experiment = args.exp

log_filename = f"./results/{experiment}.csv"

df = pd.read_csv(log_filename)

episodes = df.values[:, 0]
timesteps = df.values[:, 1]
returns = df.values[:, 2]

plt.plot(episodes, returns)
plt.title("Returns vs Episodes")
plt.ylabel("Return")
plt.xlabel("Episode")
plt.grid()
plt.show()

plt.plot(timesteps, returns)
plt.title("Returns vs Timesteps")
plt.ylabel("Return")
plt.xlabel("Timestep")
plt.grid()
plt.show()


def smooth(y, q_size=100):
    q = Deque(maxlen=q_size)
    y_avg = []
    for item in y:
        q.append(item)
        y_avg.append(np.mean(q))
    return y_avg


returns_averaged = smooth(returns)

plt.plot(episodes, returns_averaged)
plt.title("Returns vs Episodes")
plt.ylabel("Return (Moving Avg.)")
plt.xlabel("Episode")
plt.grid()
plt.show()

plt.plot(timesteps, returns_averaged)
plt.title("Returns vs Timesteps")
plt.ylabel("Return (Moving Avg.)")
plt.xlabel("Timestep")
plt.grid()
plt.show()
