import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Deque
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-e","--exp",type=str, required=True,help="The experiment name as defined in the yaml file")

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()
experiment = args.exp

log_filename = f"{experiment}.csv"

df = pd.read_csv(log_filename)

episodes = df.values[:,0]
timesteps = df.values[:,1]
returns = df.values[:,2]

plt.plot(episodes, returns)
plt.show()

plt.plot(timesteps, returns)
plt.show()