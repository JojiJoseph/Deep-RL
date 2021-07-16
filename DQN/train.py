import argparse
import yaml
import pybullet_envs
from dqn import DQN

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True,
                    help="The experiment name as defined in the yaml file")
parser.add_argument("-r", "--resume", action="store_true",
                    help="Resume the process from previous check point")  # TODO

with open("./experiments.yaml") as f:
    experiments = yaml.safe_load(f)

args = parser.parse_args()

print(args)

experiment = args.exp
hyperparams = experiments[experiment]

print(hyperparams)

algo = DQN(namespace=experiment, resume=args.resume, **hyperparams)

algo.learn()
