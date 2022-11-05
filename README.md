# Deep-RL
Contains deep reinforcement learning algorithms I have implemented.

## How to run an experiment

Go to the folder of the algorithm.

Then execute the following. See `experiments.yaml` to get `experiment_name`.

```bash
python train.py -e experiment_name
```

To run the saved agent,

```bash
python play.py -e experiment_name
```

To plot the learning curves,

```bash
python plot.py -e experiment_name
```

## Status


| Algorithm | Usable?        | TODO                            |
| --------- | -------------- | ------------------------------- |
| SAC       | Yes            |                                 |
| TD3       | Yes            |                                 |
| DDPG      | Yes            |                                 |
| VPG       | Yes (Discrete) | Test on continuous action space |
| DQN       | Yes            |                                 |
| DDQN      | Yes           |                                 |