<<<<<<< HEAD
This repository implements our algorithm.

env_back.yml includes conda env for this repo.
``` Bash
# create conda environment
conda env create -f env_back.yml
```

custom_envs includes modified environments for training.

please use commands below for training:
```
python scripts/train_multi_pri.py
```
```
python scripts/train_mpe_pri.py
```


=======
# etcpriority

Repository for **Priority-Driven Control and Communication in Decentralized Multi-Agent Systems via Reinforcement Learning**.

This repository contains the codebase for training and evaluating reinforcement learning agents in decentralized multi-agent systems with priority-driven control and communication mechanisms.

## Environment Setup

The Python environment used in this project is provided in `env.yml`.

Create the Conda environment with:

```bash
conda env create -f env.yml
conda activate etcpriority
```

## Training and Evaluation

To start training, run:

```bash
python scripts/train_multi_pri.py \
  --user_name your_wandb \
  --ppo_epoch 15 \
  --critic_hidden_size 256 \
  --actor_hidden_size 256 \
  --lr 7e-4 \
  --layer_N 1 \
  --gamma 0.99 \
  --critic_lr 7e-4 \
  --clip_param 0.2 \
  --gae_lambda 0.95 \
  --algorithm_name ippo \
  --n_training_threads 2 \
  --n_rollout_threads 14 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps 10000000 \
  --n_trajectories 15
```

To evaluate the trained models, run:

```bash
python eval_walker.py
```
>>>>>>> 670cb287b30c72a9855f8615720a54e7a5d1631e
