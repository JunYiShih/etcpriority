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
custom_envs includes modified environments for training.

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
