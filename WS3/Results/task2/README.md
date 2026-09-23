# Task 2 training results

All eleven experiments specified across Task 2 Questions 1-3 were trained for
1,000 CartPole-v1 episodes with seed 0. Evaluation used a greedy policy every
10 episodes. No plots are included yet.

Each run directory contains:

- `evaluation_returns.csv`: episode, environment step, and evaluation return
- `training_returns.csv`: per-episode training return
- `config.json`: complete run configuration and total environment-step count
- `final_model.pt`: final Q-network state dictionary

The experiment groups are `learning_rate`, `exploration`, `replay_buffer`, and
`batch_size`. Run `python run_task2.py` to reproduce missing experiments; runs
with an existing `config.json` are treated as complete and skipped.
