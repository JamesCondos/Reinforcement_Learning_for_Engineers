import gymnasium as gym
from q_learning.agent import QLAgent
import argparse
import os
import random
import numpy as np
import torch
from utils.util import collect_trajectories

from utils.logger import Logger
import time


def train(args):
    log_dir = "logs/"
    log_dir = os.path.join(log_dir, args.env_name)
    if args.off_policy:
        log_name = "QLearning"
    elif args.look_ahead_step > 0:
        log_name = f"SARSA_step{args.look_ahead_step}"
    elif args.sarsa_lambda > 0:
        log_name = f"SARSA_lambda{args.sarsa_lambda:g}"
    else:
        log_name = "SARSA"

    if args.exp_name:
        log_name = log_name + "_" + args.exp_name
    log_name = log_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(log_dir, log_name)
    if not (os.path.exists(log_dir)):
        os.makedirs(log_dir)
    logger = Logger(log_dir=log_dir)

    render_mode = None    
    if args.render:
        render_mode="human"
    
    env = gym.make(args.env_name,render_mode=render_mode)
    env.action_space.seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete, "Q Learning only supports discrete environment"


    agent = QLAgent(observation_dim=env.observation_space.n,
                    action_dim=env.action_space.n, 
                    alpha=args.learning_rate,
                    gamma=args.discount_factor,
                    off_policy=args.off_policy,
                    sarsa_lambda=args.sarsa_lambda,
                    look_ahead_step=args.look_ahead_step,
                    epsilon=args.epsilon,
                    epsilon_decay=args.epsilon_decay,
                    min_epsilon=args.epsilon_min,
                    init_qtable_value=args.init_qtable_value,
                    )
    
    total_envsteps = 0
    update_info = None
    returns = []

    for episode in range(args.episodes):

        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        
        episode_reward = 0
        cliff_falls = 0
        step = 0
        done = False

        action = agent.sample_action(state)

        while not done:

            step_result = env.step(action)

            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            if args.max_episode_steps is not None and step + 1 >= args.max_episode_steps:
                done = True

            
            next_action = agent.sample_action(next_state)

            update_info = agent.update(
                state, action, reward, next_state, next_action, done
            )

            state = next_state
            action = next_action
            episode_reward += reward
            if reward == -100:
                cliff_falls += 1
            total_envsteps += 1
            step += 1
            
            if args.render:
                env.render()

            logger.log_scalar(reward, "Reward", total_envsteps)
            logger.log_scalar(update_info['td_error'], "TD Error", total_envsteps)

        
        returns.append(episode_reward)
        logger.log_scalar(episode_reward, "Episode Return", episode)
        logger.log_scalar(step, "Episode Length", episode)
        logger.log_scalar(cliff_falls, "Cliff Falls", episode)
        logger.log_scalar(agent.epsilon, "Epsilon-Greedy Value", episode)

        if (episode+1) % 50 == 0:  # print training info
            print(f"Episode {episode+1} | Worst episode: {min(returns)} | Best episode: {max(returns)}")
            returns = []

        if args.render:
            time.sleep(0.2)
    
    if args.log_video:
        env = gym.make(args.env_name,render_mode="rgb_array")
        logging_batch = 1
        final_batch = collect_trajectories(env, agent, logging_batch, render=True)
        video = torch.tensor(final_batch['images']).permute(0,1,4,2,3)  # [N,T,H,W,C] → [N,T,C,H,W]
        print("logging video")
        logger.log_video(video, "Final Video", step=1)

    if args.save_qtable:
        q_table = np.stack(
            [agent.q_table[state] for state in range(env.observation_space.n)]
        )
        np.savez_compressed(
            os.path.join(log_dir, "q_table.npz"),
            q_table=q_table,
            seed=args.seed,
            env_name=args.env_name,
        )

    logger.flush()
    logger.close()
    
    env.close()
    
    if args.final_render:
        new_env = gym.make(args.env_name, render_mode="human")
        for _ in range(5):
            state, _ = new_env.reset()
            done = False
            while not done:
                new_env.render()
                action = agent.sample_action(state, greedy=True)
                next_state, reward, terminated, truncated, info = new_env.step(action)
                state = next_state
                done = terminated or truncated
        new_env.close()

if __name__ == '__main__':
    # Set env
    parser = argparse.ArgumentParser(description="Tabular Q-learning and SARSA")
    parser.add_argument("--env_name", type=str, required=True, help="gym environment")
    parser.add_argument("--episodes", type=int, default=1000, help="training episodes")
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="optional safety cap on the number of steps in one episode",
    )
    parser.add_argument("--render", action="store_true", help="whether to render")
    parser.add_argument("--final_render", action="store_true", help="whether to see final results")
    parser.add_argument("--log_video", action="store_true")
    parser.add_argument("--off_policy", action="store_true", help="use Q-learning instead of SARSA")
    parser.add_argument("--look_ahead_step", type=int, default=0, help="n-step SARSA")
    parser.add_argument("--sarsa_lambda", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--discount_factor", type=float, default=0.99)

    parser.add_argument("--epsilon", type=float, default=0.5, help="beginning epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.0, help="minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.99, help="epsilon decay weight")
    parser.add_argument("--init_qtable_value", type=float, default=0.0)
    parser.add_argument("--save_model_step", type=int, default=None)
    parser.add_argument("--save_qtable", action="store_true", help="save the final Q-table")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--exp_name", type=str, default=None, help="add extra info (parameters) to log")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    args.device = device

    train(args)
