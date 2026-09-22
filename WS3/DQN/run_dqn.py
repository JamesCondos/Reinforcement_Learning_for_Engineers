import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import time

from dqn.agent import DQNAgent
from utils.util import collect_trajectories
from utils.logger import Logger

   

def train(args):
    log_dir = "logs/"
    log_dir = os.path.join(log_dir, args.env_name)
    mode = "_standard" if not args.exp_name else ""
    log_name = "DQN" 
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
    eval_env = gym.make(args.env_name)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete, "DQN only supports discrete action space"

    agent = DQNAgent(env, 
                     batch_size=args.batch_size,
                     device=args.device,
                     hidden_size=args.hidden_size, layer_num=args.layer_num,
                     gamma=args.discount_factor, 
                     epsilon=args.epsilon,
                     epsilon_min=args.epsilon_min,
                     epsilon_decay=args.epsilon_decay,
                     lr = args.learning_rate,
                     memory_size=args.memory_size,
                     target_update_freq=args.target_update_freq
                     )
    
    total_envsteps = 0
    returns = []
    update_info = None

    for episode in range(args.episodes):
        
        state, _ = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False

        while not done:

            ### (TODO) Get the action from the agent with agent.sample_action(...)
            action = agent.sample_action(state)

            ### (TODO) Implement the action
            step_result = env.step(action)

            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated

            ### Add the tuple to the replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, float(terminated))

                        
            state = next_state
            episode_reward += reward
            episode_step += 1
            total_envsteps += 1
            

            if len(agent.replay_buffer) < max(args.learning_start, args.batch_size):
                continue

            ### (TODO) Sample args.batch_size from the agent's replay buffer
            batch = agent.replay_buffer.sample(args.batch_size)
            
            ### (TODO) Update the agent with the batch
            update_info = agent.update(batch)

            ### Log data
            logger.log_scalar(update_info['critic_loss'], "Network Loss", total_envsteps)
            logger.log_scalar(update_info['q_values'], "Q Values", total_envsteps)
            logger.log_scalar(update_info['target_values'], "Target Q Values", total_envsteps)

        
        
        agent.epsilon_step()
        
        returns.append(episode_reward)
        logger.log_scalar(agent.epsilon, "_Epsilon", episode)

        if (episode+1) % 10 == 0: 
                        
            eval_batch = collect_trajectories(eval_env, agent, 1, render=False)
            eval_return = np.sum(eval_batch['rewards'])
            logger.log_scalar(eval_return, "Eval Returns", total_envsteps)

            if update_info:
                print(f"Episode {episode+1}/{args.episodes}: total steps {total_envsteps},  eval return {eval_return:.2f}, "
                    f"loss {update_info['critic_loss']:.4f}, ",
                    f"epsilon {agent.epsilon:.3f}")
        
        if args.render:
            time.sleep(0.05)
            
    
    if args.log_video:
        env = gym.make(args.env_name,render_mode="rgb_array")
        logging_batch = 1
        final_batch = collect_trajectories(env, agent, logging_batch, render=True)
        video = torch.tensor(final_batch['images']).permute(0,1,4,2,3)  # [N,T,H,W,C] → [N,T,C,H,W]
        print("Logging video")
        logger.log_video(video, "Final Video", step=1)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Q Learning")
    parser.add_argument("--env_name", type=str, required=True, help="gym environment")
    parser.add_argument("--episodes", type=int, default=1000, help="training episodes")
    parser.add_argument("--memory_size", type=int, default=1e5, help="replay buffer size")
    parser.add_argument("--learning_start", type=int, default=1000, help="learning start")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--log_video", action="store_true")
    parser.add_argument("--target_update_freq", type=int, default=500, help="waiting steps to update the target network")

    parser.add_argument("--hidden_size", type=int, default=64, help="neural network hidden layer size")
    parser.add_argument("--layer_num", type=int, default=3, help="neural network layer number")
    parser.add_argument("--render", action="store_true", help="whether to render")
    parser.add_argument("--max_ep_len", type=int, help="max length for each episode")  
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0, help="beginning epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.0, help="minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="epsilon decay weight")
    parser.add_argument("--exp_name", default=None, help="add extra info to log")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    args.device = device

    train(args)
