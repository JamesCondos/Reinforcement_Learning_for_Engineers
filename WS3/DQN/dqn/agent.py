import torch
import torch.nn as nn
import torch.optim as optim

import random
from collections import deque
import numpy as np

from dqn.network import QNet


class DQNAgent:
    def __init__(self, env, device='cpu',
                 hidden_size=64, layer_num=3, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.0,
                 epsilon_decay=0.995, memory_size=10000, batch_size=64,
                 lr=1e-3, target_update_freq=10):
        self.env = env
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(int(memory_size), device=device)
        
        self.total_update_steps = 0

        self.target_update_freq = target_update_freq

        self.q_net = QNet(self.state_dim, self.act_dim, hidden_size, layer_num).to(device)
        self.target_net = QNet(self.state_dim, self.act_dim, hidden_size, layer_num).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.loss_function = nn.MSELoss()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def sample_action(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)        
        
        ### (TODO) get the action from the critic using an epsilon-greedy strategy
        action = ...
        
        return action
    
    def update(self, batch):
        
        states, actions, rewards, next_states, dones = batch
        actions = actions.long().unsqueeze(1).to(self.device)  # [batch_size, 1]

        ### (TODO) Compute target values (r + gamma*Q(s',a';w-))
        ### 1. Compute Q(s', : ;w-) for all next actions from the target network
        ### 2. Select the best action a' and get Q(s',a';w-)
        ### 3. Compute target values  
        ### Note that, if the step is the termination (done), there is no next state, i.e. Q(s',a';w-) = 0
        with torch.no_grad():
            next_qa_values = ...
            
            next_action = ...

            next_q_values = ...

            target_values = ...


        ### (TODO) Compute Q(s,a;w_i)
        ### 1. Compute Q(s, : ;w_i) for all actions from the Q network
        ### 2. Gather the value for action taken
        qa_values = ...
        q_values = ...

        
        ### (TODO) Compute loss, with the given MSE self.loss_function
        loss = ...

        ### Update the critic network wrt computed loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.total_update_steps += 1

        ### (TODO) Update the target network every self.target_update_freq with self.update_target_net()
        ...

        return {        
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "gradient": grad_norm.item(),
            "epsilon": self.epsilon,
            # "td_error": (q_values - target_values).detach().cpu().numpy(),
        }

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def epsilon_step(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class ReplayBuffer:
    def __init__(self, size=int(1e6), device='cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        
        # Stacking as a single numpy array before converting to tensor is much faster
        return (
            torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(a), dtype=torch.int64, device=self.device),
            torch.as_tensor(np.stack(r), dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(np.stack(s_), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(d), dtype=torch.float32, device=self.device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)