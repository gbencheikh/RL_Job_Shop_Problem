# src/agents/DQN_agent.py
"""
DQN agent minimal pour environnements discrets (action_space.n = nombre de jobs).
Ce fichier implémente:
- un réseau fully-connected simple
- replay buffer
- méthode act(state), remember, train_step
"""

import random
from collections import deque, namedtuple
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.net = SimpleNet(obs_dim, action_dim).to(self.device)
        self.target_net = SimpleNet(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=50000)
        self.gamma = 0.99
        self.batch_size = 64
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.update_target_every = 1000
        self.step_count = 0

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.net(state_t)
        return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return
        self.step_count += 1
        batch = self.replay.sample(self.batch_size)
        state = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + (1 - done) * (self.gamma * q_next)

        loss = nn.functional.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        # target network update
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.net.state_dict())
