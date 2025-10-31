# src/agents/PPO_agent.py
"""
Squelette PPO (actor-critic) pour espaces discrets.
Ceci est une base: l'algorithme complet (clipping, multiple epochs, advantages) est partiellement implémenté
pour te donner un point de départ.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(128,128)):
        super().__init__()
        last = obs_dim
        actor_layers = []
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(last, h))
            actor_layers.append(nn.Tanh())
            last = h
        actor_layers.append(nn.Linear(last, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # critic
        last = obs_dim
        critic_layers = []
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(last, h))
            critic_layers.append(nn.Tanh())
            last = h
        critic_layers.append(nn.Linear(last, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr=3e-4, device="cpu"):
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # buffer placeholders
        self.buffer = {"states": [], "actions": [], "rewards": [], "dones": [], "logps": [], "values": []}
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.batch_size = 64

    def act(self, state: np.ndarray):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(state_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        logp = dist.log_prob(torch.tensor(action, device=self.device))
        return action, logp.item(), value.item()

    def store(self, state, action, reward, done, logp, value):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["logps"].append(logp)
        self.buffer["values"].append(value)

    def update(self):
        # Implémentation simplifiée: calcule avantages, met à jour le réseau en une passe.
        # Pour de la performance réelle, vectoriser, normaliser, multiple epochs, minibatches.
        states = torch.tensor(np.array(self.buffer["states"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        old_logps = torch.tensor(self.buffer["logps"], dtype=torch.float32, device=self.device).unsqueeze(1)
        returns = []
        G = 0
        for r, d in zip(reversed(self.buffer["rewards"]), reversed(self.buffer["dones"])):
            if d:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
        # normalize advantages
        values = torch.tensor(self.buffer["values"], dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, _ = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logps = dist.log_prob(actions.squeeze()).unsqueeze(1)
        ratio = torch.exp(logps - old_logps)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # critic loss
        _, values_pred = self.model(states)
        critic_loss = nn.functional.mse_loss(values_pred, returns)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clear buffer
        for k in self.buffer:
            self.buffer[k] = []
