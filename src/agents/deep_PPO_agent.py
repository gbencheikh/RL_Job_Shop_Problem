"""
Deep PPO Agent - Proximal Policy Optimization avec PyTorch

PPO est souvent MEILLEUR que DQN pour plusieurs raisons :
- Apprend une politique directement (plus naturel)
- Plus stable (clipping objective)
- Meilleure exploration (stochastique par nature)
- Fonctionne bien sur problèmes complexes

Architecture Actor-Critic :
- Actor : État → Probabilités d'actions (politique)
- Critic : État → Valeur V(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple
from collections import deque

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import *
# ============================================
# RÉSEAUX ACTOR-CRITIC
# ============================================

class ActorNetwork(nn.Module):
    """
    Réseau Actor : Produit la politique π(a|s).
    
    Sortie : Probabilités sur les actions (via softmax)
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Tensor (batch_size, state_size)
            
        Returns:
            logits: Scores bruts pour chaque action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def get_action_probs(self, state: torch.Tensor, legal_actions: List[int]) -> torch.Tensor:
        """
        Obtient les probabilités d'actions (avec masquage des illégales).
        """
        logits = self.forward(state)
        
        # Masquer actions illégales
        mask = torch.full_like(logits, float('-inf'))
        mask[:, legal_actions] = 0
        masked_logits = logits + mask
        
        # Softmax pour obtenir probabilités
        probs = F.softmax(masked_logits, dim=-1)
        return probs

class CriticNetwork(nn.Module):
    """
    Réseau Critic : Estime la valeur V(s) d'un état.
    
    Sortie : Scalaire (valeur de l'état)
    """
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Tensor (batch_size, state_size)
            
        Returns:
            value: Valeur de l'état (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class ActorCriticNetwork(nn.Module):
    """
    Réseau Actor-Critic combiné (partage les couches cachées).
    
    Plus efficace car partage les features entre actor et critic.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorCriticNetwork, self).__init__()
        
        # Couches partagées
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Tête Actor
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Tête Critic
        self.critic_head = nn.Linear(hidden_size, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combiné.
        
        Returns:
            (logits, value)
        """
        features = self.shared(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


# ============================================
# ROLLOUT BUFFER
# ============================================

class PPOBuffer:
    """
    Buffer pour stocker une trajectoire complète pour PPO.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def store(self, state, action, reward, value, log_prob, done):
        """Ajoute une transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        """Retourne toutes les données sous forme de tensors."""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }
    
    def clear(self):
        """Vide le buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


# ============================================
# AGENT DEEP PPO
# ============================================

class DeepPPOAgent:
    """
    Agent PPO complet avec PyTorch.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        use_shared_network: bool = True
    ):
        """
        Initialise l'agent Deep PPO.
        
        Args:
            state_size: Dimension état
            action_size: Nombre d'actions
            learning_rate: Taux d'apprentissage
            gamma: Facteur discount
            gae_lambda: Lambda pour GAE
            clip_epsilon: Epsilon pour clipping PPO
            value_coef: Coefficient loss value
            entropy_coef: Coefficient bonus entropie
            max_grad_norm: Norme max gradient
            update_epochs: Epochs par update
            batch_size: Taille batch
            use_shared_network: Utiliser réseau partagé Actor-Critic
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        notify(f"Device utilisé: {self.device}")
        
        # Créer les réseaux
        if use_shared_network:
            self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            self.use_shared = True
        else:
            self.actor = ActorNetwork(state_size, action_size).to(self.device)
            self.critic = CriticNetwork(state_size).to(self.device)
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=learning_rate
            )
            self.use_shared = False
        
        # Buffer
        self.buffer = PPOBuffer()
        
        # Statistiques
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        self.episodes = 0
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> Tuple[int, float, float]:
        """
        Sélectionne une action selon la politique actuelle.
        
        Returns:
            (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_shared:
                logits, value = self.network(state_tensor)
            else:
                logits = self.actor(state_tensor)
                value = self.critic(state_tensor)
            
            # Masquer actions illégales
            mask = torch.full_like(logits, float('-inf'))
            mask[:, legal_actions] = 0
            masked_logits = logits + mask
            
            # Distribution catégorielle
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """
        Calcule GAE (Generalized Advantage Estimation).
        
        Returns:
            (advantages, returns)
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        # Normaliser advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """
        Met à jour la politique avec PPO.
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        # Récupérer données
        data = self.buffer.get()
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']

        # Calculer advantages et returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convertir en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Statistiques
        policy_losses = []
        value_losses = []
        entropies = []
        
        # Multiple epochs sur les mêmes données
        dataset_size = len(states)
        
        for epoch in range(self.update_epochs):
            # Shuffle des indices
            indices = np.random.permutation(dataset_size)
            
            # Mini-batches
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # ===== FORWARD PASS =====
                
                if self.use_shared:
                    logits, values = self.network(batch_states)
                else:
                    logits = self.actor(batch_states)
                    values = self.critic(batch_states).squeeze()
                
                # Distribution pour actions
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # ===== CALCUL DES LOSSES =====
                
                # 1. Policy Loss (PPO Clipped Objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Value Loss (MSE)
                value_loss = F.mse_loss(values, batch_returns)
                
                # 3. Total Loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # ===== BACKPROPAGATION =====
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.use_shared:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Statistiques
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        # Vider le buffer
        self.buffer.clear()
        
        # Moyennes
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        if self.use_shared:
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'episodes': self.episodes
            }, filepath)
        else:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'episodes': self.episodes
            }, filepath)
        saved(f"Modèle Deep PPO sauvegardé",filepath)
    
    def load(self, filepath: str):
        """Charge un modèle."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        if self.use_shared:
            self.network.load_state_dict(checkpoint['network_state_dict'])
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.episodes = checkpoint['episodes']
        
        passed(f"Modèle Deep PPO chargé : {filepath}")




# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    section("DÉMONSTRATION - Deep PPO Agent")
    
    # Test du réseau Actor-Critic
    print("Test du réseau Actor-Critic:")
    network = ActorCriticNetwork(state_size=7, action_size=6)
    print(network)
    print()
    
    # Test forward pass
    test_state = torch.randn(1, 7)
    logits, value = network(test_state)
    print(f"État test: {test_state.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Value: {value.shape}")
    print()
    
    # Créer un agent
    agent = DeepPPOAgent(state_size=7, action_size=6)
    passed(f"Agent Deep PPO créé.")
    notify(f"   - Device: {agent.device}")
    if agent.use_shared:
        print(f"   - Paramètres réseau: {sum(p.numel() for p in agent.network.parameters())}")
    else:
        actor_params = sum(p.numel() for p in agent.actor.parameters())
        critic_params = sum(p.numel() for p in agent.critic.parameters())
        print(f"   - Paramètres actor: {actor_params}")
        print(f"   - Paramètres critic: {critic_params}")
    print(f"   - Update epochs: {agent.update_epochs}")
    print(f"   - Clip epsilon: {agent.clip_epsilon}")
    
    success("Deep PPO fonctionnel !")