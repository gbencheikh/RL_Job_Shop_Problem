"""
Deep DQN Agent - Deep Q-Network avec PyTorch

Ce module implémente un vrai agent DQN avec réseaux de neurones.

Architecture :
    État → [Linear(128)] → [ReLU] → [Linear(128)] → [ReLU] → [Linear(actions)] → Q-values

Techniques utilisées :
- Experience Replay : Mémoire pour décorréler les expériences
- Target Network : Copie du réseau pour stabiliser l'apprentissage
- Epsilon-Greedy : Balance exploration vs exploitation
- Gradient Clipping : Évite les explosions de gradients
- Double DQN : Réduit le biais de surestimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
import copy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import *

# ============================================
# RÉSEAU DE NEURONES Q-NETWORK
# ============================================

class QNetwork(nn.Module):
    """
    Réseau de neurones pour approximer la fonction Q(s,a).
    
    Architecture simple mais efficace :
    - Couche d'entrée : state_size
    - 2 couches cachées : 128 neurones chacune
    - Couche de sortie : action_size (Q-value pour chaque action)
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128]):
        """
        Initialise le réseau.
        
        Args:
            state_size: Dimension de l'espace d'état
            action_size: Nombre d'actions possibles
            hidden_sizes: Tailles des couches cachées
        """
        super(QNetwork, self).__init__()
        
        # Couche d'entrée
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        
        # Couches cachées
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        # Couche de sortie
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        
        # Initialisation des poids (Xavier/Glorot)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass : calcule les Q-values pour un état.
        
        Args:
            state: Tensor de forme (batch_size, state_size)
            
        Returns:
            Q-values de forme (batch_size, action_size)
        """
        # Couche 1 avec activation ReLU
        x = F.relu(self.fc1(state))
        
        # Couche 2 avec activation ReLU
        x = F.relu(self.fc2(x))
        
        # Couche de sortie (pas d'activation)
        q_values = self.fc3(x)
        
        return q_values


# ============================================
# DUELING DQN (Architecture Avancée)
# ============================================

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN : Sépare la valeur de l'état et l'avantage des actions.
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    
    Où :
    - V(s) : Valeur de l'état (indépendante de l'action)
    - A(s,a) : Avantage d'une action par rapport aux autres
    
    Amélioration : Apprend mieux la valeur des états
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DuelingQNetwork, self).__init__()
        
        # Feature extraction (commun)
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass avec architecture Dueling."""
        features = self.feature(state)
        
        # Calculer V(s) et A(s,a)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combiner : Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


# ============================================
# REPLAY BUFFER
# ============================================

class ReplayBuffer:
    """
    Mémoire d'expériences pour Experience Replay.
    
    Stocke les transitions (s, a, r, s', done) et permet
    d'échantillonner des batches aléatoires.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Capacité maximale du buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Échantillonne un batch aléatoire.
        
        Returns:
            Tuple de tensors (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================
# AGENT DEEP DQN
# ============================================

class DeepDQNAgent:
    """
    Agent DQN complet avec PyTorch.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_dueling: bool = False,
        use_double_dqn: bool = True
    ):
        """
        Initialise l'agent Deep DQN.
        
        Args:
            state_size: Dimension de l'espace d'état
            action_size: Nombre d'actions
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Probabilité d'exploration initiale
            epsilon_decay: Décroissance d'epsilon
            epsilon_min: Epsilon minimum
            buffer_capacity: Capacité du replay buffer
            batch_size: Taille des batches
            target_update_freq: Fréquence de mise à jour du target network
            use_dueling: Utiliser Dueling DQN
            use_double_dqn: Utiliser Double DQN
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Device (GPU si disponible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        notify(f"Device utilisé: {self.device}")
        
        # Créer les réseaux
        if use_dueling:
            self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
            self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        else:
            self.policy_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net = QNetwork(state_size, action_size).to(self.device)
        
        # Copier les poids vers le target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Mode évaluation (pas de dropout, etc.)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Compteurs
        self.steps = 0
        self.episodes = 0
        
        # Statistiques
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'losses': [],
            'epsilon_history': []
        }
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """
        Sélectionne une action avec epsilon-greedy.
        
        Args:
            state: État actuel
            legal_actions: Actions légales
            
        Returns:
            Action choisie
        """
        # Exploration : action aléatoire
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation : meilleure action selon le réseau
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Masquer les actions illégales
            masked_q = np.full(self.action_size, -np.inf)
            masked_q[legal_actions] = q_values[legal_actions]
            
            return int(np.argmax(masked_q))
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire."""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """
        Effectue une itération d'apprentissage.
        
        Returns:
            Loss si apprentissage effectué, None sinon
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir en tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ===== CALCUL DE LA LOSS =====
        
        # Q-values actuelles pour les actions prises
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Q-values cibles
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN : sélection avec policy_net, évaluation avec target_net
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # DQN standard
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss (Mean Squared Error)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # ===== BACKPROPAGATION =====
        
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()             # Calculer gradients
        
        # Gradient clipping (évite l'explosion des gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()       # Mettre à jour les poids
        
        # ===== MISE À JOUR DU TARGET NETWORK =====
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Réduit epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'training_history': self.training_history
        }, filepath)
        saved("Modèle Deep DQN sauvegardé",filepath)
    
    def load(self, filepath: str):
        """Charge un modèle."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.training_history = checkpoint['training_history']
        passed(f"Modèle Deep DQN chargé : {filepath}")


# ============================================
# TESTS
# ============================================


if __name__ == "__main__":
    section("DÉMONSTRATION - Deep DQN Agent")
    
    # Test du réseau
    print("Test du réseau de neurones:")
    network = QNetwork(state_size=5, action_size=3)
    print(network)
    print()
    
    # Test forward pass
    test_state = torch.randn(1, 5)  # Batch de 1 état
    q_values = network(test_state)
    print(f"État test: {test_state.numpy()}")
    print(f"Q-values: {q_values.detach().numpy()}")
    print()
    
    # Créer un agent
    agent = DeepDQNAgent(state_size=5, action_size=3)
    passed("Agent Deep DQN créé:")
    print(f"   - Device: {agent.device}")
    print(f"   - Paramètres réseau: {sum(p.numel() for p in agent.policy_net.parameters())}")
    print(f"   - Batch size: {agent.batch_size}")
    
    passed("Deep DQN fonctionnel !")