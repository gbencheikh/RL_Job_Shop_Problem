"""
DQN Agent - Deep Q-Network pour le Job Shop

DQN (Deep Q-Network) est un algorithme de Deep Reinforcement Learning qui :
1. Utilise un réseau de neurones pour approximer la fonction Q(state, action)
2. Apprend à partir d'expériences passées (replay buffer)
3. Utilise un réseau cible pour stabiliser l'apprentissage

Concepts clés :
- Q-value : Valeur estimée d'une action dans un état donné
- Replay Buffer : Mémoire des expériences passées
- Target Network : Copie du réseau principal pour stabilité
- Epsilon-greedy : Exploration vs exploitation
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple
import pickle
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import * 

class ReplayBuffer:
    """
    Mémoire pour stocker les expériences (transitions).
    
    Une transition = (état, action, récompense, nouvel_état, terminé)
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Nombre maximum de transitions à stocker
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Échantillonne un batch aléatoire.
        
        Args:
            batch_size: Nombre de transitions à échantillonner
            
        Returns:
            Liste de transitions
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class SimpleDQNAgent:
    """
    Agent DQN simplifié (sans PyTorch pour l'instant).
    
    Utilise un réseau de neurones simple avec numpy.
    C'est une version pédagogique pour comprendre les concepts.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialise l'agent DQN.
        
        Args:
            state_size: Dimension de l'espace d'état
            action_size: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount (importance du futur)
            epsilon: Probabilité d'exploration initiale
            epsilon_decay: Décroissance d'epsilon
            epsilon_min: Epsilon minimum
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Q-table simplifié (pour version sans deep learning)
        # Dans une vraie version, ce serait un réseau de neurones
        self.q_table = {}
        
        # Statistiques
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'epsilon_history': []
        }
    
    def _state_to_key(self, state: np.ndarray) -> tuple:
        """Convertit un état en clé pour le Q-table."""
        return tuple(np.round(state, 2))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Obtient les Q-values pour toutes les actions dans un état.
        
        Args:
            state: État actuel
            
        Returns:
            Array des Q-values pour chaque action
        """
        key = self._state_to_key(state)
        if key not in self.q_table:
            # Initialiser avec des valeurs aléatoires
            self.q_table[key] = np.zeros(self.action_size)
        return self.q_table[key]
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """
        Sélectionne une action avec epsilon-greedy.
        
        Args:
            state: État actuel
            legal_actions: Liste des actions légales
            
        Returns:
            Action choisie
        """
        # Exploration : action aléatoire
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation : meilleure action selon Q-values
        q_values = self.get_q_values(state)
        
        # Ne considérer que les actions légales
        legal_q_values = [(q_values[a], a) for a in legal_actions]
        return max(legal_q_values)[1]
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire."""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self, batch_size: int = 32):
        """
        Apprend à partir d'un batch d'expériences.
        
        Args:
            batch_size: Taille du batch
        """
        if len(self.memory) < batch_size:
            return
        
        # Échantillonner un batch
        batch = self.memory.sample(batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Q-value cible
            if done:
                target = reward
            else:
                # Q-learning : target = reward + gamma * max(Q(next_state))
                next_q_values = self.get_q_values(next_state)
                target = reward + self.gamma * np.max(next_q_values)
            
            # Mise à jour Q-table
            key = self._state_to_key(state)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.action_size)
            
            # Update : Q(s,a) = Q(s,a) + lr * (target - Q(s,a))
            current_q = self.q_table[key][action]
            self.q_table[key][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Réduit epsilon (moins d'exploration au fil du temps)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'training_history': self.training_history
            }, f)

        saved("Modèle sauvegardé", filepath)
    
    def load(self, filepath: str):
        """Charge un modèle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.training_history = data['training_history']
        
        passed(f"Modèle chargé : {filepath}")

# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    section("DÉMONSTRATION - Agent DQN Simple")
    
    # Créer un agent
    agent = SimpleDQNAgent(state_size=5, action_size=3)
    
    passed(f"Agent créé:")
    print(f"   - Taille état: {agent.state_size}")
    print(f"   - Nombre d'actions: {agent.action_size}")
    print(f"   - Epsilon initial: {agent.epsilon}")
    print(f"   - Taille du buffer: {len(agent.memory)}")
    
    # Test de sélection d'action
    print("\nTest de sélection d'action:")
    test_state = np.array([0.5, 0.3, 0.8, 0.1, 0.6])
    legal_actions = [0, 1, 2]
    
    for i in range(5):
        action = agent.select_action(test_state, legal_actions)
        print(f"   Essai {i+1}: Action {action}")
    
    passed("Agent DQN fonctionnel !")