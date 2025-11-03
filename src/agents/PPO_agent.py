"""
PPO Agent - Proximal Policy Optimization pour le Job Shop

PPO (Proximal Policy Optimization) est un algorithme de Policy Gradient qui :
1. Apprend une politique (actor) pour choisir les actions
2. Apprend une fonction de valeur (critic) pour évaluer les états
3. Utilise un clipping pour stabiliser l'apprentissage
4. Est généralement plus stable et performant que DQN

Concepts clés :
- Actor : Réseau qui produit les probabilités d'actions
- Critic : Réseau qui estime la valeur d'un état
- Advantage : Mesure si une action est meilleure que la moyenne
- Clipping : Limite les changements de politique pour stabilité
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict
import pickle
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import * 

class RolloutBuffer:
    """
    Buffer pour stocker une trajectoire complète (rollout).
    
    Contrairement au replay buffer de DQN, PPO utilise des expériences
    récentes et les jette après apprentissage.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        """Ajoute une transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        """Retourne toutes les données."""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )
    
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


class SimplePPOAgent:
    """
    Agent PPO simplifié (sans PyTorch pour l'instant).
    
    Version pédagogique qui utilise des approximations simples.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialise l'agent PPO.
        
        Args:
            state_size: Dimension de l'espace d'état
            action_size: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount
            gae_lambda: Lambda pour GAE (Generalized Advantage Estimation)
            clip_epsilon: Paramètre de clipping pour PPO
            value_coef: Coefficient de la loss du critic
            entropy_coef: Coefficient du bonus d'entropie (exploration)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Buffer pour collecter les trajectoires
        self.buffer = RolloutBuffer()
        
        # Tables pour policy et value (version simplifiée)
        self.policy_table = {}  # État -> logits des actions
        self.value_table = {}   # État -> valeur de l'état
        
        # Statistiques
        self.training_history = {
            'episode_rewards': [],
            'episode_makespans': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def _state_to_key(self, state: np.ndarray) -> tuple:
        """Convertit un état en clé."""
        return tuple(np.round(state, 2))
    
    def _get_policy_logits(self, state: np.ndarray) -> np.ndarray:
        """
        Obtient les logits de la politique pour un état.
        
        Returns:
            Logits (scores bruts avant softmax)
        """
        key = self._state_to_key(state)
        if key not in self.policy_table:
            # Initialiser avec des valeurs aléatoires
            self.policy_table[key] = np.random.randn(self.action_size) * 0.1
        return self.policy_table[key]
    
    def _get_action_probs(self, state: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """
        Calcule les probabilités d'actions (softmax sur actions légales).
        
        Args:
            state: État actuel
            legal_actions: Actions légales
            
        Returns:
            Probabilités pour chaque action (0 pour illégales)
        """
        logits = self._get_policy_logits(state)
        
        # Masquer les actions illégales
        masked_logits = np.full(self.action_size, -np.inf)
        masked_logits[legal_actions] = logits[legal_actions]
        
        # Softmax
        exp_logits = np.exp(masked_logits - np.max(masked_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def _get_value(self, state: np.ndarray) -> float:
        """Obtient la valeur estimée d'un état."""
        key = self._state_to_key(state)
        if key not in self.value_table:
            self.value_table[key] = 0.0
        return self.value_table[key]
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> Tuple[int, float, float]:
        """
        Sélectionne une action selon la politique actuelle.
        
        Args:
            state: État actuel
            legal_actions: Actions légales
            
        Returns:
            (action, log_prob, value)
        """
        probs = self._get_action_probs(state, legal_actions)
        
        # Échantillonner une action selon les probabilités
        action = np.random.choice(self.action_size, p=probs)
        
        # Log-probabilité de l'action choisie
        log_prob = np.log(probs[action] + 1e-10)
        
        # Valeur de l'état
        value = self._get_value(state)
        
        return action, log_prob, value
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                    dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les advantages avec GAE (Generalized Advantage Estimation).
        
        GAE permet un meilleur compromis biais-variance dans l'estimation
        de l'advantage.
        
        Args:
            rewards: Récompenses
            values: Valeurs des états
            dones: Indicateurs de fin d'épisode
            
        Returns:
            (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        # Calcul inverse (du dernier step au premier)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
            # Return = advantage + value
            returns[t] = advantages[t] + values[t]
        
        # Normaliser les advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def learn(self, epochs: int = 4):
        """
        Met à jour la politique avec les données du buffer.
        
        Args:
            epochs: Nombre de passages sur les données
        """
        if len(self.buffer) == 0:
            return
        
        # Récupérer les données
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # Calculer advantages et returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Entraînement sur plusieurs epochs
        for epoch in range(epochs):
            
            # Pour chaque transition
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                old_log_prob = old_log_probs[i]
                advantage = advantages[i]
                return_val = returns[i]
                
                # ===== UPDATE ACTOR (POLICY) =====
                
                # Nouvelle log-prob avec politique actuelle
                key = self._state_to_key(state)
                logits = self.policy_table.get(key, np.random.randn(self.action_size) * 0.1)
                
                # Softmax pour avoir les probs
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                new_log_prob = np.log(probs[action] + 1e-10)
                
                # Ratio de probabilités
                ratio = np.exp(new_log_prob - old_log_prob)
                
                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss = -min(surr1, surr2)
                
                # Gradient (approximation simple)
                grad = np.zeros_like(logits)
                grad[action] = -advantage * (1 if surr1 < surr2 else 0)
                
                # Update
                self.policy_table[key] = logits - self.learning_rate * grad
                
                # ===== UPDATE CRITIC (VALUE) =====
                
                current_value = self.value_table.get(key, 0.0)
                value_loss = (return_val - current_value) ** 2
                
                # Gradient descent sur value
                value_grad = -2 * (return_val - current_value)
                self.value_table[key] = current_value - self.learning_rate * self.value_coef * value_grad
        
        # Vider le buffer après apprentissage
        self.buffer.clear()
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'policy_table': self.policy_table,
                'value_table': self.value_table,
                'training_history': self.training_history
            }, f)
        saved(f"Modèle PPO sauvegardé", filepath)
    
    def load(self, filepath: str):
        """Charge un modèle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.policy_table = data['policy_table']
            self.value_table = data['value_table']
            self.training_history = data['training_history']

        print(f"Modèle PPO chargé : {filepath}")

# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    
    section("DÉMONSTRATION - Agent PPO")
    
    # Créer un agent
    agent = SimplePPOAgent(state_size=5, action_size=3)
    
    passed(f"Agent PPO créé:")
    print(f"   - Taille état: {agent.state_size}")
    print(f"   - Nombre d'actions: {agent.action_size}")
    print(f"   - Learning rate: {agent.learning_rate}")
    print(f"   - Gamma: {agent.gamma}")
    print(f"   - Clip epsilon: {agent.clip_epsilon}")
    
    # Test de sélection d'action
    print("\nTest de sélection d'action:")
    test_state = np.array([0.5, 0.3, 0.8, 0.1, 0.6])
    legal_actions = [0, 1, 2]
    
    for i in range(5):
        action, log_prob, value = agent.select_action(test_state, legal_actions)
        print(f"   Essai {i+1}: Action {action}, log_prob={log_prob:.3f}, value={value:.3f}")
    
    passed("Agent PPO fonctionnel !")