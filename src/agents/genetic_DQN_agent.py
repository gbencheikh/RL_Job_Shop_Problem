"""
Genetic DQN - Population d'agents DQN qui évoluent par algorithme génétique

Combine :
- Deep Q-Learning (apprentissage individuel)
- Algorithme Génétique (évolution de population)
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import *
from src.agents.deep_DQN_agent import DeepDQNAgent, QNetwork

class GeneticPopulation:
    """
    Gère une population d'agents DQN qui évoluent génétiquement.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        population_size: int = 20,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.02,
        crossover_rate: float = 0.7
    ):
        """
        Args:
            population_size: Nombre d'agents dans la population
            elite_ratio: Proportion des meilleurs à conserver (élitisme)
            mutation_rate: Probabilité de mutation des poids
            mutation_strength: Amplitude de la mutation
            crossover_rate: Probabilité de crossover
        """
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        
        # Créer la population initiale
        self.population = [
            DeepDQNAgent(state_size, action_size, epsilon=0.1)
            for _ in range(population_size)
        ]
        
        # Fitness de chaque agent
        self.fitness = np.zeros(population_size)
        
        self.generation = 0
    
    def evaluate_population(self, env, num_episodes: int = 5):
        """
        Évalue tous les agents de la population.
        
        Args:
            env: Environnement Job Shop
            num_episodes: Nombre d'épisodes par agent
            
        Returns:
            Liste des fitness (makespans moyens)
        """
        print(f"Génération {self.generation} - Évaluation de {self.population_size} agents...")
        
        for i, agent in enumerate(self.population):
            makespans = []
            
            for _ in range(num_episodes):
                state, _ = env.reset()
                done = False
                
                while not done:
                    legal_actions = env._get_legal_actions()
                    print("State size:", len(state))
                    action = agent.select_action(state, legal_actions)
                    state, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                
                makespans.append(info['makespan'])
            
            # Fitness = négatif du makespan moyen (on veut maximiser)
            self.fitness[i] = -np.mean(makespans)
            
            if (i + 1) % 5 == 0:
                print(f"  Agent {i+1}/{self.population_size}: Makespan moy = {-self.fitness[i]:.2f}")
        
        # Trier par fitness
        sorted_indices = np.argsort(self.fitness)[::-1]  # Décroissant
        best_makespan = -self.fitness[sorted_indices[0]]
        worst_makespan = -self.fitness[sorted_indices[-1]]
        avg_makespan = -np.mean(self.fitness)
        
        print(f"Statistiques Génération {self.generation}:")
        print(f"  Meilleur: {best_makespan:.2f}")
        print(f"  Moyen: {avg_makespan:.2f}")
        print(f"  Pire: {worst_makespan:.2f}")
        
        return self.fitness
    
    def selection(self) -> List[int]:
        """
        Sélectionne les parents pour la reproduction (tournament selection).
        
        Returns:
            Indices des agents sélectionnés
        """
        num_elite = int(self.population_size * self.elite_ratio)
        
        # Trier par fitness
        sorted_indices = np.argsort(self.fitness)[::-1]
        
        # Élitisme : garder les meilleurs
        selected = list(sorted_indices[:num_elite])
        
        # Sélection par tournoi pour le reste
        for _ in range(self.population_size - num_elite):
            # Choisir 3 agents au hasard et prendre le meilleur
            tournament = np.random.choice(self.population_size, 3, replace=False)
            winner = tournament[np.argmax(self.fitness[tournament])]
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1: DeepDQNAgent, parent2: DeepDQNAgent) -> DeepDQNAgent:
        """
        Crée un enfant par crossover de deux parents.
        
        Méthode : Uniform crossover au niveau des poids
        """
        child = DeepDQNAgent(self.state_size, self.action_size, epsilon=0.1)
        
        # Crossover des poids du policy network
        for child_param, parent1_param, parent2_param in zip(
            child.policy_net.parameters(),
            parent1.policy_net.parameters(),
            parent2.policy_net.parameters()
        ):
            # Masque aléatoire : 50% parent1, 50% parent2
            mask = torch.rand_like(child_param.data) > 0.5
            child_param.data = torch.where(mask, parent1_param.data, parent2_param.data)
        
        return child
    
    def mutate(self, agent: DeepDQNAgent):
        """
        Applique une mutation aux poids du réseau.
        
        Méthode : Gaussian noise
        """
        for param in agent.policy_net.parameters():
            if torch.rand(1).item() < self.mutation_rate:
                # Ajouter du bruit gaussien
                noise = torch.randn_like(param.data) * self.mutation_strength
                param.data += noise
    
    def evolve(self):
        """
        Crée la prochaine génération par sélection, crossover et mutation.
        """
        print(f"Évolution vers la génération {self.generation + 1}...")
        
        # Sélection
        selected_indices = self.selection()
        selected_agents = [self.population[i] for i in selected_indices]
        
        # Nouvelle population
        new_population = []
        
        # Élitisme : copier les meilleurs sans modification
        num_elite = int(self.population_size * self.elite_ratio)
        for i in range(num_elite):
            elite = copy.deepcopy(selected_agents[i])
            new_population.append(elite)
            print(f"  Élite {i+1}: Fitness = {self.fitness[selected_indices[i]]:.2f}")
        
        # Reproduction
        while len(new_population) < self.population_size:
            # Sélectionner deux parents
            parent1, parent2 = np.random.choice(selected_agents, 2, replace=False)
            
            # Crossover
            if np.random.rand() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def get_best_agent(self) -> DeepDQNAgent:
        """Retourne le meilleur agent de la population."""
        best_idx = np.argmax(self.fitness)
        return self.population[best_idx]
    
    def train_population(self, env, generations: int = 20, episodes_per_gen: int = 5):
        """
        Entraîne la population sur plusieurs générations.
        
        Args:
            env: Environnement
            generations: Nombre de générations
            episodes_per_gen: Épisodes d'évaluation par génération
        """
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': []
        }
        
        for gen in range(generations):
            # Évaluation
            self.evaluate_population(env, num_episodes=episodes_per_gen)
            
            # Statistiques
            best_fitness = np.max(self.fitness)
            avg_fitness = np.mean(self.fitness)
            worst_fitness = np.min(self.fitness)
            
            history['best_fitness'].append(-best_fitness)  # Makespan
            history['avg_fitness'].append(-avg_fitness)
            history['worst_fitness'].append(-worst_fitness)
            
            # Évolution (sauf dernière génération)
            if gen < generations - 1:
                self.evolve()
        
        return history


# ============================================
# FONCTION D'ENTRAÎNEMENT
# ============================================

if __name__ == "__main__":
    section("DÉMONSTRATION - Genetic DQN")
    
    # Test de création
    pop = GeneticPopulation(
        state_size=7,
        action_size=6,
        population_size=10
    )
    
    passed(f"Population créée:")
    print(f"   - Taille: {pop.population_size}")
    print(f"   - Élitisme: {pop.elite_ratio*100}%")
    print(f"   - Mutation rate: {pop.mutation_rate}")
    
    success("Genetic DQN fonctionnel !")
