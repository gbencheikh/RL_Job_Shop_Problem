"""
Job Shop Environment - Environnement Gymnasium pour le Job Shop Problem

Cet environnement permet à un agent RL d'apprendre à ordonnancer des opérations.

Concepts RL:
- État (State): Quelles opérations sont faites, machines disponibles, temps actuel
- Action: Choisir quelle opération ordonnancer parmi les opérations disponibles
- Récompense (Reward): Négative = makespan (on veut minimiser)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.environments.job_shop_instance import JobShopInstance
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import *

class JobShopEnv(gym.Env):
    """
    Environnement Gymnasium pour le Job Shop Scheduling Problem.
    
    L'agent doit choisir séquentiellement quelle opération ordonnancer.
    
    État (Observation):
        - Pour chaque job: progression (opérations complétées / total)
        - Pour chaque machine: temps de disponibilité
        - Temps actuel
        
    Action:
        - Entier entre 0 et num_jobs-1
        - Représente: "Ordonnancer la prochaine opération du job X"
        
    Récompense:
        - Récompense négative à chaque step = durée de l'opération
        - Objectif: Minimiser le total = minimiser le makespan
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, instance: JobShopInstance, render_mode: Optional[str] = None):
        """
        Initialise l'environnement.
        
        Args:
            instance: Instance de Job Shop à résoudre
            render_mode: Mode de rendu ('human' ou None)
        """
        super().__init__()
        
        self.instance = instance
        self.render_mode = render_mode
        self.operation_log = []

        # Espaces d'observation et d'action
        # Observation: [job_progress × num_jobs, machine_times × num_machines, current_time]
        obs_size = instance.num_jobs + instance.num_machines + 1
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action: choisir un job (0 à num_jobs-1)
        self.action_space = spaces.Discrete(instance.num_jobs)
        
        # État interne
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Réinitialise l'environnement.
        
        Returns:
            observation: État initial
            info: Informations additionnelles
        """
        super().reset(seed=seed)
        
        self.operation_log = []

        # Progression de chaque job (nombre d'opérations complétées)
        self.job_progress = np.zeros(self.instance.num_jobs, dtype=np.int32)
        
        # Temps où chaque machine sera disponible
        self.machine_available_time = np.zeros(self.instance.num_machines, dtype=np.float32)
        
        # Temps où chaque job sera disponible (dernière op terminée)
        self.job_available_time = np.zeros(self.instance.num_jobs, dtype=np.float32)
        
        # Temps actuel
        self.current_time = 0.0
        
        # Ordonnancement (pour le rendu)
        self.schedule = []
        
        # Nombre total d'opérations complétées
        self.operations_done = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Construit le vecteur d'observation.
        
        Returns:
            Vecteur numpy représentant l'état actuel
        """
        # Progression normalisée de chaque job (0 à 1)
        job_progress_normalized = np.array([
            self.job_progress[j] / self.instance.get_num_operations(j)
            for j in range(self.instance.num_jobs)
        ], dtype=np.float32)
        
        # Temps de disponibilité des machines (normalisé par current_time + 1)
        machine_times = self.machine_available_time / (self.current_time + 1)
        
        # Temps actuel (normalisé)
        current_time_normalized = np.array([self.current_time / 100.0], dtype=np.float32)
        
        # Concaténer tout
        obs = np.concatenate([
            job_progress_normalized,
            machine_times,
            current_time_normalized
        ])
        
        return obs
    
    def _get_legal_actions(self) -> List[int]:
        """
        Retourne les actions légales (jobs qui ont encore des opérations).
        
        Returns:
            Liste des IDs de jobs disponibles
        """
        legal_actions = []
        for job_id in range(self.instance.num_jobs):
            # Un job est disponible si il lui reste des opérations
            if self.job_progress[job_id] < self.instance.get_num_operations(job_id):
                legal_actions.append(job_id)
        return legal_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: ID du job dont on veut ordonnancer la prochaine opération
            
        Returns:
            observation: Nouvel état
            reward: Récompense (négative)
            terminated: Episode terminé?
            truncated: Episode tronqué? (toujours False ici)
            info: Informations additionnelles
        """
        job_id = action
        
        # Vérifier que l'action est légale
        legal_actions = self._get_legal_actions()
        
        if job_id not in legal_actions:
            # Action illégale = grosse pénalité
            return self._get_observation(), -1000.0, False, False, {
                'error': 'Action illégale',
                'legal_actions': legal_actions
            }
        
        # Récupérer l'opération à ordonnancer
        op_index = self.job_progress[job_id]
        machine_id, duration = self.instance.get_operation(job_id, op_index)
        
        # Calculer le temps de début de l'opération
        # = max(machine disponible, job disponible)
        start_time = max(
            self.machine_available_time[machine_id],
            self.job_available_time[job_id]
        )
        
        # Mettre à jour les temps
        end_time = start_time + duration
        self.machine_available_time[machine_id] = end_time
        self.job_available_time[job_id] = end_time
        self.current_time = max(self.current_time, end_time)
        
        # Enregistrer dans l'ordonnancement
        self.schedule.append({
            'job_id': job_id,
            'op_index': op_index,
            'machine_id': machine_id,
            'start_time': start_time,
            'duration': duration,
            'end_time': end_time
        })
        
        # Mettre à jour la progression
        self.job_progress[job_id] += 1
        self.operations_done += 1
        
        self.operation_log.append((job_id, op_index, machine_id, start_time, end_time))

        # Récompense = durée négative (on veut minimiser le temps)
        reward = -duration
        
        # Vérifier si l'épisode est terminé
        terminated = (self.operations_done == self.instance.total_operations)
        
        # Si terminé, récompense finale = -makespan
        if terminated:
            makespan = self.current_time
            reward = -makespan
        
        # Observer le nouvel état
        observation = self._get_observation()
        
        info = {
            'makespan': self.current_time if terminated else None,
            'legal_actions': self._get_legal_actions(),
            'operations_done': self.operations_done
        }
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """Affiche l'état actuel de l'environnement."""
        if self.render_mode == 'human':
            print(f"\n{'='*60}")
            print(f"Temps actuel: {self.current_time}")
            print(f"Opérations complétées: {self.operations_done}/{self.instance.total_operations}")
            print(f"\nProgression des jobs:")
            for j in range(self.instance.num_jobs):
                progress = self.job_progress[j]
                total = self.instance.get_num_operations(j)
                print(f"  Job {j}: {progress}/{total} opérations")
            print(f"\nDisponibilité des machines:")
            for m in range(self.instance.num_machines):
                print(f"  Machine {m}: disponible à t={self.machine_available_time[m]}")
            print(f"{'='*60}\n")
    
    def get_schedule(self) -> List[Dict]:
        """Retourne l'ordonnancement actuel."""
        return self.schedule


# ============================================
# TESTS ET DÉMONSTRATION
# ============================================

if __name__ == "__main__":
    section("DÉMONSTRATION - Job Shop Environment")
    
    # Créer une instance simple
    instance = JobShopInstance.create_simple_instance()
    print("Instance:")
    print(instance)
    
    # Créer l'environnement
    env = JobShopEnv(instance, render_mode='human')
    
    print("Test 1: Interaction manuelle avec l'environnement")
    print("-" * 60)
    
    # Reset
    obs, info = env.reset()
    print(f"Observation initiale: {obs}")
    print(f"Forme: {obs.shape}")
    env.render()
    
    # Politique simple: toujours choisir le premier job disponible
    print("\nExécution avec politique gloutonne:")
    print("-" * 60)
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        # Choisir le premier job légal
        legal_actions = env._get_legal_actions()
        action = legal_actions[0]
        
        print(f"\nStep {step_count}: Choisir Job {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        print(f"  Récompense: {reward}")
        print(f"  Actions légales: {info['legal_actions']}")
        
        if done:
            print(f"\n{'='*60}")
            passed(f"Episode terminé !")
            print(f"Makespan final: {info['makespan']}")
            print(f"Récompense totale: {total_reward}")
            print(f"Nombre de steps: {step_count}")
            print(f"{'='*60}")
    
    # Afficher l'ordonnancement final
    print("\nOrdonnancement final:")
    print("-" * 60)
    schedule = env.get_schedule()
    for op in schedule:
        print(f"Job {op['job_id']}, Op {op['op_index']}: "
              f"Machine {op['machine_id']} [{op['start_time']:.1f} - {op['end_time']:.1f}]")