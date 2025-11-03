"""
Heuristic Agents - Agents basés sur des heuristiques classiques

Ces agents n'apprennent pas, mais utilisent des règles prédéfinies.
Ils servent de baseline pour comparer les performances du RL.

Heuristiques implémentées:
- SPT (Shortest Processing Time): Choisir l'opération la plus courte
- LPT (Longest Processing Time): Choisir l'opération la plus longue
- Random: Choisir aléatoirement
- FIFO: Premier arrivé, premier servi
"""
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import * 
import random
from typing import List
from src.environments.job_shop_instance import JobShopInstance


class HeuristicAgent:
    """Classe de base pour les agents heuristiques."""
    
    def __init__(self, instance: JobShopInstance):
        self.instance = instance
        
    def select_action(self, legal_actions: List[int], job_progress: List[int]) -> int:
        """
        Sélectionne une action parmi les actions légales.
        
        Args:
            legal_actions: Liste des jobs disponibles
            job_progress: Progression de chaque job
            
        Returns:
            ID du job à ordonnancer
        """
        raise NotImplementedError


class RandomAgent(HeuristicAgent):
    """Agent qui choisit aléatoirement parmi les actions légales."""
    
    def select_action(self, legal_actions: List[int], job_progress: List[int]) -> int:
        return random.choice(legal_actions)


class SPTAgent(HeuristicAgent):
    """
    Shortest Processing Time.
    Choisit l'opération avec la plus petite durée.
    """
    
    def select_action(self, legal_actions: List[int], job_progress: List[int]) -> int:
        # Pour chaque job légal, obtenir la durée de la prochaine opération
        durations = []
        for job_id in legal_actions:
            op_index = job_progress[job_id]
            _, duration = self.instance.get_operation(job_id, op_index)
            durations.append((duration, job_id))
        
        # Retourner le job avec la plus petite durée
        return min(durations)[1]


class LPTAgent(HeuristicAgent):
    """
    Longest Processing Time.
    Choisit l'opération avec la plus grande durée.
    """
    
    def select_action(self, legal_actions: List[int], job_progress: List[int]) -> int:
        durations = []
        for job_id in legal_actions:
            op_index = job_progress[job_id]
            _, duration = self.instance.get_operation(job_id, op_index)
            durations.append((duration, job_id))
        
        return max(durations)[1]


class FIFOAgent(HeuristicAgent):
    """
    First In First Out.
    Choisit le premier job dans l'ordre.
    """
    
    def select_action(self, legal_actions: List[int], job_progress: List[int]) -> int:
        return min(legal_actions)


class MostWorkRemainingAgent(HeuristicAgent):
    """
    Choisit le job avec le plus de travail restant.
    """
    
    def select_action(self, legal_actions: List[int], job_progress: List[int]) -> int:
        work_remaining = []
        for job_id in legal_actions:
            # Calculer le temps total restant pour ce job
            remaining_time = 0
            for op_idx in range(job_progress[job_id], self.instance.get_num_operations(job_id)):
                _, duration = self.instance.get_operation(job_id, op_idx)
                remaining_time += duration
            work_remaining.append((remaining_time, job_id))
        
        return max(work_remaining)[1]


# ============================================
# FONCTION D'ÉVALUATION
# ============================================

def evaluate_agent(agent: HeuristicAgent, env, num_episodes: int = 10, verbose: bool = False):
    """
    Évalue un agent heuristique.
    
    Args:
        agent: L'agent à évaluer
        env: L'environnement Job Shop
        num_episodes: Nombre d'épisodes à exécuter
        verbose: Afficher les détails
        
    Returns:
        Liste des makespans obtenus
    """
    makespans = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            legal_actions = env._get_legal_actions()
            job_progress = env.job_progress.tolist()
            
            action = agent.select_action(legal_actions, job_progress)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        makespan = info['makespan']
        makespans.append(makespan)
        
        if verbose:
            print(f"Episode {episode + 1}: Makespan = {makespan}")
    
    return makespans


# ============================================
# TESTS ET DÉMONSTRATION
# ============================================

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from src.environments.job_shop_env import JobShopEnv
    
    section("DÉMONSTRATION - Agents Heuristiques")
       
    # Créer une instance
    configuration()
    instance = JobShopInstance.create_simple_instance()
    
    # Afficher l'instance
    print("Instance:")
    print(instance)
    
    # Créer l'environnement
    env = JobShopEnv(instance)
    
    # Tester différents agents
    agents = {
        'Random': RandomAgent(instance),
        'SPT': SPTAgent(instance),
        'LPT': LPTAgent(instance),
        'FIFO': FIFOAgent(instance),
        'MostWork': MostWorkRemainingAgent(instance)
    }
    
    print("Évaluation des agents (10 épisodes chacun):")
    print("-" * 60)
    
    results = {}
    for name, agent in agents.items():
        makespans = evaluate_agent(agent, env, num_episodes=10)
        avg_makespan = sum(makespans) / len(makespans)
        best_makespan = min(makespans)
        results[name] = {
            'avg': avg_makespan,
            'best': best_makespan,
            'all': makespans
        }
        print(f"\n{name}:")
        print(f"  Makespan moyen: {avg_makespan:.2f}")
        print(f"  Meilleur makespan: {best_makespan}")
    
    section("Classement (par meilleur makespan):")
    
    sorted_agents = sorted(results.items(), key=lambda x: x[1]['best'])
    for i, (name, res) in enumerate(sorted_agents, 1):
        print(f"{i}. {name}: {res['best']} (avg: {res['avg']:.2f})")