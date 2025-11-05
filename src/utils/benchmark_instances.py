"""
Benchmark Instances - Instances classiques du Job Shop Problem

Ce module contient les instances benchmark les plus célèbres :
- Fisher & Thompson (FT06, FT10, FT20)
- Lawrence (LA01-LA40)
- Applegate & Cook (ORB01-ORB10)
- Taillard (TA01-TA80)

Format des fichiers :
Ligne 1 : num_jobs num_machines
Lignes suivantes : pour chaque job
    machine1 duration1 machine2 duration2 ...
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from typing import List
from src.environments.job_shop_instance import JobShopInstance
from src.environments.job_shop_env import JobShopEnv
from typing import Dict, List, Tuple

def small_example() -> JobShopInstance:
    # Exemple très simple (3 jobs, 3 machines)
    jobs = [
        [(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)]
    ]
    return JobShopInstance(jobs=jobs, n_machines=3, name="small_example")

def random_benchmark(seed: int = 0):
    return JobShopInstance.generate_random_instance(n_jobs=5, n_machines=4, max_ops_per_job=4, max_duration=10, seed=seed)

# ============================================
# INSTANCES FISHER & THOMPSON
# ============================================

# FT06 : 6 jobs × 6 machines
# Makespan optimal connu : 55
FT06_DATA = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)]
]

# FT10 : 10 jobs × 10 machines
# Makespan optimal connu : 930
FT10_DATA = [
    [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
    [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
    [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
    [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
    [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
    [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
    [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
    [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
    [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)],
    [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)]
]

# FT20 : 20 jobs × 5 machines
# Makespan optimal connu : 1165
FT20_DATA = [
    [(0, 29), (1, 9), (2, 49), (3, 62), (4, 44)],
    [(0, 43), (2, 75), (3, 69), (1, 46), (4, 72)],
    [(1, 91), (0, 39), (2, 90), (4, 12), (3, 45)],
    [(1, 81), (2, 71), (0, 9), (4, 85), (3, 22)],
    [(2, 14), (0, 22), (1, 26), (3, 21), (4, 72)],
    [(2, 84), (1, 52), (4, 48), (0, 47), (3, 6)],
    [(1, 46), (0, 61), (2, 32), (3, 32), (4, 30)],
    [(2, 31), (0, 46), (1, 32), (3, 19), (4, 36)],
    [(0, 76), (1, 76), (3, 85), (2, 40), (4, 26)],
    [(1, 85), (0, 61), (2, 64), (3, 47), (4, 90)],
    [(0, 78), (2, 36), (1, 11), (4, 56), (3, 21)],
    [(2, 90), (0, 11), (1, 28), (3, 46), (4, 30)],
    [(0, 85), (2, 74), (1, 10), (3, 89), (4, 33)],
    [(2, 95), (1, 99), (0, 52), (3, 98), (4, 43)],
    [(0, 6), (1, 61), (4, 69), (2, 49), (3, 53)],
    [(1, 2), (0, 95), (2, 72), (4, 65), (3, 25)],
    [(0, 37), (2, 13), (1, 21), (3, 89), (4, 55)],
    [(0, 86), (1, 74), (4, 88), (2, 48), (3, 79)],
    [(1, 69), (0, 51), (2, 11), (3, 89), (4, 74)],
    [(0, 13), (1, 7), (2, 76), (3, 52), (4, 45)]
]


# ============================================
# INSTANCES LAWRENCE (LA)
# ============================================

# LA01 : 10 jobs × 5 machines
# Makespan optimal : 666
LA01_DATA = [
    [(0, 21), (1, 53), (2, 95), (3, 55), (4, 34)],
    [(0, 21), (3, 52), (4, 16), (2, 26), (1, 71)],
    [(1, 39), (3, 98), (0, 42), (2, 31), (4, 12)],
    [(1, 77), (0, 55), (4, 79), (2, 66), (3, 77)],
    [(0, 83), (3, 34), (2, 64), (1, 19), (4, 37)],
    [(1, 54), (2, 43), (4, 79), (0, 92), (3, 62)],
    [(3, 69), (4, 77), (1, 87), (2, 87), (0, 93)],
    [(2, 38), (0, 60), (1, 41), (3, 24), (4, 83)],
    [(3, 17), (1, 49), (4, 25), (0, 44), (2, 98)],
    [(4, 77), (3, 79), (2, 43), (1, 75), (0, 96)]
]

# LA02 : 10 jobs × 5 machines
# Makespan optimal : 655
LA02_DATA = [
    [(0, 20), (3, 87), (1, 31), (4, 76), (2, 17)],
    [(4, 25), (2, 32), (0, 24), (1, 18), (3, 81)],
    [(1, 72), (2, 23), (4, 28), (0, 58), (3, 99)],
    [(2, 86), (1, 76), (4, 97), (0, 45), (3, 90)],
    [(4, 27), (0, 42), (3, 48), (2, 17), (1, 46)],
    [(1, 67), (0, 98), (4, 48), (3, 27), (2, 62)],
    [(4, 28), (1, 12), (3, 19), (0, 80), (2, 50)],
    [(1, 63), (0, 94), (2, 98), (3, 50), (4, 80)],
    [(4, 14), (0, 75), (2, 50), (1, 41), (3, 55)],
    [(4, 72), (2, 18), (1, 37), (3, 79), (0, 61)]
]


# ============================================
# CLASSE DE GESTION DES BENCHMARKS
# ============================================

class BenchmarkLibrary:
    """
    Bibliothèque d'instances benchmark pour Job Shop.
    """
    
    INSTANCES = {
        'FT06': {'data': FT06_DATA, 'optimal': 55, 'size': (6, 6)},
        'FT10': {'data': FT10_DATA, 'optimal': 930, 'size': (10, 10)},
        'FT20': {'data': FT20_DATA, 'optimal': 1165, 'size': (20, 5)},
        'LA01': {'data': LA01_DATA, 'optimal': 666, 'size': (10, 5)},
        'LA02': {'data': LA02_DATA, 'optimal': 655, 'size': (10, 5)},
    }
    
    @classmethod
    def get_instance(cls, name: str) -> JobShopInstance:
        """
        Récupère une instance benchmark.
        
        Args:
            name: Nom de l'instance (ex: 'FT06', 'LA01')
            
        Returns:
            Instance JobShopInstance
        """
        if name not in cls.INSTANCES:
            available = ', '.join(cls.INSTANCES.keys())
            raise ValueError(f"Instance '{name}' inconnue. Disponibles: {available}")
        
        data = cls.INSTANCES[name]['data']
        return JobShopInstance(data)
    
    @classmethod
    def get_optimal_makespan(cls, name: str) -> int:
        """Retourne le makespan optimal connu."""
        if name not in cls.INSTANCES:
            return None
        return cls.INSTANCES[name]['optimal']
    
    @classmethod
    def get_instance_info(cls, name: str) -> Dict:
        """Retourne les informations sur une instance."""
        if name not in cls.INSTANCES:
            return None
        return cls.INSTANCES[name]
    
    @classmethod
    def list_instances(cls) -> List[str]:
        """Liste toutes les instances disponibles."""
        return list(cls.INSTANCES.keys())
    
    @classmethod
    def get_all_instances(cls) -> Dict[str, JobShopInstance]:
        """Retourne toutes les instances."""
        return {name: cls.get_instance(name) for name in cls.INSTANCES.keys()}


# ============================================
# FONCTION D'ÉVALUATION SUR BENCHMARKS
# ============================================

def evaluate_on_benchmark(agent, instance_name: str, num_episodes: int = 10) -> Dict:
    """
    Évalue un agent sur une instance benchmark.
    
    Args:
        agent: Agent à évaluer
        instance_name: Nom de l'instance
        num_episodes: Nombre d'épisodes
        
    Returns:
        Dictionnaire avec résultats et statistiques
    """
    from ..environments.job_shop_env import JobShopEnv
    import numpy as np
    
    # Charger l'instance
    instance = BenchmarkLibrary.get_instance(instance_name)
    optimal = BenchmarkLibrary.get_optimal_makespan(instance_name)
    
    # Créer l'environnement
    env = JobShopEnv(instance)
    
    makespans = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            legal_actions = env._get_legal_actions()
            
            # Sélectionner action selon le type d'agent
            if hasattr(agent, 'select_action'):
                # Heuristique
                action = agent.select_action(legal_actions, env.job_progress.tolist())
            else:
                # RL agent (à adapter selon votre implémentation)
                action = agent.select_action(state, legal_actions)
            
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        makespans.append(info['makespan'])
    
    # Statistiques
    avg_makespan = np.mean(makespans)
    best_makespan = min(makespans)
    worst_makespan = max(makespans)
    std_makespan = np.std(makespans)
    
    # Gap par rapport à l'optimal
    gap_avg = ((avg_makespan - optimal) / optimal) * 100 if optimal else None
    gap_best = ((best_makespan - optimal) / optimal) * 100 if optimal else None
    
    return {
        'instance': instance_name,
        'optimal': optimal,
        'avg': avg_makespan,
        'best': best_makespan,
        'worst': worst_makespan,
        'std': std_makespan,
        'gap_avg': gap_avg,
        'gap_best': gap_best,
        'makespans': makespans
    }


# ============================================
# TESTS ET DÉMONSTRATION
# ============================================

if __name__ == "__main__":
    
    from src.utils.notifier import * 

    section("BIBLIOTHÈQUE D'INSTANCES BENCHMARK")
    
    # Lister les instances
    print("Instances disponibles:")
    print("-" * 60)
    for name in BenchmarkLibrary.list_instances():
        info = BenchmarkLibrary.get_instance_info(name)
        jobs, machines = info['size']
        optimal = info['optimal']
        print(f"  {name:6s} : {jobs:2d} jobs × {machines:2d} machines | Optimal = {optimal:4d}")
    
    print()
    
    # Charger et afficher FT06
    section("Détails de FT06")
    instance = BenchmarkLibrary.get_instance('FT06')
    print(instance)
    
    optimal = BenchmarkLibrary.get_optimal_makespan('FT06')
    print(f"Makespan optimal connu : {optimal}")
    
    success("Bibliothèque chargée avec succès !")