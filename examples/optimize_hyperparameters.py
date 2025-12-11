"""
Test de différentes configurations pour optimiser Deep DQN
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_env import JobShopEnv
from src.utils.benchmark_instances import BenchmarkLibrary
from src.agents.deep_DQN_agent import DeepDQNAgent
from examples.train_deep_DQN_agent import train_deep_dqn
from src.utils.notifier import * 
import numpy as np


def test_configuration(config_name, config, instance_name='FT06', num_episodes=1000):
    """
    Teste une configuration d'hyperparamètres.
    """
    section(f"Test : {config_name}")
    
    instance = BenchmarkLibrary.get_instance(instance_name)
    optimal = BenchmarkLibrary.get_optimal_makespan(instance_name)
    env = JobShopEnv(instance)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Créer l'agent avec la config
    agent = DeepDQNAgent(
        state_size=state_size,
        action_size=action_size,
        **config
    )
    
    # Entraîner
    results = train_deep_dqn(env, agent, num_episodes=num_episodes, verbose=False)
    
    best_makespan = results['best_makespan']
    gap = ((best_makespan - optimal) / optimal) * 100
    
    print(f"Résultats {config_name}:")
    print(f"   Meilleur makespan: {best_makespan:.1f}")
    print(f"   Gap: {gap:.2f}%")
    
    return {
        'config': config_name,
        'makespan': best_makespan,
        'gap': gap,
        'makespans': results['makespans']
    }


def main():
    section("OPTIMISATION DES HYPERPARAMÈTRES")
    
    # Configurations à tester
    configs = {
        'Baseline': {
            'learning_rate': 0.001,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'use_dueling': False,
            'use_double_dqn': True
        },
        'LR_Lower': {
            'learning_rate': 0.0005,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'use_dueling': False,
            'use_double_dqn': True
        },
        'Epsilon_Slower': {
            'learning_rate': 0.001,
            'epsilon_decay': 0.998,
            'batch_size': 64,
            'use_dueling': False,
            'use_double_dqn': True
        },
        'Batch_Larger': {
            'learning_rate': 0.001,
            'epsilon_decay': 0.995,
            'batch_size': 128,
            'use_dueling': False,
            'use_double_dqn': True
        },
        'Dueling_DQN': {
            'learning_rate': 0.001,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'use_dueling': True,
            'use_double_dqn': True
        },
        'Best_Combined': {
            'learning_rate': 0.0005,
            'epsilon_decay': 0.998,
            'batch_size': 128,
            'use_dueling': True,
            'use_double_dqn': True
        }
    }
    
    results = []
    
    for config_name, config in configs.items():
        result = test_configuration(config_name, config, num_episodes=500)
        results.append(result)
    
    # Résumé
    section(f"RÉSUMÉ - Classement par Performance")
    
    results.sort(key=lambda x: x['makespan'])
    
    print(f"{'Configuration':<20} {'Makespan':<12} {'Gap %':<10}")
    print("-" * 42)
    for r in results:
        print(f"{r['config']:<20} {r['makespan']:<12.1f} {r['gap']:<10.2f}")
    
    success(f"Meilleure configuration:, results[0]['config']")

if __name__ == "__main__":
    main()