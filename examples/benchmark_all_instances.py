"""
Benchmark complet de tous les agents sur toutes les instances
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.notifier import * 
from src.utils.benchmark_instances import BenchmarkLibrary, evaluate_on_benchmark
from src.agents.Heuristic_agent import (
    SPTAgent, LPTAgent, RandomAgent, 
    FIFOAgent, MostWorkRemainingAgent
)
from src.environments.job_shop_instance import JobShopInstance


def benchmark_heuristics():
    """
    Benchmark de toutes les heuristiques sur toutes les instances.
    """
    section("BENCHMARK COMPLET - HEURISTIQUES SUR INSTANCES CLASSIQUES")
    
    instances = BenchmarkLibrary.list_instances()
    
    # Créer les agents
    results_by_instance = {}
    
    for instance_name in instances:
        print(f"\n{'='*70}")
        print(f"Instance: {instance_name}")
        info = BenchmarkLibrary.get_instance_info(instance_name)
        jobs, machines = info['size']
        optimal = info['optimal']
        print(f"{'='*70}")
        print(f"Taille: {jobs} jobs × {machines} machines")
        print(f"Makespan optimal connu: {optimal}")
        print(f"{'-'*70}\n")
        
        instance = BenchmarkLibrary.get_instance(instance_name)
        
        # Agents heuristiques
        agents = {
            'SPT': SPTAgent(instance),
            'LPT': LPTAgent(instance),
            'FIFO': FIFOAgent(instance),
            'MWR': MostWorkRemainingAgent(instance),
            'Random': RandomAgent(instance)
        }
        
        results = {}
        
        print(f"{'Agent':<10} {'Meilleur':<10} {'Moyen':<10} {'Pire':<10} {'Gap%':<10}")
        print("-" * 70)
        
        for name, agent in agents.items():
            res = evaluate_on_benchmark(agent, instance_name, num_episodes=20)
            results[name] = res
            
            print(f"{name:<10} "
                  f"{res['best']:<10.1f} "
                  f"{res['avg']:<10.1f} "
                  f"{res['worst']:<10.1f} "
                  f"{res['gap_best']:<10.2f}")
        
        results_by_instance[instance_name] = results
    
    # ========================================
    # RÉSUMÉ GLOBAL
    # ========================================
    
    section("RÉSUMÉ GLOBAL")
    
    # Calculer le gap moyen par agent
    agent_names = ['SPT', 'LPT', 'FIFO', 'MWR', 'Random']
    avg_gaps = {name: [] for name in agent_names}
    
    for instance_name, results in results_by_instance.items():
        for agent_name in agent_names:
            gap = results[agent_name]['gap_best']
            if gap is not None:
                avg_gaps[agent_name].append(gap)
    
    # Afficher le classement
    print(f"{'Agent':<10} {'Gap moyen (%)':<15} {'Meilleur sur':<15}")
    print("-" * 70)
    
    for name in agent_names:
        gaps = avg_gaps[name]
        avg_gap = np.mean(gaps) if gaps else float('inf')
        
        # Compter sur combien d'instances cet agent est le meilleur
        best_count = 0
        for results in results_by_instance.values():
            best_agent = min(results.items(), key=lambda x: x[1]['best'])
            if best_agent[0] == name:
                best_count += 1
        
        print(f"{name:<10} {avg_gap:<15.2f} {best_count}/{len(instances)}")
    
    # ========================================
    # VISUALISATION
    # ========================================
    
    print("\nGénération du graphique comparatif...\n")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(instances))
    width = 0.15
    
    for i, agent_name in enumerate(agent_names):
        gaps = [results_by_instance[inst][agent_name]['gap_best'] 
                for inst in instances]
        offset = width * (i - 2)
        ax.bar(x + offset, gaps, width, label=agent_name, alpha=0.8)
    
    ax.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap par rapport à l\'optimal (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Heuristiques sur Instances Benchmark', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Optimal')
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent.parent / 'results' / 'plots' / 'benchmark_heuristics.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    saved("Graphique sauvegardé",save_path)
    plt.show()
    
    passed("BENCHMARK TERMINÉ !")
    
    return results_by_instance


if __name__ == "__main__":
    benchmark_heuristics()