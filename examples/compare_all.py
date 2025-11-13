"""
Comparaison de tous les agents sur différentes instances
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_instance import JobShopInstance
from src.environments.job_shop_env import JobShopEnv
from src.agents.Heuristic_agent import SPTAgent, LPTAgent, RandomAgent, FIFOAgent, MostWorkRemainingAgent, evaluate_agent
from src.agents.DQN_agent import SimpleDQNAgent
from src.agents.PPO_agent import SimplePPOAgent
from src.utils.visualization import GanttVisualizer
from src.utils.notifier import * 

"""
Comparaison finale complète : DQN vs PPO vs Heuristiques
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_env import JobShopEnv
from src.utils.benchmark_instances import BenchmarkLibrary
from src.agents.deep_DQN_agent import DeepDQNAgent
from src.agents.deep_PPO_agent import DeepPPOAgent
from src.agents.Heuristic_agent import SPTAgent, LPTAgent, evaluate_agent
from src.utils.notifier import * 

def evaluate_model(agent, env, num_episodes=20):
    """Évalue un agent RL."""
    makespans = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            legal_actions = env._get_legal_actions()
            
            if isinstance(agent, DeepPPOAgent):
                action, _, _ = agent.select_action(state, legal_actions)
            else:
                action = agent.select_action(state, legal_actions)
            
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        makespans.append(info['makespan'])
    
    return makespans

def compare_agents(instance: JobShopInstance, num_runs: int = 5):
    """
    Compare DQN et PPO sur plusieurs runs.
    """
    section("COMPARAISON DQN vs PPO")
    
    print(f"Instance: {instance.num_jobs} jobs × {instance.num_machines} machines")
    print(f"Nombre de runs: {num_runs}\n")
    
    env = JobShopEnv(instance)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    dqn_results = []
    ppo_results = []
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{num_runs}")
        print(f"{'='*60}")
        
        # ===== TEST DQN =====
        print("\nTest DQN...")
        dqn_agent = SimpleDQNAgent(state_size, action_size, epsilon=0.3)
        
        # Mini-entraînement
        makespans_dqn = []
        for ep in range(100):
            state, _ = env.reset()
            done = False
            while not done:
                legal_actions = env._get_legal_actions()
                action = dqn_agent.select_action(state, legal_actions)
                next_state, reward, terminated, truncated, info = env.step(action)
                dqn_agent.remember(state, action, reward, next_state, terminated)
                dqn_agent.learn(32)
                state = next_state
                done = terminated or truncated
            makespans_dqn.append(info['makespan'])
            dqn_agent.decay_epsilon()
        
        best_dqn = min(makespans_dqn[-20:])
        dqn_results.append(best_dqn)
        print(f"   Meilleur makespan DQN: {best_dqn:.2f}")
        
        # ===== TEST PPO =====
        print("\nTest PPO...")
        ppo_agent = SimplePPOAgent(state_size, action_size)
        
        makespans_ppo = []
        for ep in range(100):
            state, _ = env.reset()
            done = False
            while not done:
                legal_actions = env._get_legal_actions()
                action, log_prob, value = ppo_agent.select_action(state, legal_actions)
                next_state, reward, terminated, truncated, info = env.step(action)
                ppo_agent.buffer.add(state, action, reward, value, log_prob, terminated)
                state = next_state
                done = terminated or truncated
            makespans_ppo.append(info['makespan'])
            if (ep + 1) % 10 == 0:
                ppo_agent.learn(4)
        
        best_ppo = min(makespans_ppo[-20:])
        ppo_results.append(best_ppo)
        print(f"   Meilleur makespan PPO: {best_ppo:.2f}")
    
    # ===== HEURISTIQUE BASELINE =====
    section("Baseline heuristique (SPT)")
    
    spt_agent = SPTAgent(instance)
    spt_makespans = evaluate_agent(spt_agent, env, num_episodes=20, verbose=False)
    best_spt = min(spt_makespans)
    print(f"Meilleur makespan SPT: {best_spt:.2f}")
    
    # ===== RÉSULTATS =====
    
    section("RÉSULTATS FINAUX")
    
    print(f"{'Algorithme':<15} {'Moyen':<10} {'Meilleur':<10} {'Pire':<10}")
    print("-" * 45)
    print(f"{'DQN':<15} {np.mean(dqn_results):<10.2f} {min(dqn_results):<10.2f} {max(dqn_results):<10.2f}")
    print(f"{'PPO':<15} {np.mean(ppo_results):<10.2f} {min(ppo_results):<10.2f} {max(ppo_results):<10.2f}")
    print(f"{'SPT':<15} {np.mean(spt_makespans):<10.2f} {best_spt:<10.2f} {max(spt_makespans):<10.2f}")
    
    # Visualisation comparative
    print(f"\nGraphique comparatif...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = [1, 2, 3]
    data = [dqn_results, ppo_results, spt_makespans[:num_runs]]
    labels = ['DQN', 'PPO', 'SPT']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Makespan', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison DQN vs PPO vs SPT', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Sauvegarder
    save_path = Path(__file__).parent.parent / 'results' / 'plots' / 'dqn_vs_ppo_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    saved(f"Graphique sauvegardé", save_path)
    plt.show()
    
    passed("COMPARAISON TERMINÉE !")
    
    return {
        'dqn': dqn_results,
        'ppo': ppo_results,
        'spt': spt_makespans
    }

def main():
    section("COMPARAISON DE TOUS LES AGENTS")
    
    instances = ['FT06']  
    
    all_results = {}
    
    for instance_name in instances:
        
        print(f"Instance: {instance_name}")
                
        instance = BenchmarkLibrary.get_instance(instance_name)
        optimal = BenchmarkLibrary.get_optimal_makespan(instance_name)
        env = JobShopEnv(instance)
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        results = {}
        
        # ===== HEURISTIQUES =====
        print("Évaluation des heuristiques...")
        spt = SPTAgent(instance)
        lpt = LPTAgent(instance)
        
        spt_makespans = evaluate_agent(spt, env, num_episodes=20, verbose=False)
        lpt_makespans = evaluate_agent(lpt, env, num_episodes=20, verbose=False)
        
        results['SPT'] = spt_makespans
        results['LPT'] = lpt_makespans
        
        # ===== DEEP DQN =====
        dqn_path = Path(__file__).parent.parent / 'results' / 'models' / f'deep_dqn_{instance_name}.pth'
        if dqn_path.exists():
            print("Évaluation Deep DQN...")
            dqn_agent = DeepDQNAgent(state_size, action_size)
            dqn_agent.load(str(dqn_path))
            dqn_agent.epsilon = 0.0
            
            dqn_makespans = evaluate_model(dqn_agent, env, num_episodes=20)
            results['Deep DQN'] = dqn_makespans
        
        # ===== DEEP PPO =====
        ppo_path = Path(__file__).parent.parent / 'results' / 'models' / f'deep_ppo_{instance_name}.pth'
        if ppo_path.exists():
            print("Évaluation Deep PPO...")
            ppo_agent = DeepPPOAgent(state_size, action_size)
            ppo_agent.load(str(ppo_path))
            
            ppo_makespans = evaluate_model(ppo_agent, env, num_episodes=20)
            results['Deep PPO'] = ppo_makespans
        
        all_results[instance_name] = results
        
        # ===== AFFICHAGE =====
        
        print(f"RÉSULTATS {instance_name}")
                
        print(f"{'Méthode':<15} {'Meilleur':<10} {'Moyen':<10} {'Pire':<10} {'Gap %':<10}")
        print("-" * 65)
        
        print(f"{'Optimal':<15} {optimal:<10} {optimal:<10} {optimal:<10} {0.0:<10.2f}")
        
        for method, makespans in results.items():
            best = min(makespans)
            avg = np.mean(makespans)
            worst = max(makespans)
            gap = ((best - optimal) / optimal) * 100
            
            print(f"{method:<15} {best:<10.1f} {avg:<10.1f} {worst:<10.1f} {gap:<10.2f}")
        
        # ===== VISUALISATION =====
        print(f"Génération du graphique...")
        
        _, ax = plt.subplots(figsize=(12, 6))
        
        # Boxplot
        data = [results[method] for method in results.keys()]
        labels = list(results.keys())
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(labels)]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Ligne optimale
        ax.axhline(y=optimal, color='red', linestyle='--', linewidth=2, label=f'Optimal ({optimal})')
        
        ax.set_ylabel('Makespan', fontsize=14, fontweight='bold')
        ax.set_title(f'Comparaison Complète - {instance_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = Path(__file__).parent.parent / 'results' / 'plots' / f'final_comparison_{instance_name}.png'
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        saved(f"Graphique sauvegardé",save_path)
        plt.show()
    
    passed("COMPARAISON FINALE TERMINÉE !")

if __name__ == "__main__":
    main()