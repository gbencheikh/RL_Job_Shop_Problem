"""
Script d'évaluation d'un modèle entraîné
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_instance import JobShopInstance
from src.environments.job_shop_env import JobShopEnv
from src.agents.DQN_agent import SimpleDQNAgent
from src.utils.visualization import GanttVisualizer
from src.utils.notifier import *

def evaluate_model(model_path: str, instance: JobShopInstance, num_episodes: int = 10):
    """
    Évalue un modèle entraîné.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        instance: Instance à résoudre
        num_episodes: Nombre d'épisodes de test
    """
    section(f"ÉVALUATION DU MODÈLE")
    
    # Créer l'environnement
    env = JobShopEnv(instance)
    
    # Créer et charger l'agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = SimpleDQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # Pas d'exploration en évaluation
    
    print(f"Instance: {instance.num_jobs} jobs × {instance.num_machines} machines")
    print(f"Évaluation sur {num_episodes} épisodes\n")
    
    makespans = []
    best_schedule = None
    best_makespan = float('inf')
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            legal_actions = env._get_legal_actions()
            action = agent.select_action(state, legal_actions)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        makespan = info['makespan']
        makespans.append(makespan)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = env.get_schedule()
        
        print(f"Episode {episode + 1:2d}: Makespan = {makespan:.2f}")
    
    # Statistiques
    section(f"STATISTIQUES")

    print(f"Makespan moyen: {np.mean(makespans):.2f}")
    print(f"Makespan médian: {np.median(makespans):.2f}")
    print(f"Meilleur: {best_makespan:.2f}")
    print(f"Pire: {max(makespans):.2f}")
    print(f"Écart-type: {np.std(makespans):.2f}")
    print(f"{'='*60}\n")
    
    # Visualiser la meilleure solution
    section("Visualisation de la meilleure solution...")

    visualizer = GanttVisualizer()
    visualizer.plot_schedule(
        best_schedule,
        instance.num_machines,
        title=f"Meilleure Solution (Makespan = {best_makespan:.1f})"
    )
    
    return makespans, best_schedule


def main():
    section("ÉVALUATION D'UN MODÈLE DQN")
    
    # Charger l'instance (même que pour l'entraînement)
    instance = JobShopInstance.create_simple_instance()
    
    # Chemin du modèle
    model_path = Path(__file__).parent.parent / 'results' / 'models' / 'dqn_agent.pkl'
    
    if not model_path.exists():
        error(f"Erreur: Modèle non trouvé à {model_path}")
        print("   Veuillez d'abord entraîner un modèle avec train_dqn.py\n")
        return
    
    # Évaluer
    evaluate_model(str(model_path), instance, num_episodes=10)
    
    passed("Évaluation terminée !\n")


if __name__ == "__main__":
    main()