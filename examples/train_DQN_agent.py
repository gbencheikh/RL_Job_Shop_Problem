"""
Script d'entraînement d'un agent DQN sur le Job Shop Problem

Ce script :
1. Crée une instance Job Shop
2. Initialise l'environnement et l'agent DQN
3. Entraîne l'agent sur plusieurs épisodes
4. Compare avec des heuristiques
5. Visualise les résultats
"""

import sys
from pathlib import Path
import numpy as np
import time

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_instance import JobShopInstance
from src.environments.job_shop_env import JobShopEnv
from src.agents.DQN_agent import SimpleDQNAgent
from src.agents.Heuristic_agent import SPTAgent, LPTAgent, RandomAgent, evaluate_agent
from src.utils.visualization import GanttVisualizer, plot_training_curve
from src.utils.logger import SimpleLogger
from src.utils.notifier import * 

def train_DQN_agent(env, agent, num_episodes=1000, batch_size=32, verbose=True):
    """
    Entraîne un agent DQN.
    
    Args:
        env: Environnement Job Shop
        agent: Agent DQN
        num_episodes: Nombre d'épisodes d'entraînement
        batch_size: Taille du batch pour l'apprentissage
        verbose: Afficher les progrès
        
    Returns:
        Historique de l'entraînement
    """
    episode_rewards = []
    episode_makespans = []
    best_makespan = float('inf')
    best_schedule = None
    
    print(f"DÉBUT DE L'ENTRAÎNEMENT")
    
    print(f"Episodes: {num_episodes}")
    print(f"Epsilon initial: {agent.epsilon:.3f}")
    print(f"Taux d'apprentissage: {agent.learning_rate}")
    print(f"{'='*60}\n")
    
    logger = SimpleLogger()

    start_time = time.time()
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Sélectionner une action
            legal_actions = env._get_legal_actions()
            action = agent.select_action(state, legal_actions)
            
            # Exécuter l'action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Stocker la transition
            agent.remember(state, action, reward, next_state, done)
            
            # Apprendre
            agent.learn(batch_size)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        logger.info(f"Episode {episode} reward {episode_reward:.3f} eps {agent.epsilon:.3f}")

        # Enregistrer les statistiques
        makespan = info['makespan']
        episode_rewards.append(episode_reward)
        episode_makespans.append(makespan)
        
        # Sauvegarder le meilleur ordonnancement
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = env.get_schedule()
        
        # Réduire epsilon
        agent.decay_epsilon()
        
        # Affichage
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_makespan = np.mean(episode_makespans[-50:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Makespan: {avg_makespan:6.2f} | "
                  f"Best: {best_makespan:6.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Temps: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    logger.info("Training finished.")
    
    passed(f"ENTRAÎNEMENT TERMINÉ")
    
    print(f"{'='*60}")
    print(f"Temps total: {total_time:.2f}s ({total_time/num_episodes:.3f}s/épisode)")
    print(f"Meilleur makespan: {best_makespan}")
    print(f"{'='*60}\n")
    
    # Stocker l'historique dans l'agent
    agent.training_history['episode_rewards'] = episode_rewards
    agent.training_history['episode_makespans'] = episode_makespans
    agent.training_history['epsilon_history'] = [agent.epsilon] * num_episodes  # Simplifié
    
    return {
        'rewards': episode_rewards,
        'makespans': episode_makespans,
        'best_makespan': best_makespan,
        'best_schedule': best_schedule
    }


def compare_with_heuristics(instance, env, dqn_makespan):
    """
    Compare l'agent DQN avec des heuristiques classiques.
    
    Args:
        instance: Instance Job Shop
        env: Environnement
        dqn_makespan: Meilleur makespan du DQN
        
    Returns:
        Dictionnaire des résultats
    """
    
    heuristics = {
        'SPT': SPTAgent(instance),
        'LPT': LPTAgent(instance),
        'Random': RandomAgent(instance)
    }
    
    results = {}
    
    for name, agent in heuristics.items():
        makespans = evaluate_agent(agent, env, num_episodes=10, verbose=False)
        avg_makespan = np.mean(makespans)
        best_makespan = min(makespans)
        
        results[name] = {
            'avg': avg_makespan,
            'best': best_makespan
        }
        
        print(f"{name:10s} | Moyen: {avg_makespan:6.2f} | Meilleur: {best_makespan:6.2f}")
    
    # Ajouter DQN aux résultats
    results['DQN'] = {
        'avg': dqn_makespan,  # On n'a qu'un seul run
        'best': dqn_makespan
    }
    
    print(f"{'DQN':10s} | Meilleur: {dqn_makespan:6.2f}")
    
    return results


def main():
    """Fonction principale d'entraînement."""
    """
    1. Crée une instance Job Shop
    2. Initialise l'environnement et l'agent DQN
    3. Entraîne l'agent sur plusieurs épisodes
    4. Compare avec des heuristiques
    5. Visualise les résultats
    """
    section("JOB SHOP SCHEDULING avec DEEP Q-LEARNING")
    
    # ========================================
    # 1. CONFIGURATION
    # ========================================
    
    section("1. Crée une instance Job Shop")
    configuration() 
    instance = JobShopInstance.create_simple_instance()
    # Pour une instance plus complexe :
    # instance = JobShopInstance.create_random_instance(3, 3, 1, 10)
    
    # Afficher l'instance
    print(f"\nInstance:")
    print(instance)
    
    # ========================================
    # 2. Initialise l'environnement et l'agent DQN
    # ========================================
    section("2. Initialise l'environnement et l'agent DQN")
    configuration() 

    # Créer l'environnement
    env = JobShopEnv(instance)
    
    # Créer l'agent DQN
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = SimpleDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    passed(f"Agent DQN créé:")
    print(f"   - Dimension état: {state_size}")
    print(f"   - Nombre d'actions: {action_size}")
    
    # ========================================
    # 3. ENTRAÎNEMENT
    # ========================================
    section("3. Entraîne l'agent sur plusieurs épisodes")
    num_episodes = 500
    training_results = train_DQN_agent(
        env, 
        agent, 
        num_episodes=num_episodes,
        batch_size=32,
        verbose=True
    )
    
    # SAUVEGARDE DU MODÈLE
    
    print("Sauvegarde du modèle...")
    save_dir = Path(__file__).parent.parent / 'results' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_dir / 'dqn_agent.pkl'))
    
    # ========================================
    # 4. COMPARAISON AVEC HEURISTIQUES
    # ========================================
    section("4. COMPARAISON AVEC HEURISTIQUES")

    comparison_results = compare_with_heuristics(
        instance, 
        env, 
        training_results['best_makespan']
    )
    
    # ========================================
    # 5. VISUALISATIONS
    # ========================================
    
    section("5. VISUALISATIONS")
    
    visualizer = GanttVisualizer()
    
    # Diagramme de Gantt de la meilleure solution
    print("Diagramme de Gantt - Meilleure solution DQN")
    save_path = Path(__file__).parent.parent / 'results' / 'plots' / 'best_solution_gantt.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualizer.plot_schedule(
        training_results['best_schedule'],
        instance.num_machines,
        title=f"Meilleure Solution DQN (Makespan = {training_results['best_makespan']:.1f})",
        save_path=str(save_path)
    )
    
    # Courbe d'apprentissage
    print("\nCourbe d'apprentissage")
    # Convertir makespans en récompenses négatives pour cohérence
    rewards = [-m for m in training_results['makespans']]
    save_path = Path(__file__).parent.parent / 'results' / 'plots' / 'training_curve.png'
    
    plot_training_curve(
        rewards,
        window_size=min(50, num_episodes // 10),
        title="Apprentissage de l'Agent DQN",
        save_path=str(save_path)
    )
    
    success("ENTRAÎNEMENT COMPLET TERMINÉ !")


if __name__ == "__main__":
    main()