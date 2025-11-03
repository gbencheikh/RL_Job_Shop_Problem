import sys
from pathlib import Path
import numpy as np
import time

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_instance import JobShopInstance
from src.environments.job_shop_env import JobShopEnv
from src.agents.PPO_agent import SimplePPOAgent
from src.agents.Heuristic_agent import SPTAgent, LPTAgent, RandomAgent, evaluate_agent
from src.utils.visualization import GanttVisualizer, plot_training_curve
from src.utils.logger import SimpleLogger
from src.utils.notifier import * 

def train_PPO_agent(env, agent, num_episodes: int = 1000, 
                   update_frequency: int = 10, 
                   verbose: bool = True):
    """
    Entraîne un agent PPO.
    
    Args:
        env: Environnement Job Shop
        agent: Agent PPO
        num_episodes: Nombre d'épisodes
        update_frequency: Fréquence de mise à jour (en épisodes)
        verbose: Afficher les progrès
        
    Returns:
        Historique de l'entraînement
    """
    import time
    
    episode_rewards = []
    episode_makespans = []
    best_makespan = float('inf')
    best_schedule = None
    
    section(f"ENTRAÎNEMENT PPO")

    print(f"Episodes: {num_episodes}")
    print(f"Mise à jour toutes les {update_frequency} épisodes")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Sélectionner une action
            legal_actions = env._get_legal_actions()
            action, log_prob, value = agent.select_action(state, legal_actions)
            
            # Exécuter l'action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Stocker dans le buffer
            agent.buffer.add(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
        
        # Enregistrer les stats
        makespan = info['makespan']
        episode_rewards.append(episode_reward)
        episode_makespans.append(makespan)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = env.get_schedule()
        
        # Mise à jour de la politique
        if (episode + 1) % update_frequency == 0:
            agent.learn(epochs=4)
        
        # Affichage
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_makespan = np.mean(episode_makespans[-50:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Makespan: {avg_makespan:6.2f} | "
                  f"Best: {best_makespan:6.2f} | "
                  f"Temps: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    passed(f"ENTRAÎNEMENT PPO TERMINÉ")
    
    print(f"Temps total: {total_time:.2f}s")
    print(f"Meilleur makespan: {best_makespan}")
    print(f"{'='*60}\n")
    
    agent.training_history['episode_rewards'] = episode_rewards
    agent.training_history['episode_makespans'] = episode_makespans
    
    return {
        'rewards': episode_rewards,
        'makespans': episode_makespans,
        'best_makespan': best_makespan,
        'best_schedule': best_schedule
    }

def main():
    
    section("JOB SHOP SCHEDULING avec PPO")
    
    
    # ========================================
    # 1. CONFIGURATION
    # ========================================
    
    configuration()
    
    # Instance
    instance = JobShopInstance.create_simple_instance()
    print(f"Instance:")
    print(instance)
    
    # Environnement
    env = JobShopEnv(instance)
    
    # Agent PPO
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = SimplePPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2
    )
    
    passed(f"Agent PPO créé:")
    print(f"   - Dimension état: {state_size}")
    print(f"   - Nombre d'actions: {action_size}")
    print(f"   - Learning rate: {agent.learning_rate}")
    
    # ========================================
    # 2. ENTRAÎNEMENT
    # ========================================
    
    section("2. ENTRAÎNEMENT")

    num_episodes = 500
    training_results = train_PPO_agent(
        env,
        agent,
        num_episodes=num_episodes,
        update_frequency=10,  
        verbose=True
    )
    
    # ========================================
    # 3. SAUVEGARDE
    # ========================================
    
    section("Sauvegarde du modèle...")
    save_dir = Path(__file__).parent.parent / 'results' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_dir / 'ppo_agent.pkl'))
    
    # ========================================
    # 4. COMPARAISON
    # ========================================
    
    section(f"3. COMPARAISON AVEC HEURISTIQUES")
        
    heuristics = {
        'SPT': SPTAgent(instance),
        'LPT': LPTAgent(instance),
        'Random': RandomAgent(instance)
    }
    
    print(f"{'Agent':<15} {'Meilleur':<10} {'Moyen':<10}")
    print("-" * 35)
    
    for name, heuristic in heuristics.items():
        makespans = evaluate_agent(heuristic, env, num_episodes=10, verbose=False)
        print(f"{name:<15} {min(makespans):<10.2f} {np.mean(makespans):<10.2f}")
    
    print(f"{'PPO':<15} {training_results['best_makespan']:<10.2f} {np.mean(training_results['makespans'][-100:]):<10.2f}")
    print(f"{'='*35}\n")
    
    # ========================================
    # 5. VISUALISATIONS
    # ========================================
    
    section("4. VISUALISATIONS")
    
    visualizer = GanttVisualizer()
    plots_dir = Path(__file__).parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Gantt
    print("Diagramme de Gantt - Meilleure solution PPO")
    visualizer.plot_schedule(
        training_results['best_schedule'],
        instance.num_machines,
        title=f"Meilleure Solution PPO (Makespan = {training_results['best_makespan']:.1f})",
        save_path=str(plots_dir / 'ppo_best_solution.png')
    )
    
    # Courbe d'apprentissage
    print("\nCourbe d'apprentissage")
    rewards = [-m for m in training_results['makespans']]
    plot_training_curve(
        rewards,
        window_size=50,
        title="Apprentissage PPO",
        save_path=str(plots_dir / 'ppo_training_curve.png')
    )
    
if __name__ == "__main__":
    main()