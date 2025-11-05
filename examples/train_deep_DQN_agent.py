"""
Entraînement Deep DQN sur Job Shop
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_env import JobShopEnv
from src.utils.benchmark_instances import BenchmarkLibrary
from src.agents.deep_DQN_agent import DeepDQNAgent
from src.agents.Heuristic_agent import SPTAgent, evaluate_agent
from src.utils.visualization import GanttVisualizer, plot_training_curve
import numpy as np
from src.utils.notifier import * 

# ============================================
# FONCTION D'ENTRAÎNEMENT
# ============================================

def train_deep_dqn(env, agent, num_episodes: int = 1000, verbose: bool = True):
    """
    Entraîne un agent Deep DQN.
    
    Args:
        env: Environnement Job Shop
        agent: Agent DeepDQNAgent
        num_episodes: Nombre d'épisodes
        verbose: Afficher les progrès
        
    Returns:
        Historique de l'entraînement
    """
    import time
    
    episode_rewards = []
    episode_makespans = []
    losses = []
    best_makespan = float('inf')
    best_schedule = None
    
    section(f"ENTRAÎNEMENT DEEP DQN")

    print(f"Episodes: {num_episodes}")
    print(f"Batch size: {agent.batch_size}")
    print(f"Target update freq: {agent.target_update_freq}")
    print(f"Double DQN: {agent.use_double_dqn}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Sélectionner action
            legal_actions = env._get_legal_actions()
            action = agent.select_action(state, legal_actions)
            
            # Exécuter action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Stocker transition
            agent.remember(state, action, reward, next_state, done)
            
            # Apprendre
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
        
        # Statistiques
        makespan = info['makespan']
        episode_rewards.append(episode_reward)
        episode_makespans.append(makespan)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = env.get_schedule()
        
        # Decay epsilon
        agent.decay_epsilon()
        agent.episodes += 1
        
        # Affichage
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_makespan = np.mean(episode_makespans[-50:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            elapsed = time.time() - start_time
            
            print(f"Ep {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Makespan: {avg_makespan:6.2f} | "
                  f"Best: {best_makespan:6.2f} | "
                  f"Loss: {avg_loss:6.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.memory):5d} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    passed(f"ENTRAÎNEMENT TERMINÉ")

    print(f"Temps total: {total_time:.2f}s")
    print(f"Meilleur makespan: {best_makespan}")
    print(f"Buffer final: {len(agent.memory)} transitions")
    print(f"{'='*60}\n")
    
    agent.training_history['episode_rewards'] = episode_rewards
    agent.training_history['episode_makespans'] = episode_makespans
    agent.training_history['losses'] = losses
    
    return {
        'rewards': episode_rewards,
        'makespans': episode_makespans,
        'losses': losses,
        'best_makespan': best_makespan,
        'best_schedule': best_schedule
    }


def main():
    section("🧠 JOB SHOP avec DEEP Q-LEARNING (PyTorch)")
    
    # ========================================
    # 1. CHOISIR L'INSTANCE
    # ========================================
    
    instance_name = 'FT06'  # Commencer avec FT06
    print(f"Instance: {instance_name}")
    instance = BenchmarkLibrary.get_instance(instance_name)
    optimal = BenchmarkLibrary.get_optimal_makespan(instance_name)
    print(instance)
    print(f"Makespan optimal connu: {optimal}\n")
    
    # ========================================
    # 2. CRÉER L'ENVIRONNEMENT ET L'AGENT
    # ========================================
    
    env = JobShopEnv(instance)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DeepDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        target_update_freq=100,
        use_double_dqn=True
    )
    
    print(f"Agent Deep DQN:")
    print(f"   - Paramètres: {sum(p.numel() for p in agent.policy_net.parameters())}")
    print(f"   - Device: {agent.device}")
    
    # ========================================
    # 3. ENTRAÎNEMENT
    # ========================================
    
    num_episodes = 1000
    results = train_deep_dqn(env, agent, num_episodes=num_episodes, verbose=True)
    
    # ========================================
    # 4. SAUVEGARDE
    # ========================================
    
    save_dir = Path(__file__).parent.parent / 'results' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_dir / f'deep_dqn_{instance_name}.pth'))
    
    # ========================================
    # 5. COMPARAISON
    # ========================================
    
    section(f"📊 COMPARAISON")
    
    spt = SPTAgent(instance)
    spt_makespans = evaluate_agent(spt, env, num_episodes=10, verbose=False)
    
    print(f"{'Méthode':<15} {'Meilleur':<10} {'Gap %':<10}")
    print("-" * 35)
    print(f"{'Optimal':<15} {optimal:<10} {0.0:<10.2f}")
    print(f"{'SPT':<15} {min(spt_makespans):<10.1f} {((min(spt_makespans)-optimal)/optimal*100):<10.2f}")
    print(f"{'Deep DQN':<15} {results['best_makespan']:<10.1f} {((results['best_makespan']-optimal)/optimal*100):<10.2f}")
    
    # ========================================
    # 6. VISUALISATIONS
    # ========================================
    
    print(f"Visualisations...")
    
    visualizer = GanttVisualizer()
    plots_dir = Path(__file__).parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Gantt
    visualizer.plot_schedule(
        results['best_schedule'],
        instance.num_machines,
        title=f"Deep DQN - {instance_name} (Makespan = {results['best_makespan']:.1f})",
        save_path=str(plots_dir / f'deep_dqn_{instance_name}_gantt.png')
    )
    
    # Courbe d'apprentissage
    rewards = [-m for m in results['makespans']]
    plot_training_curve(
        rewards,
        window_size=50,
        title=f"Apprentissage Deep DQN - {instance_name}",
        save_path=str(plots_dir / f'deep_dqn_{instance_name}_training.png')
    )
    
    success("ENTRAÎNEMENT DEEP DQN TERMINÉ !")

if __name__ == "__main__":
    main()