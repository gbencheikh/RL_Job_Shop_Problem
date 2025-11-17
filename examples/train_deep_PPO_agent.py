import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.environments.job_shop_env import JobShopEnv
from src.utils.benchmark_instances import BenchmarkLibrary
from src.agents.deep_PPO_agent import DeepPPOAgent
from src.agents.Heuristic_agent import SPTAgent, evaluate_agent
from src.utils.visualization import GanttVisualizer, plot_training_curve
import numpy as np
from src.utils.notifier import * 

# ============================================
# FONCTION D'ENTRAÎNEMENT
# ============================================

def train_deep_PPO(env, agent, num_episodes: int = 1000, 
                   update_frequency: int = 10, verbose: bool = True):
    """
    Entraîne un agent Deep PPO.
    
    Args:
        env: Environnement Job Shop
        agent: Agent DeepPPOAgent
        num_episodes: Nombre d'épisodes
        update_frequency: Fréquence de mise à jour (en épisodes)
        verbose: Afficher les progrès
        
    Returns:
        Historique de l'entraînement
    """
    import time
    
    episode_rewards = []
    episode_makespans = []
    policy_losses = []
    value_losses = []
    entropies = []
    best_makespan = float('inf')
    best_schedule = None
    
    
    section(f"ENTRAÎNEMENT DEEP PPO")
    
    print(f"Episodes: {num_episodes}")
    print(f"Update frequency: {update_frequency}")
    print(f"Batch size: {agent.batch_size}")
    print(f"Clip epsilon: {agent.clip_epsilon}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Sélectionner action
            legal_actions = env._get_legal_actions()
            action, log_prob, value = agent.select_action(state, legal_actions)
            
            # Exécuter action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Stocker dans le buffer
            agent.buffer.store(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
        
        # Statistiques
        makespan = info['makespan']
        episode_rewards.append(episode_reward)
        episode_makespans.append(makespan)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = env.get_schedule()
        
        # Mise à jour de la politique
        if (episode + 1) % update_frequency == 0:
            policy_loss, value_loss, entropy = agent.update()
            if policy_loss is not None:
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
        
        agent.episodes += 1
        
        # Affichage
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_makespan = np.mean(episode_makespans[-50:])
            avg_p_loss = np.mean(policy_losses[-10:]) if policy_losses else 0
            avg_v_loss = np.mean(value_losses[-10:]) if value_losses else 0
            avg_entropy = np.mean(entropies[-10:]) if entropies else 0
            elapsed = time.time() - start_time
            
            print(f"Ep {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Makespan: {avg_makespan:6.2f} | "
                  f"Best: {best_makespan:6.2f} | "
                  f"P_loss: {avg_p_loss:6.4f} | "
                  f"V_loss: {avg_v_loss:6.4f} | "
                  f"Entropy: {avg_entropy:5.3f} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    
    passed(f"ENTRAÎNEMENT TERMINÉ")
    
    print(f"Temps total: {total_time:.2f}s")
    print(f"Meilleur makespan: {best_makespan}")
    print(f"{'='*60}\n")
    
    agent.training_history['episode_rewards'] = episode_rewards
    agent.training_history['episode_makespans'] = episode_makespans
    agent.training_history['policy_losses'] = policy_losses
    agent.training_history['value_losses'] = value_losses
    agent.training_history['entropies'] = entropies
    
    return {
        'rewards': episode_rewards,
        'makespans': episode_makespans,
        'best_makespan': best_makespan,
        'best_schedule': best_schedule,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies
    }


"""
Entraînement Deep PPO sur Job Shop
"""

def main():
    section("JOB SHOP avec DEEP PPO (PyTorch)")
    
    # ========================================
    # 1. CHOISIR L'INSTANCE
    # ========================================
    
    instance_name = 'FT06'
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
    
    agent = DeepPPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        batch_size=64,
        use_shared_network=True
    )
    
    print(f"Agent Deep PPO:")
    if agent.use_shared:
        params = sum(p.numel() for p in agent.network.parameters())
        print(f"   - Paramètres (Actor-Critic partagé): {params}")
    print(f"   - Device: {agent.device}")
    print(f"   - Clip epsilon: {agent.clip_epsilon}")
    
    # ========================================
    # 3. ENTRAÎNEMENT
    # ========================================
    
    num_episodes = 1000
    results = train_deep_PPO(
        env, 
        agent, 
        num_episodes=num_episodes,
        update_frequency=10,
        verbose=True
    )
    
    # ========================================
    # 4. SAUVEGARDE
    # ========================================
    
    save_dir = Path(__file__).parent.parent / 'results' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_dir / f'deep_ppo_{instance_name}.pth'))
    
    # ========================================
    # 5. COMPARAISON
    # ========================================
    
    section(f"📊 COMPARAISON")
    
    # SPT
    spt = SPTAgent(instance)
    spt_makespans = evaluate_agent(spt, env, num_episodes=10, verbose=False)
    
    # Charger DQN si existe
    dqn_path = save_dir / f'deep_dqn_{instance_name}.pth'
    if dqn_path.exists():
        from src.agents.deep_DQN_agent import DeepDQNAgent
        dqn_agent = DeepDQNAgent(state_size, action_size)
        dqn_agent.load(str(dqn_path))
        dqn_agent.epsilon = 0.0
        
        dqn_makespans = []
        for _ in range(10):
            state, _ = env.reset()
            done = False
            while not done:
                legal_actions = env._get_legal_actions()
                action = dqn_agent.select_action(state, legal_actions)
                state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            dqn_makespans.append(info['makespan'])
        
        dqn_best = min(dqn_makespans)
    else:
        dqn_best = None
    
    print(f"{'Méthode':<15} {'Meilleur':<10} {'Gap %':<10}")
    print("-" * 35)
    print(f"{'Optimal':<15} {optimal:<10} {0.0:<10.2f}")
    print(f"{'SPT':<15} {min(spt_makespans):<10.1f} {((min(spt_makespans)-optimal)/optimal*100):<10.2f}")
    if dqn_best:
        print(f"{'Deep DQN':<15} {dqn_best:<10.1f} {((dqn_best-optimal)/optimal*100):<10.2f}")
    print(f"{'Deep PPO':<15} {results['best_makespan']:<10.1f} {((results['best_makespan']-optimal)/optimal*100):<10.2f}")
    
    # ========================================
    # 6. VISUALISATIONS
    # ========================================
    
    print(f"\nVisualisations...")
    
    visualizer = GanttVisualizer()
    plots_dir = Path(__file__).parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Gantt
    visualizer.plot_schedule(
        results['best_schedule'],
        instance.num_machines,
        title=f"Deep PPO - {instance_name} (Makespan = {results['best_makespan']:.1f})",
        save_path=str(plots_dir / f'deep_ppo_{instance_name}_gantt.png')
    )
    
    # Courbe d'apprentissage
    rewards = [-m for m in results['makespans']]
    plot_training_curve(
        rewards,
        window_size=50,
        title=f"Apprentissage Deep PPO - {instance_name}",
        save_path=str(plots_dir / f'deep_ppo_{instance_name}_training.png')
    )
    
    # Losses PPO
    if results['policy_losses']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Policy Loss
        axes[0].plot(results['policy_losses'])
        axes[0].set_title('Policy Loss')
        axes[0].set_xlabel('Update')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Value Loss
        axes[1].plot(results['value_losses'])
        axes[1].set_title('Value Loss')
        axes[1].set_xlabel('Update')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # Entropy
        axes[2].plot(results['entropies'])
        axes[2].set_title('Entropy (Exploration)')
        axes[2].set_xlabel('Update')
        axes[2].set_ylabel('Entropy')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(plots_dir / f'deep_ppo_{instance_name}_losses.png'), dpi=300)
        
        plt.close()
    
    success("ENTRAÎNEMENT DEEP PPO TERMINÉ !")
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()