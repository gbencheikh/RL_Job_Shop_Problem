import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.utils.notifier import *
from src.agents.deep_DQN_agent import DeepDQNAgent, QNetwork
from src.agents.genetic_DQN_agent import GeneticPopulation
from src.environments.job_shop_instance import JobShopInstance
from src.environments.job_shop_env import JobShopEnv
from src.utils.logger import SimpleLogger
from src.utils.benchmark_instances import BenchmarkLibrary
from src.utils.visualization import GanttVisualizer

def train_genetic_population(env, population, generations: int = 20):
    
    """Entraîne une population génétique."""
    
    section("ENTRAÎNEMENT GENETIC DQN")
    
    logger = SimpleLogger()

    start_time = time.time()

    print(f"Population: {population.population_size}")
    print(f"Générations: {generations}")
    print(f"Élitisme: {population.elite_ratio*100}%")
    print(f"Mutation rate: {population.mutation_rate}")
    print("="*60)

    history = population.train_population(env, generations=generations)
    total_time = time.time() - start_time

    logger.info("Training finished.")
    passed("ENTRAÎNEMENT TERMINÉ")
    print(f"Temps total: {total_time:.2f}s")
    print(f"Meilleur makespan final: {history['best_fitness'][-1]:.2f}")
    print("="*60 + "\n")
    
    return history

if __name__ == "__main__":
    section("1. Crée une instance Job Shop")

    instance_name = 'FT06'
    print(f"Instance: {instance_name}")
    instance = BenchmarkLibrary.get_instance(instance_name)
    optimal = BenchmarkLibrary.get_optimal_makespan(instance_name)
    print(instance)
    print(f"Makespan optimal connu: {optimal}\n")

    # Afficher l'instance
    print(f"Instance:")
    print(instance)
    
    # ========================================
    # 2. Initialise l'environnement et l'agent DQN
    # ========================================
    section("2. Initialise l'environnement et l'agent genetic DQN")
    configuration() 

    # Créer l'environnement
    env = JobShopEnv(instance)

    # Créer l'agent genetic DQN
    population = GeneticPopulation(
        state_size= env.observation_space.shape[0],
        action_size=env.action_space.n,
        population_size=10
    )
    
    passed(f"Population créée:")
    print(f"   - Taille: {population.population_size}")
    print(f"   - Élitisme: {population.elite_ratio*100}%")
    print(f"   - Mutation rate: {population.mutation_rate}")
    
    section("4. Entraîne une population génétique")

    results = train_genetic_population(env, population, generations= 20) 
    
    best_agent = population.get_best_agent()

    section("Sauvegarde du modèle...")
    save_dir = Path(__file__).parent.parent / 'results' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    best_agent.save(str(save_dir / f'genetic_dqn_{instance_name}.pth'))


    # Résultats
    print("\n=== RÉSULTATS FINAUX ===")
    print("Makespan final :", results['best_fitness'][-1])
    
    print("\nOrdonnancement final:")
    schedule = env.get_schedule()
    for op in schedule:
        print(
            f"Job {op['job_id']} - Op {op['op_index']} | "
            f"Machine {op['machine_id']} | "
            f"Start: {op['start_time']:.1f} | End: {op['end_time']:.1f}"
        )

    print(f"Visualisations...")
    
    visualizer = GanttVisualizer()
    plots_dir = Path(__file__).parent.parent / 'results' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Gantt
    visualizer.plot_schedule(
        schedule,
        instance.num_machines,
        title=f"genetic Deep DQN - {instance_name} (Makespan = {results['best_fitness'][-1]:.1f})",
        save_path=str(plots_dir / f'genetic_deep_dqn_{instance_name}_gantt.png')
    )