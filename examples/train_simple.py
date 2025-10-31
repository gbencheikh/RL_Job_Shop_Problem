# examples/train_simple.py
"""
Script d'entraînement simple avec DQN sur une petite instance.
Exécuter: python examples/train_simple.py
"""

import numpy as np
import torch
from src.environments.job_shop_instance import generate_random_instance
from src.environments.job_shop_env import JobShopEnv
from src.agents.DQN_agent import DQNAgent
from src.utils.logger import SimpleLogger

def main():
    # configuration minimale
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    instance = generate_random_instance(n_jobs=4, n_machines=3, max_ops_per_job=4, max_duration=8, seed=seed)
    env = JobShopEnv(instance)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, device="cpu")
    logger = SimpleLogger()

    n_episodes = 200
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            a = agent.act(obs)
            next_obs, r, done, _, _ = env.step(a)
            agent.remember(obs, a, r, next_obs, done)
            agent.train_step()
            obs = next_obs
            total_reward += r
        logger.info(f"Episode {ep} reward {total_reward:.3f} eps {agent.eps:.3f}")
    # save model
    agent.save("results/models/dqn_simple.pth")
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
